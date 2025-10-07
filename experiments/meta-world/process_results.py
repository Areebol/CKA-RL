from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import pathlib
from tqdm import tqdm
import numpy as np
import os, sys
import argparse
from tabulate import tabulate

NUM_TASKS = 20

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs_all", type=str,
        help="directory where the tensorboard data is stored")
    parser.add_argument("--no-cache", default=False, action="store_true",
        help="wheter to disable the cache option. If not provided and `--save-dir` exists, skips processing tensorboard files")
    parser.add_argument("--save-csv", default="data/agg_results.csv", type=str,
        help="filename of the CSV to store the processed tensorboard results. Once processed, can be used as cache.")
    parser.add_argument("--tag", default="Debug", type=str)
    parser.add_argument("--smoothing-window", type=int, default=100)
    # fmt: on
    return parser.parse_args()


def parse_metadata(ea):
    md = ea.Tensors("hyperparameters/text_summary")[0]
    md_bytes = md.tensor_proto.SerializeToString()

    # remove first non-ascii characters and parse
    start = md_bytes.index(b"|")
    md_str = md_bytes[start:].decode("ascii")

    md = {}
    for row in md_str.split("\n")[2:]:
        s = row.split("|")[1:-1]
        k = s[0]
        if s[1].isdigit():
            v = int(s[1])
        elif s[1].replace(".", "").isdigit():
            v = float(s[1])
        elif s[1] == "True" or s[1] == "False":
            v = s[1] == "True"
        else:
            v = s[1]
        md[k] = v
    return md


def parse_tensorboard(path, scalars, single_pts=[]):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()

    # make sure the scalars are in the event accumulator tags
    if sum([s not in ea.Tags()["scalars"] for s in scalars]) > 0:
        print(f"** Scalar not found. Skipping file {path}")
        return None

    md = parse_metadata(ea)

    for name in single_pts:
        if name in ea.Tags()["scalars"]:
            md[name] = pd.DataFrame(ea.Scalars(name))["value"][0]

    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}, md


def smooth_avg(df, xkey, ykey, w=20):
    g = df.groupby(xkey)[ykey]
    mean = g.mean().rolling(window=w).mean()
    std = g.std().rolling(window=w).mean()

    y = mean.values
    y_std = std.values

    x = mean.reset_index()[xkey].values

    return x, y, y_std


def areas_up_down(method_x, method_y, baseline_x, baseline_y):
    up_idx = method_y > baseline_y
    down_idx = method_y < baseline_y

    assert (
        method_x == baseline_x
    ).all(), "The X axis of the baseline and method must be equal."
    x = method_x

    area_up = np.trapz(y=method_y[up_idx], x=x[up_idx]) - np.trapz(
        y=baseline_y[up_idx], x=x[up_idx]
    )
    area_down = np.trapz(y=baseline_y[down_idx], x=x[down_idx]) - np.trapz(
        y=method_y[down_idx], x=x[down_idx]
    )
    return area_up, area_down


def remove_nan(x, y):
    no_nan = ~np.isnan(y)
    return x[no_nan], y[no_nan]


def compute_forward_transfer(df, methods, smoothing_window):
    methods = methods.copy()
    methods.remove("simple")

    table = []
    results = {}
    for task_id in range(NUM_TASKS):
        # do not compute forward transfer for the first task
        if task_id == 0:
            table.append([0] + [None] * len(methods))
            continue

        baseline = df[
            (df["model_type"] == "simple") & (df["task_id"] == (task_id % 10))
        ]

        # get the curve of the `simple` method
        x_baseline, y_baseline, _ = smooth_avg(
            baseline, xkey="step", ykey="value", w=smoothing_window
        )
        x_baseline, y_baseline = remove_nan(x_baseline, y_baseline)

        baseline_area_down = np.trapz(y=y_baseline, x=x_baseline)
        baseline_area_up = np.max(x_baseline) - baseline_area_down

        table_row = [task_id]
        for j, name in enumerate(methods):
            method = df[(df["model_type"] == name) & (df["task_id"] == task_id)]
            x_method, y_method, _ = smooth_avg(
                method, xkey="step", ykey="value", w=smoothing_window
            )
            x_method, y_method = remove_nan(x_method, y_method)

            # this can happen if a method hasn't the results for all tasks
            if len(x_baseline) > len(x_method):
                table_row.append(None)
                continue

            area_up, area_down = areas_up_down(
                x_method,
                y_method,
                x_baseline,
                y_baseline,
            )
            if task_id == 18 and (name in ["componet"] or "fuse" in name):
                print(f"Method {name} task {task_id} up: {area_up} down: {area_down}")
                print(baseline_area_up)
            ft = (area_up - area_down) / baseline_area_up
            table_row.append(round(ft, 2))

            if methods[j] not in results.keys():
                results[methods[j]] = []
            results[methods[j]].append(ft)

        table.append(table_row)

    table.append([None] * len(table_row))

    row = ["Avg."]
    for i in range(len(methods)):
        vals = []
        for r in table[1:]:
            v = r[i + 1]
            if v is not None:
                vals.append(v)
        mean = np.mean(vals)
        std = np.std(vals)
        row.append((round(mean, 4), round(std, 4)))
    table.append(row)

    print("\n== FORWARD TRANSFER ==")
    print(
        tabulate(
            table,
            headers=["Task ID"] + methods,
            tablefmt="rounded_outline",
        )
    )
    print("\n")
    return results


def compute_performance(df, methods):
    table = []
    avgs = [[] for _ in range(len(methods))]
    results = {}
    eval_steps = df["step"].unique()[-10]
    for i in range(NUM_TASKS):
        row = [i]
        for j, m in enumerate(methods):
            task_id = i if m != "simple" else i % 10

            method = "simple" if task_id == 0 and m in ["componet", "finetune", "fusenet_merge"] else m

            s = df[
                (df["task_id"] == task_id)
                & (df["model_type"] == method)
                & (df["step"] >= eval_steps)
            ]["value"].values

            if len(s) == 0:
                s = np.array([np.nan])

            if methods[j] not in results.keys():
                results[methods[j]] = []
            results[methods[j]].append((s.mean(), s.std()))

            avg = round(s.mean(), 2)
            std = round(s.std(), 2)

            row.append((avg, std))
            if not np.isnan(s).any() and i > 0:
                avgs[j] += list(s)

        table.append(row)

    avgs = [(round(np.mean(v), 6), round(np.std(v), 2)) for v in avgs]
    table.append([None] * len(row))
    table.append(["Avg."] + avgs)

    for i in range(len(table)):
        for j in range(len(table[0])):
            e = table[i][j]
            if type(e) != tuple:
                continue
            if np.isnan(e[0]):
                table[i][j] = None
    print("\n== PERFORMANCE ==")
    print(
        tabulate(
            table,
            headers=["Task ID"] + methods,
            tablefmt="rounded_outline",
        )
    )
    print("\n")

    return results


def count(df, methods):
    counts = (
        df.groupby(["task_id", "model_type"])["seed"].unique().apply(lambda x: len(x))
    )
    counts = counts.reset_index()
    table = []
    all_vals = []
    for task_id in range(NUM_TASKS):
        row = [task_id]
        for method in methods:
            c = counts[
                (counts["task_id"] == task_id) & (counts["model_type"] == method)
            ]
            val = c["seed"].iloc[0] if not c.empty else None
            row.append(val)
            if val is not None:
                all_vals.append(val)
        table.append(row)

    s = sum(all_vals)
    total = len(methods) * 20 * 10 - (12 * 10)
    print(f"\n\n-----------------------------------")
    print(f" Total percentage: {round(100*s/total, 3)}% [{s}/{total}]")
    print(f"-----------------------------------\n\n")

    print(
        tabulate(
            table,
            headers=["Task ID"] + methods,
            tablefmt="rounded_outline",
        )
    )


def process_eval(df, perf_data):
    perf = {}
    forg = {}
    for method in df["algorithm"].unique():
        s = df[df["algorithm"] == method].groupby("test task")["success"]
        avgs = s.mean().reset_index().sort_values(by=["test task"])
        stds = s.std().reset_index().sort_values(by=["test task"])
        perf[method] = {}
        perf[method]["avgs"] = list(avgs["success"])
        perf[method]["stds"] = list(stds["success"])

        print(f"** Eval performance: {method}")
        [
            print(f"({round(m, 2)}, {round(s, 2)})", end=" ")
            for m, s in zip(perf[method]["avgs"], perf[method]["stds"])
        ]
        print(
            f"\nAvg. and std.: {round(np.mean(perf[method]['avgs']), 2)} {round(np.std(perf[method]['avgs']), 2)} "
        )
        print()

        all_forgs = []
        forg[method] = []
        for i in range(len(avgs)):
            prev = perf_data[method][i][0]
            lasts = df[(df["algorithm"] == method) & (df["test task"] == i)][
                "success"
            ].values

            f = prev - lasts
            all_forgs += list(f)
            forg[method].append((np.mean(prev - lasts), np.std(prev - lasts)))

        print(f"** Eval forgetting: {method}")
        [print(f"{round(m, 2)} [{round(s, 2)}]", end=" ") for m, s in forg[method]]
        print(
            f" => Avg. and std.: {round(np.mean(all_forgs), 2)} [{round(np.std(all_forgs), 2)}]"
        )
        print("\n")

    return perf, forg


if __name__ == "__main__":
    sys.path.append("../../")

    args = parse_args()
    METHOD_NAMES = {
        "simple": "Baseline",
        "finetune": "FT",
        "componet": "CompoNet",
        "prognet": "ProgressiveNet",
        "packnet": "PackNet",
        "cka_rl": "CKA-RL",
        "masknet": "MaskNet",
        "crelus": "CRELUS",
    }
    # hardcoded settings
    scalar = "charts/success"
    final_success = "charts/test_success"
    total_timesteps = 1e6
    methods = ["simple", "componet", "finetune", "prognet", "packnet", "cka-rl", "masknet", "cbpnet", "crelus"]
    fancy_names = ["Baseline", "CompoNet", "FT", "ProgressiveNet", "PackNet", "CKA-RL", "MaskNet", "CReLUs"]
    method_colors = ["darkgray", "tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]

    #
    # Extract data from tensorboard results to an actually useful CSV
    #
    args.save_csv = f"data/{args.tag}/agg_results.csv"
    exists = os.path.exists(args.save_csv)
    if args.no_cache or (not exists and not args.no_cache):
        dfs = []
        for path in tqdm(list(pathlib.Path(args.runs_dir).rglob("*events.out*"))):
            # print("*** Processing ", path)
            res = parse_tensorboard(str(path), [scalar], [final_success])
            if res is not None:
                dic, md = res
            else:
                print("No data. Skipping...")
                continue

            df = dic[scalar]
            df = df[["step", "value"]]
            df["seed"] = md["seed"]
            df["task_id"] = md["task_id"]
            df["model_type"] = md["model_type"]

            if final_success in md:
                df[final_success] = md[final_success]

            dfs.append(df)
        df = pd.concat(dfs)
        df.to_csv(args.save_csv, index=False)
    else:
        print(f"\n\nReloading cache data from: {args.save_csv}")
        df = pd.read_csv(args.save_csv)

    #
    # Compute performance and forward transfer
    #
    count(df, methods)

    data_perf = compute_performance(df, methods, args.fuse_type)
    ft_data = compute_forward_transfer(df, methods, args.smoothing_window)