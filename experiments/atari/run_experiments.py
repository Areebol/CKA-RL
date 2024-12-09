import subprocess
import argparse
import random
from task_utils import TASKS, get_method_type
from icecream import ic
    
method_choices = ["baseline",         # F1
                  "finetune",         # FN
                  "componet",         # CompoNet
                  "packnet",          # PackNet
                  "prognet",          # ProgNet
                  "tv_1",             # TV1 Task-Vector-1: Do Task-Vector on Encoder & Actor both
                  "tv_2",             # TV1 Task-Vector-1: Do Task-Vector on Encoder only
                  "fuse_1",           # Fuse 1: Do fuse to Actor only
                  "fuse_2",           # Fuse 2: Do fuse to Actor only + Fix alpha
                  "fuse_3",           # Fuse 1: Do fuse to Actor only + `add Delta theta_0 = 0` + `large alpha's learning rate`
                  ]
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument("--method_type", type=str, choices=method_choices, required=True)
    parser.add_argument("--env", type=str, choices=["ALE/SpaceInvaders-v5", "ALE/Freeway-v5"], default="ALE/Freeway-v5")
    parser.add_argument("--seed", type=int, required=False, default=42)

    parser.add_argument("--first-mode", type=int, required=True)
    parser.add_argument("--last-mode", type=int, required=True)
    parser.add_argument("--debug", type=bool, default=False)
    
    # fmt: on
    return parser.parse_args()


args = parse_args()
print(args)
modes = TASKS[args.env]
first_mode = args.first_mode
last_mode = args.last_mode
debug = args.debug
method_type = args.method_type


seed = random.randint(0, 1e6) if args.seed is None else args.seed

env_name = args.env.split("/")[1].split("-")[0] # e.g. ALE/Freeway-v5 -> Freeway
run_name = (
    lambda task_id: f"{env_name}_{task_id}_{get_method_type(args)}" # e.g. Freeway_1_FN、Freeway_2_TV_1
)

first_idx = modes.index(first_mode)
last_idx = modes.index(last_mode)

for i, task_id in enumerate(modes[first_idx:last_idx+1]):
    # params
    save_dir = f"agents/{env_name}"
    params = f"--method-type={method_type} --env-id={args.env} --seed={seed}"
    params += f" --mode={task_id} --save-dir={save_dir}"
    
    # debug
    params += (" --track" if not debug else " --no-track")
    if debug:
        params += f" --total-timesteps=10000"

    # method specific CLI arguments
    if args.method_type == "componet":
        params += " --componet-finetune-encoder"
    if args.method_type == "packnet":
        params += f" --total-task-num={len(modes)}"

    if first_idx > 0 or i > 0:
        # multiple previous modules
        if args.method_type in ["componet", "prognet", "tv_1", "tv_2", "fuse_1", "fuse_2", "fuse_3"]:
            params += " --prev-units"
            for i in modes[: modes.index(task_id)]:
                params += f" {save_dir}/{run_name(i)}"
        # single previous module
        elif args.method_type in ["finetune", "packnet"]:
            params += f" --prev-units {save_dir}/{run_name(task_id-1)}"
            
    # Launch experiment
    cmd = f"python3 run_ppo.py {params}"
    print(cmd)
    res = subprocess.run(cmd.split(" "))
    if res.returncode != 0:
        print(f"*** Process returned code {res.returncode}. Stopping on error.")
        quit(1)
