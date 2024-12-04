TASKS = {
    "ALE/SpaceInvaders-v5": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "ALE/Freeway-v5": [0, 1, 2, 3, 4, 5, 6, 7],
}


def parse_name_info(name):
    fields = name.split("_")
    print(fields)
    if "SpaceInvaders" in fields[0]:
        env = "ALE/SpaceInvaders-v5"
    elif "Freeway" in fields[0]:
        env = "ALE/Freeway-v5"
    mode = int(fields[1])
    algorithm = fields[2]
    if len(fields) > 3:
        algorithm += "_" + fields[3]
    # seed = int(fields[3])
    # return env, mode, algorithm, seed
    return env, mode, algorithm


def path_from_other_mode(base_path, new_mode):
    sep_idx = base_path.index("_")
    double_sep_idx = base_path.rindex("_")
    new_path = base_path[: sep_idx + 1] + str(new_mode) + base_path[double_sep_idx:]
    return new_path

def get_method_type(args):
    if args.method_type == "baseline":
        return "F1"
    elif args.method_type == "finetune":
        return "FN"
    elif args.method_type == "componet":
        return "CompoNet"
    elif args.method_type == "packnet":
        return "PackNet"
    elif args.method_type == "prognet":
        return "ProgNet"
    elif args.method_type == "tv_1":
        return "TV1"
    elif args.method_type == "tv_2":
        return "TV2"
    else:
        return None