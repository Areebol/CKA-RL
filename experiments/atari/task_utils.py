TASKS = {
    "ALE/SpaceInvaders-v5": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "ALE/Freeway-v5": [0, 1, 2, 3, 4, 5, 6, 7],
}


def parse_name_info(name):
    fields = name.split("_")
    print(fields)
    if "Freeway" in fields[0]:
        env = "ALE/Freeway-v5"
    else:
        env = "ALE/SpaceInvaders-v5"
        
    mode = int(fields[1])
    algorithm = fields[2]
    seed = int(fields[3])
    return env, mode, algorithm, seed
    # return env, mode, algorithm


def path_from_other_mode(base_path, new_mode):
    results = base_path.split("/")
    a = results[-1]
    a_parts = a.split("_")
    seed = a_parts[-1]
    method = a_parts[-2]
    mode = a_parts[-3]
    env = a_parts[-4]
    new_path = "/".join(results[:-1]) + "/" + env + "_" + str(new_mode) + "_" + method + "_" + seed
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
    elif args.method_type == "tvnet":
        return "TVNet"
    else:
        return None