import subprocess
import argparse
import random
from task_utils import TASKS
from icecream import ic

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    # parser.add_argument("--algorithm", type=str, choices=["componet", "finetune", "from-scratch", "prog-net", "packnet", "tvnet"], required=True)
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--env", type=str, choices=["ALE/SpaceInvaders-v5", "ALE/Freeway-v5"], default="ALE/SpaceInvaders-v5")
    parser.add_argument("--seed", type=int, required=False, default=42)

    parser.add_argument("--start-mode", type=int, required=True)
    parser.add_argument("--first-mode", type=int, required=True)
    parser.add_argument("--last-mode", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=1e6)
    parser.add_argument("--debug", type=bool, default=False)
    
    # fmt: on
    return parser.parse_args()


args = parse_args()

modes = TASKS[args.env]
timesteps = args.timesteps
start_mode = args.start_mode
first_mode = args.first_mode
last_mode = args.last_mode
debug = args.debug

model_type_map = {
    "finetune": "cnn-simple-ft",
    "componet": "cnn-componet",
    "from-scratch": "cnn-simple",
    "prog-net": "prog-net",
    "packnet": "packnet",
    "tvnet": "cnn-tvnet", # Add tvnet
    "cnn-tvnet-fte": "cnn-tvnet-fte",
}

model_type = model_type_map.get(args.algorithm)

seed = random.randint(0, 1e6) if args.seed is None else args.seed

run_name = (
    lambda task_id: f"{args.env.replace('/', '-')}_{task_id}__{model_type}__run_ppo__{seed}"
)

first_idx = modes.index(start_mode)
last_idx = modes.index(last_mode)
# ic(first_idx)
# ic(last_idx)
# ic(modes)
for i, task_id in enumerate(modes[first_idx:last_idx+1]):
    params = f"--track --model-type={model_type} --env-id={args.env} --seed={seed}"
    params += f" --mode={task_id} --save-dir=agents --total-timesteps={timesteps}"

    # algorithm specific CLI arguments
    if args.algorithm == "componet":
        params += " --componet-finetune-encoder"
    if args.algorithm == "packnet":
        params += f" --total-task-num={len(modes)}"

    if first_idx > 0 or i > 0:
        # multiple previous modules
        if args.algorithm in ["componet", "prog-net", "tvnet"]:
            params += " --prev-units"
            for i in modes[: modes.index(task_id)]:
                params += f" agents/{run_name(i)}"
        # single previous module
        elif args.algorithm in ["finetune", "packnet"]:
            params += f" --prev-units agents/{run_name(task_id-1)}"
            
    params += f" --wandb_mode=" + ("online" if not debug else "offline")

    # Launch experiment
    cmd = f"python3 run_ppo.py {params}"
    print(cmd)
    res = subprocess.run(cmd.split(" "))
    if res.returncode != 0:
        print(f"*** Process returned code {res.returncode}. Stopping on error.")
        quit(1)
