import subprocess
import argparse
import random
from task_utils import TASKS
from icecream import ic
from loguru import logger
    
method_choices = ["Baseline",         # F1
                  "Finetune",         # FN
                  "CompoNet",         # CompoNet
                  "PackNet",          # PackNet
                  "ProgNet",          # ProgNet
                  "TvNet",            # TV1 Task-Vector-1: Do Task-Vector on Encoder & Actor both
                  "FuseNet",          # FuseNet
                  ]
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method_type", type=str, choices=method_choices, required=True)
    parser.add_argument("--env", type=str, choices=["ALE/SpaceInvaders-v5", "ALE/Freeway-v5"], default="ALE/Freeway-v5")
    parser.add_argument("--seed", type=int, required=False, default=42)

    parser.add_argument("--first-mode", type=int, required=True)
    parser.add_argument("--last-mode", type=int, required=True)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--tag", type=str, default="Debug")
    parser.add_argument("--alpha_factor", type=float, default=None)
    parser.add_argument("--fix_alpha", type=bool, default=False)
    parser.add_argument("--alpha_learning_rate", type=float, default=2.5e-4)
    
    return parser.parse_args()


args = parse_args()
logger.info(f"experiments args : {args}")
modes = TASKS[args.env]
first_mode = args.first_mode
last_mode = args.last_mode
debug = args.debug
method_type = args.method_type


seed = random.randint(0, 1e6) if args.seed is None else args.seed

env_name = args.env.split("/")[1].split("-")[0] # e.g. ALE/Freeway-v5 -> Freeway
run_name = (
    lambda task_id: f"{env_name}_{task_id}_{args.method_type}_{args.seed}" # e.g. Freeway_1_FN、Freeway_2_TV_1
)

first_idx = modes.index(first_mode)
last_idx = modes.index(last_mode)

for i, task_id in enumerate(modes[first_idx:last_idx+1]):
    # params
    save_dir = f"agents/{env_name}/{args.tag}"
    params = f"--method-type={method_type} --env-id={args.env} --seed={seed}"
    params += f" --mode={task_id}"
    params += f" --tag={args.tag}"
    if args.alpha_factor is not None:
        params += f" --alpha_factor={args.alpha_factor}"
    if args.fix_alpha:
        params += f" --fix_alpha"
    params += f" --alpha_learning_rate={args.alpha_learning_rate}"
        
    # debug mode
    params += (" --track" if not debug else " --no-track")
    if debug:
        logger.debug(f"Running experiment within bugging mode")
        params += f" --total-timesteps=3000"
        
    # method specific CLI arguments
    if args.method_type == "CompoNet":
        params += " --componet-finetune-encoder"
    if args.method_type == "PackNet":
        params += f" --total-task-num={len(modes)}"

    if first_idx > 0 or i > 0:
        # multiple previous modules
        if args.method_type in ["CompoNet", "ProgNet", "TvNet", "FuseNet"]:
            params += " --prev-units"
            logger.info(f"Method {args.method_type} need prevs-units, adding prevs-units")
            for i in modes[: modes.index(task_id)]:
                params += f" {save_dir}/{run_name(i)}"
        # single previous module
        elif args.method_type in ["Finetune", "PackNet"]:
            params += f" --prev-units {save_dir}/{run_name(task_id-1)}"
            
    # Launch experiment
    cmd = f"python run_ppo.py {params}"
    logger.info(f"Running experiment script: {cmd}")
    res = subprocess.run(cmd.split(" "))
    if res.returncode != 0:
        logger.error(f"Process {cmd} \n returned code {res.returncode}. Stopping on error.")
        quit(1)
