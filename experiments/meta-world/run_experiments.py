import subprocess
import argparse
import random
from tasks import tasks


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=[
            "simple",
            "componet",
            "finetune",
            "from-scratch",
            "prognet",
            "packnet",
            "fusenet",
            "fusenet_merge",
            "masknet",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-run", default=False, action="store_true")

    parser.add_argument("--start-mode", type=int, default=0)
    parser.add_argument("--tag", type=str, default="Debug")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fuse_shared", action="store_true")
    parser.add_argument("--fuse_heads", action="store_true")
    parser.add_argument("--pool_size", default=4)
    parser.add_argument("--encoder_from_base", action="store_true")
    return parser.parse_args()


args = parse_args()

modes = list(range(20)) if args.algorithm != "simple" else list(range(10))
# args.start_mode = 3
# NOTE: If the algoritm is not `simple`, it always should start from the second task
if args.algorithm not in ["simple", "packnet", "prognet", "fusenet", "fusenet_merge", "masknet"] and args.start_mode == 0:
    start_mode = 1
else:
    start_mode = args.start_mode

run_name = (
    lambda task_id: f"task_{task_id}__{args.algorithm if task_id > 0 or args.algorithm in ['packnet', 'prognet', 'fusenet', 'fusenet_merge', 'masknet'] else 'simple'}__run_sac__{args.seed}"
)

first_idx = modes.index(start_mode)
for i, task_id in enumerate(modes[first_idx:]):
    params = f"--model-type={args.algorithm} --task-id={task_id} --seed={args.seed} --tag={args.tag}"
    if args.fuse_shared:
        params += " --fuse-shared" 
    else:
        params += " --no-fuse-shared"
    if args.fuse_heads:
        params += " --fuse-heads" 
    else:
        params += " --no-fuse-heads"
    if args.debug:
        params += " --total-timesteps=1"
    if args.encoder_from_base:
        params += " --encoder-from-base"
    else:
        params += " --no-encoder-from-base"
    
    save_dir = f"agents/{args.tag}"
    params += f" --save-dir={save_dir}"
    params += f" --pool_size={args.pool_size}"

    if first_idx > 0 or i > 0:
        # multiple previous modules
        if args.algorithm in ["componet", "prognet", "fusenet", "fusenet_merge"]:
            params += " --prev-units"
            for i in modes[: modes.index(task_id)]:
                params += f" {save_dir}/{run_name(i)}"
        # single previous module
        elif args.algorithm in ["finetune", "packnet", "masknet"]:
            params += f" --prev-units {save_dir}/{run_name(task_id-1)}"

    # Launch experiment
    cmd = f"python3 run_sac.py {params}"
    print(cmd)

    if not args.no_run:
        res = subprocess.run(cmd.split(" "))
        if res.returncode != 0:
            print(f"*** Process returned code {res.returncode}. Stopping on error.")
            quit(1)
