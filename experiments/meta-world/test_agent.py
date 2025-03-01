import gymnasium as gym
import numpy as np
import torch
from models import *
from tasks import get_task, get_task_name
import argparse
import os
from run_sac import Actor
import random

def path_from_other_mode(base_path, new_mode, method, seed):
    results = base_path.split("/")
    new_path = "/".join(results[:-1]) + "/" + "task_" + str(new_mode) + "__" + method + "__run_sac__" + str(seed)
    return new_path

def parse_args():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--load", type=str, required=True)
    
    parser.add_argument("--task-id", type=int, required=False, default=None)
    parser.add_argument('--train-task', type=int, default=0)
    parser.add_argument("--seed", type=int, required=False, default=None)
    parser.add_argument('--csv', default=None, type=str)
    
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument('--render', default=False, action='store_true')
    # fmt: on

    return parser.parse_args()


def parse_load_name(path):
    name = os.path.basename(path)
    s = name.split("__")
    method = s[1]
    seed = int(s[-1])
    task = int(s[0].split("_")[-1])
    return method, seed, task


def make_env(task_id, render_human=False):
    def thunk():
        env = get_task(task_id)
        if render_human:
            env.render_mode = "human"
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


@torch.no_grad()
def eval_agent(agent, test_env, num_evals, device):
    obs, _ = test_env.reset()
    ep_rets = []
    successes = []
    ep_ret = 0
    for _ in range(num_evals):
        while True:
            obs = torch.Tensor(obs).to(device).unsqueeze(0)
            action, _ = agent(obs)
            action = action[0].cpu().numpy()
            obs, reward, termination, truncation, info = test_env.step(action)

            ep_ret += reward

            if termination or truncation:
                successes.append(info["success"])
                ep_rets.append(ep_ret)
                print(ep_ret, successes[-1])
                # resets
                obs, _ = test_env.reset()
                ep_ret = 0
                break

    print(f"\nTEST: ep_ret={np.mean(ep_rets)}, success={np.mean(successes)}\n")
    return successes


if __name__ == "__main__":
    args = parse_args()
    method = args.method
    seed = args.seed
    train_task = args.train_task
    task_id = args.task_id
    prev_paths = [path_from_other_mode(args.load, i, method, seed) for i in range(args.train_task)]

    print(
        f"Method: {method}, seed: {seed}, train task: {train_task}, test task: {task_id}, save-csv: {args.csv}"
    )
    print(prev_paths)
    envs = gym.vector.SyncVectorEnv([make_env(task_id, render_human=args.render)])
    env = envs.envs[0]
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = np.prod(envs.single_action_space.shape)
    
    # set the seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if method in ["simple", "finetune"]:
        model = SimpleAgent.load(args.load, map_location=device)
    elif method == "prognet":
        model = ProgressiveNetAgent.load(
            dirname=args.load, 
            obs_dim=obs_dim, 
            act_dim=act_dim, prev_paths=prev_paths, map_location=device
        ).to(device)
    elif method == "packnet":
        model = PackNetAgent.load(
            args.load, 
            task_id=task_id,
            map_location=device).to(device)
    elif method == "masknet":
        model = MaskNetAgent.load(
            args.load, 
            map_location=device).to(device)
        model.set_task(task_id,new_task=False)
    elif method == "creus":
        model = CReLUsAgent.load(
            args.load, 
            map_location=device).to(device)
    elif method == "rewire":
        pass
    elif method == "componet":
        if len(prev_paths) > 0:
            prev_paths[0] = path_from_other_mode(args.load, 0, "simple", seed)
        model = CompoNetAgent.load(
            dirname=args.load,
            obs_dim=obs_dim,
            act_dim=act_dim,
            prev_paths=prev_paths,
            map_location=device).to(device)
    elif method == "cbpnet":
        model = CbpAgent.load(
            args.load,
            map_location=device).to(device)
    elif method == "fusenet":
        model = FuseNetAgent.load(
            args.load,
            map_location=device).to(device)
    elif method == "fusenet_merge":
        model = FuseMergeNetAgent.load(
            args.load,
            obs_dim=obs_dim, 
            act_dim=act_dim, 
            base_dir=None,
            latest_dir=None,
            map_location=device).to(device)
    

    agent = Actor(envs, model)

    successes = eval_agent(agent, env, num_evals=args.num_episodes, device=device)

    if args.csv:
        exists = os.path.exists(args.csv)
        with open(args.csv, "w" if not exists else "a") as f:
            if not exists:
                f.write("algorithm,test task,train task,seed,success\n")
            for v in successes:
                f.write(f"{method},{task_id},{train_task},{seed},{v}\n")
