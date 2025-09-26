import gymnasium as gym
import numpy as np
import torch
import argparse
import os
import time
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    MaxAndSkipEnv,
)
from torch.profiler import profile, record_function, ProfilerActivity

from models import (
    CnnSimpleAgent,
    CnnCompoNetAgent,
    ProgressiveNetAgent,
    PackNetAgent,
    FuseNetAgent,
    CnnMaskAgent,
    FuseNetwMergeAgent,
    RewireAgent,
    CnnCbpAgent,
    CReLUsAgent,
)
from task_utils import parse_name_info, path_from_other_mode


def parse_arguments():
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--load", type=str, default="/lichenghao/lzh/CRL/componet/experiments/atari/agents/SpaceInvaders/ModelZoo/SpaceInvaders_9_FuseNet_42")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=int, default=None)

    parser.add_argument("--train_mode", type=int)
    parser.add_argument("--max-timesteps", type=int, default=1000)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--csv', default=None, type=str)
    # fmt: on

    return parser.parse_args()


def make_env(env_id, idx, run_name, render_mode=None, mode=None):
    def thunk():
        env = gym.make(env_id, mode=mode, render_mode=render_mode)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        # env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        # env = EpisodicLifeEnv(env)

        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk

def convert_algorithm(algorithm):
    if algorithm not in ["Baseline", "Finetune"]:
        return algorithm
    conversion_dict = {
        "Baseline": "F1",
        "Finetune": "FN",
    }
    return conversion_dict.get(algorithm, "unknown")

def get_num_task(env):
    if "SpaceInvaders" in env:
        return 10
    elif "Freeway" in env:
        return 8

def estimate_memory(model, input_shape=(1, 4, 84, 84), dtype=torch.float32):
    dummy_input = torch.randn(*input_shape, device='cuda', dtype=dtype)

    model = model.to(dtype=dtype, device='cuda')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        model.get_action_and_value(dummy_input)

    total_mem = torch.cuda.max_memory_allocated()
    total_mem_MB = total_mem / (1024 ** 2)

    param_size_MB = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

    return {
        'Model param size (MB)': param_size_MB,
        'Peak memory during forward (MB)': total_mem_MB,
        'Activation memory (approx)': total_mem_MB - param_size_MB
    }

if __name__ == "__main__":
    args = parse_arguments()
    headers = ["Task.ID", "Infer.Time", "Param.Mem", "Act.Mem"]
    results = []
    
    for i in range(0,10):
        # env_name, train_mode, algorithm, seed = parse_name_info(args.load.split("/")[-1])
        # load = f"/lichenghao/lzh/workspace/CRL/componet/experiments/atari/agents/SpaceInvaders/ModelZoo/SpaceInvaders_{i}_FuseNet_42"
        # load = f"/lichenghao/lzh/workspace/CRL/componet/experiments/atari/agents/SpaceInvaders/ModelZoo/SpaceInvaders_{i}_CompoNet_42"
        load = f"/lichenghao/lzh/workspace/CRL/componet/experiments/atari/agents/SpaceInvaders/ModelZoo/SpaceInvaders_{i}_ProgNet_42"
        # load = f"/lichenghao/lzh/workspace/CRL/componet/experiments/atari/agents/Freeway/ModelZoo/Freeway_{i}_ProgNet_42"
        # load = f"/lichenghao/lzh/workspace/CRL/componet/experiments/atari/agents/Freeway/ModelZoo/Freeway_{i}_CompoNet_42"
        # load = f"/lichenghao/lzh/workspace/CRL/componet/experiments/atari/agents/Freeway/ModelZoo/Freeway_{i}_FuseNet_42"
        env_name, train_mode, algorithm, seed = parse_name_info(load.split("/")[-1])
        mode = train_mode if args.mode is None else args.mode
        seed = args.seed
        # print(
            # f"\nEnvironment: {env_name}, train/test mode: {train_mode}/{mode}, algorithm: {algorithm}, seed: {seed}\n"
        # )

        # make the environment
        envs = gym.vector.SyncVectorEnv([make_env(env_name, 1, run_name="test", mode=mode)])
        env_fn = make_env(
            env_name,
            0,
            run_name="test",
            mode=mode,
            render_mode="human" if args.render else None,
        )
        env = env_fn()

        # load the model
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if algorithm in ["F1", "FN", "Baseline", "Finetune"]:
            agent = CnnSimpleAgent.load(
                load, envs, load_critic=False, map_location=device
            )
        elif algorithm == "CompoNet":
            prevs_paths = [path_from_other_mode(load, j) for j in range(i)]
            agent = CnnCompoNetAgent.load(
                load, envs, prevs_paths=prevs_paths, map_location=device
            )
        elif algorithm == "ProgNet":
            prevs_paths = [path_from_other_mode(load, j) for j in range(i)]
            agent = ProgressiveNetAgent.load(
                load, envs, prevs_paths=prevs_paths, map_location=device
            )
        elif algorithm == "PackNet":
            task_id = None if args.mode == None else args.mode + 1
            agent = PackNetAgent.load(load, task_id=task_id, map_location=device)
            agent.network.set_view(task_id)

            if mode != train_mode:
                # load the actor and critic heads from the model trained in the testing task (game mode)
                path = path_from_other_mode(load, mode)
                ac = PackNetAgent.load(path, map_location=device)
                agent.critic = ac.critic
                agent.actor = ac.actor
        elif algorithm == "FuseNet":
            agent = FuseNetAgent.load(load, envs, map_location=device)
        elif algorithm == "FuseNetwMerge":
            agent = FuseNetwMergeAgent.load(load, envs, map_location=device)
        elif algorithm == "MaskNet":
            agent = CnnMaskAgent.load(load, envs, num_tasks=get_num_task(env_name), map_location=device)
        elif algorithm == "Rewire":
            agent = RewireAgent.load(load, envs, map_location=device)
            agent.set_task(args.mode)
        elif algorithm == "CReLUs":
            agent = CReLUsAgent.load(load, envs, map_location=device)
        elif algorithm == "CbpNet":
            agent = CnnCbpAgent.load(load, envs, map_location=device)
        else:
            print(f"Loading of agent type `{algorithm}` is not implemented.")
            quit(1)

        agent.to(device)

        #
        # Main loop
        # ~~~~~~~~~
        loop_times = 10000
        costs = []
        
        for _ in range(loop_times):
            # observation = torch.from_numpy(np.array(observation)).to(device) / 255.0
            # observation = observation.unsqueeze(0)
            with torch.no_grad():
                agent.eval()
                observation = torch.randn([1,4,84,84]).to("cuda")
                start = time.time()
                action, _, _, _ = agent.get_action_and_value(observation)
                end = time.time()
                cost = end - start
                costs.append(cost)
        mem_info = estimate_memory(model=agent)
        avg_cost = sum(costs) / len(costs)
        print(f"Avg. cost of mode {i} is {avg_cost}")
        print(mem_info)
        results.append([i, avg_cost, mem_info['Model param size (MB)'], mem_info['Activation memory (approx)']])

    import csv
    output_file = f"./{algorithm}_infer_mem.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # 写入表头
        for row in results:
            formatted_row = [row[0]] + ["%.4f" % x for x in row[1:]]
            writer.writerow(formatted_row)

    print(f"结果已保存到 {output_file}")
    
    # print("Avg. episodic return:", np.mean(ep_rets))

    # if args.csv:
    #     exists = os.path.exists(args.csv)
    #     with open(args.csv, "w" if not exists else "a") as f:
    #         if not exists:
    #             f.write("algorithm,environment,train mode,test mode,seed,ep ret\n")
    #         for v in ep_rets:
    #             f.write(f"{convert_algorithm(algorithm)},{env_name},{train_mode},{mode},{seed},{v}\n")
