import gymnasium as gym
import numpy as np
import torch
import argparse
import os
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    MaxAndSkipEnv,
)

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
    parser.add_argument("--load", type=str, required=True)

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
    
def get_actor(load, envs, device, algorithm="F1"):
    if algorithm in ["F1", "FN", "Baseline", "Finetune"]:
        agent = CnnSimpleAgent.load(
            load, envs, load_critic=False, map_location=device
        )
    elif algorithm == "FuseNet":
        agent = FuseNetAgent.load(
            load, envs, load_critic=False, map_location=device
        )
    return agent.actor

import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

def get_model_weights(model: nn.Module) -> np.ndarray:
    """ 获取模型的所有参数拼接成一个一维 numpy 数组 """
    params = []
    for name, param in model.named_parameters():
        if name in ["0.weight", "2.weight"]:  # Only include parameters with "weight" in their name
            params.append(param.data.cpu().numpy().ravel())
    return np.concatenate(params)

def cosine_sim(model_a: nn.Module, model_b: nn.Module) -> float:
    """ 计算两个模型之间的余弦相似度 """
    a = get_model_weights(model_a)
    b = get_model_weights(model_b)
    return cosine_similarity([a], [b])[0][0]

def get_weight_vectors(model: torch.nn.Module, base: np.array):
    """
    提取一个 actor 中的 10 个 weight 矩阵，并分别展平为向量。
    返回一个列表，包含 10 个向量。
    """
    vectors = []
    sd = model.state_dict()

    # 9 个 adaptive 权重
    w_multi_1 = sd['network.0.weights'].cpu().numpy()  # shape: [9, 512, 512]
    w_multi_2 = sd['network.2.weights'].cpu().numpy()  # shape: [9, 512, 512]
    for i in range(w_multi_1.shape[0]):
        vectors.append(np.concatenate((w_multi_1[i], w_multi_2[i]), axis=0).ravel())
        

    w_main_1 = sd['network.0.weight'].cpu().numpy()  # shape: [512, 512]
    w_main_2 = sd['network.2.weight'].cpu().numpy()  # shape: [512, 512]
    w_main = np.concatenate((w_main_1, w_main_2), axis=0)  # shape: [1024, 512]
    # vectors.append(w_main.ravel())
    vectors.append(w_main.ravel() - base)
    
    assert len(vectors) == 10, f"Expected 10 weight matrices, got {len(vectors)}"

    return vectors

def get_base(model: torch.nn.Module):
    """
    提取一个 actor 中的 10 个 weight 矩阵，并分别展平为向量。
    返回一个列表，包含 10 个向量。
    """
    vectors = []
    sd = model.state_dict()

    w_main_1 = sd['network.0.weight'].cpu().numpy()  # shape: [512, 512]
    w_main_2 = sd['network.2.weight'].cpu().numpy()  # shape: [512, 512]
    base = np.concatenate((w_main_1, w_main_2), axis=0).ravel()  # shape: [1024, 512]

    return base

if __name__ == "__main__":
    env_name = "ALE/Freeway-v5"
    train_mode = 0
    algorithm = "FN"
    seed = 42
    mode = 0
    print(
        f"\nEnvironment: {env_name}, train/test mode: {train_mode}/{mode}, algorithm: {algorithm}, seed: {seed}\n"
    )

    # make the environment
    envs = gym.vector.SyncVectorEnv([make_env(env_name, 1, run_name="test", mode=mode)])
    env_fn = make_env(
        env_name,
        0,
        run_name="test",
        mode=mode,
        render_mode=None,
    )
    env = env_fn()
    # load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    actors = []
    FN_load_paths = [f"/lichenghao/lzh/CRL/componet/experiments/atari/agents/SpaceInvaders/ModelZoo/SpaceInvaders_{i}_Finetune_42" for i in range(10)]
    for load_path in FN_load_paths:
        actors.append(get_actor(load_path, envs, device, "Finetune"))
    results = []

    for name, param in actors[0].named_parameters():
        print(f"Parameter Name: {name}, Shape: {param.shape}")

    from itertools import product
    for i, j in product(range(len(actors)), repeat=2):
    # for i, j in combinations(range(len(actors)), 2):
        sim = cosine_sim(actors[i], actors[j])
        results.append({
            "M1": i,
            "M2": j,
            "Similarity": sim
        })
    # 转换为 DataFrame 并保存为 CSV
    df = pd.DataFrame(results)
    df.to_csv("FN_space_actor_similarity.csv", index=False)   
      
    # CKA-RL 
    # SpaceInvaders
    # CKA_load_path = "/lichenghao/lzh/CRL/componet/experiments/atari/agents/SpaceInvaders/ModelZoo/SpaceInvaders_9_FuseNet_42"
    # CKA_base_load_path = "/lichenghao/lzh/CRL/componet/experiments/atari/agents/SpaceInvaders/ModelZoo/SpaceInvaders_0_FuseNet_42"
    # Freeway
    CKA_load_path = "/lichenghao/lzh/CRL/componet/experiments/atari/agents/SpaceInvaders/ModelZoo/SpaceInvaders_9_FuseNet_42"
    CKA_base_load_path = "/lichenghao/lzh/CRL/componet/experiments/atari/agents/SpaceInvaders/ModelZoo/SpaceInvaders_0_FuseNet_42"

    actor = get_actor(CKA_load_path, envs, device, "FuseNet")
    base_actor = get_actor(CKA_base_load_path, envs, device, "FuseNet")
    base = get_base(base_actor)
    vectors = get_weight_vectors(actor, base)
    
    # Print the parameters of the actor
    for name, param in actor.named_parameters():
        print(f"Parameter Name: {name}, Shape: {param.shape}")
    

    results = []
    n_weights = len(vectors)
    from tqdm import tqdm
    # for i, j in tqdm(combinations(range(n_weights), 2), total=(n_weights * (n_weights - 1)) // 2):
    for i, j in product(range(len(actors)), repeat=2):
        sim = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
        results.append({
            "M1": i,
            "M2": j,
            "Similarity": sim
        })

    # 保存为 CSV
    df = pd.DataFrame(results)
    csv_path = f"CKA_space_actor_similarity.csv"
    df.to_csv(csv_path, index=False)
 
