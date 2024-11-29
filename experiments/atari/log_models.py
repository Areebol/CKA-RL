# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from typing import Literal, Tuple, Optional
import pathlib

from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from models import (
    CnnSimpleAgent,
    DinoSimpleAgent,
    CnnCompoNetAgent,
    ProgressiveNetAgent,
    PackNetAgent,
    CnnFairAgent,
)

def make_env(env_id, mode=None):
    def thunk():
        if mode is None:
            env = gym.make(env_id)
        else:
            env = gym.make(env_id, mode=mode)
       
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)

        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk


if __name__ == "__main__":
    envs = gym.vector.SyncVectorEnv([make_env("ALE/SpaceInvaders-v5", 0)],)
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = CnnSimpleAgent(envs)
    print("cnn-simple",agent)
   
    agent = CnnCompoNetAgent(
        envs,
        prevs_paths=[],
        finetune_encoder=False,
    )
    print("com-net",agent)
   
    agent = CnnFairAgent(envs)
    print("cnn-fair",agent)