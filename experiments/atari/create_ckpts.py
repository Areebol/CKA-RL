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
import sys
from typing import Literal, Tuple, Optional
import pathlib
from loguru import logger
from tqdm import tqdm
from models.cbp_modules import GnT
from model_utils.AdamGnT import AdamGnT
# from task_utils import get_method_type

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
    CnnCompoNetAgent,
    ProgressiveNetAgent,
    PackNetAgent,
    CnnTvNetAgent, 
    FuseNetAgent,
    FuseNetwMergeAgent,
    CnnMaskAgent,
    CnnCbpAgent,
    # CSPAgent,
    RewireAgent,
    CReLUsAgent,
)


@dataclass
class Args:
    # Model type
    method_type: str = "Baseline"
    """The name of the model to use as agent."""
    dino_size: Literal["s", "b", "l", "g"] = "s"
    """Size of the dino model (only needed when using dino)"""
    prev_units: Tuple[pathlib.Path, ...] = ()
    """Paths to the previous models. Only used when employing a CompoNet or cnn-simple-ft (finetune) agent"""
    mode: int = None
    """Playing mode for the Atari game. The default mode is used if not provided"""
    componet_finetune_encoder: bool = False
    """Whether to train the CompoNet's encoder from scratch of finetune it from the encoder of the previous task"""
    total_task_num: Optional[int] = None
    """Total number of tasks, required when using PackNet"""
    prevs_to_noise: Optional[int] = 0
    """Number of previous policies to set to randomly selected distributions, only valid when method_type is `CompoNet`"""

    # Experiment arguments
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Atari-PPO"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "ALE/Freeway-v5"
    """the id of the environment"""
    total_timesteps: int = int(1e6)
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    
    fuse_lr_scale: float = 100.0
    """scale the learning rate of the alpha parameter"""
    
    debug: bool = False
    """log level"""
    tag: str = "Debug"
    """experiment tag"""
    
    alpha_factor: float = 1e-2
    """fuse net's alpha initialization factor 1 * alpha_factor"""
    fix_alpha: bool = False
    """fuse net's alpha would be fix to constant"""
    alpha_learning_rate: float = 2.5e-4
    """the learning rate of alpha optimizer"""
    delta_theta_mode: str = "T" # T or TAT
    """the mode to cacluate delta theta"""
    fuse_encoder: bool = False # True or False
    """whether to fuse encoder"""
    fuse_actor: bool = True # True or False
    """whether to fuse actor"""
    reset_actor: bool = True # True or False
    """whether to reset actor"""
    global_alpha: bool = True # True or False
    """whether to use global alpha for whole agent"""
    alpha_init: str = "Randn" # "Randn" "Major" "Uniform"
    """how to init alpha in FuseNet"""
    alpha_major: float = 0.6 
    """init major""" # Major alpha init, theta_{i-1} will be init to log(major) + C, others will be uniform
    pool_size: int = 3
    """pool size for FuseNetwMerge"""

    task_id: int = 0
    """task id for the current task"""

def make_env(env_id, idx, capture_video, run_name, mode=None):
    def thunk():
        if mode is None:
            env = gym.make(env_id)
        else:
            env = gym.make(env_id, mode=mode)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
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
    args = tyro.cli(Args)
    if args.debug is False:
        logger.remove() 
        handler_id = logger.add(sys.stderr, level="INFO") 
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # m = f"{args.mode}" if args.mode is not None else ""
    m = f"{args.task_id}" if args.mode is not None else ""
    env_name = args.env_id.split("/")[1].split("-")[0] # e.g. ALE/Freeway-v5 -> Freeway
    run_name = f"{env_name}_{m}_{args.method_type}_{args.seed}"
    ao_exist = False # has alpha_optimizer if True
    
    logger.info(f"Run's name: {run_name}")

    logs = {"global_step": [0], "episodic_return": [0]}
    # logger.info(f"Tensorboard writing to runs/{args.tag}/{run_name}")
    writer = SummaryWriter(f"runs/{args.tag}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id, i, args.capture_video, run_name, mode=args.mode)
            for i in range(args.num_envs)
        ],
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    logger.info(f"Method: {args.method_type}")
    if args.method_type == "Baseline":
        agent = CnnSimpleAgent(envs).to(device)
    elif args.method_type == "Finetune":
        if len(args.prev_units) > 0:
            agent = CnnSimpleAgent.load(
                args.prev_units[0], envs, load_critic=False, reset_actor=False
            ).to(device)
        else:
            agent = CnnSimpleAgent(envs).to(device)
    elif args.method_type == "CompoNet":
        agent = CnnCompoNetAgent(
            envs,
            prevs_paths=args.prev_units,
            finetune_encoder=args.componet_finetune_encoder,
            map_location=device,
        ).to(device)
    elif args.method_type == "ProgNet":
        agent = ProgressiveNetAgent(
            envs, prevs_paths=args.prev_units, map_location=device
        ).to(device)
    elif args.method_type == "PackNet":
        # retraining in 20% of the total timesteps
        packnet_retrain_start = args.total_timesteps - int(args.total_timesteps * 0.2)

        if args.total_task_num is None:
            print("CLI argument `total_task_num` is required when using PackNet.")
            quit(1)

        if len(args.prev_units) == 0:
            agent = PackNetAgent(
                envs,
                task_id=(args.mode + 1),
                is_first_task=True,
                total_task_num=args.total_task_num,
            ).to(device)
        else:
            agent = PackNetAgent.load(
                args.prev_units[0],
                task_id=args.mode + 1,
                restart_actor_critic=True,
                freeze_bias=True,
            ).to(device)
    elif args.method_type == "TvNet":
        agent = CnnTvNetAgent(envs, prevs_paths=args.prev_units, 
                              map_location=device).to(device)
    elif args.method_type == "FuseNet":
        base_dir = args.prev_units[0] if len(args.prev_units) > 0 else None
        encoder_dir = args.prev_units[-1] if len(args.prev_units) > 0 else None
        agent = FuseNetAgent(envs, 
                             base_dir=base_dir, 
                             prevs_paths=args.prev_units,
                             alpha_factor=args.alpha_factor,
                             fix_alpha=args.fix_alpha,
                             delta_theta_mode=args.delta_theta_mode,
                             fuse_encoder=args.fuse_encoder,
                             fuse_actor=args.fuse_actor,
                             reset_actor=args.reset_actor,
                             global_alpha=args.global_alpha,
                             alpha_init=args.alpha_init,
                             alpha_major=args.alpha_major,
                             map_location=device).to(device)
        agent.log_alphas()
    elif args.method_type == "FuseNetwMerge":
        base_dir = args.prev_units[0] if len(args.prev_units) > 0 else None
        latest_dir = args.prev_units[-1] if len(args.prev_units) > 0 else None
        agent = FuseNetwMergeAgent(envs, 
                             base_dir=base_dir, 
                             latest_dir=latest_dir,
                             alpha_factor=args.alpha_factor,
                             fix_alpha=args.fix_alpha,
                             delta_theta_mode=args.delta_theta_mode,
                             fuse_encoder=args.fuse_encoder,
                             fuse_actor=args.fuse_actor,
                             reset_actor=args.reset_actor,
                             global_alpha=args.global_alpha,
                             alpha_init=args.alpha_init,
                             alpha_major=args.alpha_major,
                             pool_size=args.pool_size,
                             map_location=device).to(device)
        agent.log_alphas()
    elif args.method_type == "MaskNet":
        logger.info(f"num_task: {args.total_task_num}")
        logger.info(f"task: {args.mode}")
        if len(args.prev_units) > 0:
            logger.info(f"loading from {args.prev_units[0]}")
            agent = CnnMaskAgent.load(args.prev_units[0], envs, num_tasks=args.total_task_num, load_critic=False, reset_actor=False).to(device)
            agent.set_task(args.mode, new_task=True)
        else:
            agent = CnnMaskAgent(envs, num_tasks=args.total_task_num).to(device)
            agent.set_task(args.mode, new_task=False)
    elif args.method_type == "CbpNet":
        if len(args.prev_units) > 0:
            agent = CnnCbpAgent.load(
                args.prev_units[0], envs, load_critic=False, reset_actor=False
            ).to(device)
        else:
            agent = CnnCbpAgent(envs).to(device)
    elif args.method_type == "CSP":
        raise NotImplementedError
    elif args.method_type == "Rewire":
        if len(args.prev_units) > 0:
            agent = RewireAgent.load(
                args.prev_units[0], envs, load_critic=False, reset_actor=False
            ).to(device)
        else:
            agent = RewireAgent(envs).to(device)
        agent.set_task()
    elif args.method_type == "CReLUs":
        if len(args.prev_units) > 0:
            agent = CReLUsAgent.load(
                args.prev_units[0], envs, load_critic=False, reset_actor=False
            ).to(device)
        else:
            agent = CReLUsAgent(envs).to(device)
    else:
        logger.error(f"Method type {args.method_type} is not valid.")
        quit(1)
        
    if args.tag is not None:
        log_dir = f"./data/{env_name}/{args.tag}/{args.method_type}/{args.task_id}"  
        # logger.info(f"saved log return to {log_dir}/returns.csv") # ./data/Freeway/tag/FuseNet/mode/returns.csv
        os.makedirs(log_dir, exist_ok=True)
    
        # logger.info(f"Saving trained agent to `./agents/{env_name}/{args.tag}/{run_name}`")
        agent.save(dirname=f"./agents/{env_name}/{args.tag}/{run_name}") # ./agents/Freeway/tag/run_name
