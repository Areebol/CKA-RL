import os
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from .cnn_encoder import CnnEncoder
import sys, os

sys.path.append(os.path.dirname(__file__) + "/../../../")
from componet import CompoNet, FirstModuleWrapper


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CnnTVNetAgent(nn.Module):
    def __init__(self, envs, prevs_paths=[], finetune_encoder=False, map_location=None):
        super().__init__()
        hidden_dim = 512

        # 1. 保持finetune_encoder，不需要额外训练cnn network
        if not finetune_encoder or len(prevs_paths) == 0:
            self.encoder = CnnEncoder(hidden_dim=hidden_dim, layer_init=layer_init)
        else:
            self.encoder = torch.load(
                f"{prevs_paths[-1]}/encoder.pt", map_location=map_location
            )
            print("==> Encoder loaded from last TVNet module")

        # 2. Critic重新初始化
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1)

        # 3. Actor初始化
        if len(prevs_paths) > 0:
            self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
            self.buffers = ...
        else:
            self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)

    def get_value(self, x):
        return self.critic(self.encoder(x))

    def get_action_and_value(
        self, x, action=None,
    ):
        hidden = self.encoder(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        if hasattr(self.actor, "previous_units"):
            del self.actor.previous_units
        torch.save(self.actor, f"{dirname}/actor.pt")
        torch.save(self.critic, f"{dirname}/crititc.pt")
        torch.save(self.encoder, f"{dirname}/encoder.pt")

    def load(dirname, envs, prevs_paths=[], map_location=None):
        print("Loading previous:", prevs_paths)

        model = CnnTVNetAgent(
            envs=envs, prevs_paths=prevs_paths, map_location=map_location
        )
        model.encoder = torch.load(f"{dirname}/encoder.pt", map_location=map_location)

        # load the state dict of the actor
        actor = torch.load(f"{dirname}/actor.pt", map_location=map_location)
        curr = model.actor.state_dict()
        other = actor.state_dict()
        for k in other:
            curr[k] = other[k]
        model.actor.load_state_dict(curr)

        model.critic = torch.load(f"{dirname}/crititc.pt", map_location=map_location)
        return model
