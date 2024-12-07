import os
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from .cnn_encoder import CnnEncoder
import sys, os
from .fuse_module import FuseLinear

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.orthogonal_(layer.weight_eps, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    torch.nn.init.constant_(layer.bias_eps, bias_const)
    
    if hasattr(layer, 'weights'):
        torch.nn.init.orthogonal_(layer.weights, std)
    if hasattr(layer, 'biaes'):
        torch.nn.init.constant_(layer.biaes, bias_const)
    return layer

def critic_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CnnFuse1Net(nn.Module):
    def __init__(self, envs, base_dir, prevs_paths=[], map_location=None):
        super().__init__()
        self.hidden_dim = 512
        self.envs = envs
        self.num_weights = -1
        if self.num_weights > 0:
            self.alpha = nn.Parameter(torch.randn(self.num_weights))
        else:
            self.alpha = None

        # if base_dir:
        #     print("Loading actor_base.pt and encoder.pt")
        #     self.network = torch.load(f"{base_dir}/encoder.pt", map_location=map_location)
        # else:
        #     print("Creating new actor_base and encoder")
  
        # weights = [torch.load(f"{path}/actor_weight.pt", map_location=map_location) for path in prevs_paths]
        
        self.actor = nn.Sequential(
            layer_init(FuseLinear(512, 512, num_weights=self.num_weights, alpha=self.alpha)),
            nn.ReLU(),
            layer_init(FuseLinear(512, envs.single_action_space.n, num_weights=self.num_weights, alpha=self.alpha)),
        )
        
        self.critic = critic_init(nn.Linear(512, 1), std=1)
        self.network = CnnEncoder(hidden_dim=512, layer_init=critic_init)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.actor, f"{dirname}/actor.pt")
        torch.save(self.network, f"{dirname}/encoder.pt")
        torch.save(self.critic, f"{dirname}/critic.pt")

    def load(dirname, envs, load_critic=True, reset_actor=False, map_location=None):
        model = CnnFuse1Net(envs)
        return model
