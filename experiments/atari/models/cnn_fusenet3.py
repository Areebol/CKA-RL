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
    torch.nn.init.constant_(layer.bias, bias_const)
    
    if hasattr(layer, 'weight_eps'):
        torch.nn.init.orthogonal_(layer.weight_eps, std)
    if hasattr(layer, 'bias_eps'):
        torch.nn.init.constant_(layer.bias_eps, bias_const)
    if hasattr(layer, 'weights'):
        torch.nn.init.orthogonal_(layer.weights, std)
    if hasattr(layer, 'biaes'):
        torch.nn.init.constant_(layer.biaes, bias_const)
    return layer

def load_actor_base_and_vectors(base_dir, prevs_paths):
    base = []
    vectors = []
    num_weights = -1
    if base_dir:
        print("Loading base from", f"{base_dir}/actor.pt")
        base_state_dict = torch.load(f"{base_dir}/actor.pt").state_dict()
        num_weights += 1
        for i in [0,2]:
            base.append({"weight":base_state_dict[f"{i}.weight_eps"],"bias":base_state_dict[f"{i}.bias_eps"]})
    if len(prevs_paths):
        for i in [0,2]:
            vector_weight = []
            vector_bias = []
            for p in prevs_paths:
                print("Loading vector from", f"{p}/actor.pt")
                vector_state_dict = torch.load(f"{p}/actor.pt").state_dict()
                vector_weight.append(vector_state_dict[f"{i}.weight_eps"])
                vector_bias.append(vector_state_dict[f"{i}.bias_eps"])
            vectors.append({"weight":torch.stack(vector_weight),
                            "bias":torch.stack(vector_bias)})
    num_weights += vectors[0]["weight"].shape[0] if vectors else 0
    if base == []:
        base = [None,None]
    if vectors == []:
        vectors = [None,None]
    return base, vectors, num_weights

class CnnFuse3Net(nn.Module):
    def __init__(self, envs, base_dir, prevs_paths=[], map_location=None):
        super().__init__()
        self.hidden_dim = 512
        self.envs = envs
        self.i = 0
        base, vectors, self.num_weights = load_actor_base_and_vectors(base_dir, prevs_paths)
        
        if self.num_weights > 0:
            self.alpha = nn.Parameter(torch.randn(self.num_weights), requires_grad=True)
            print("Alpha's shape:", self.alpha.shape)
            self.alpha = nn.Parameter(torch.randn(self.num_weights), requires_grad=True)
            print("Alpha's shape:", self.alpha.shape)
        else:
            self.alpha = None
        
        self.actor = nn.Sequential(
            layer_init(FuseLinear(512, 512, num_weights=self.num_weights, alpha=self.alpha)),
            nn.ReLU(),
            layer_init(FuseLinear(512, envs.single_action_space.n, num_weights=self.num_weights, alpha=self.alpha)),
        )
        
        self.actor[0].set_base_and_vectors(base[0], vectors[0])
        self.actor[2].set_base_and_vectors(base[1], vectors[1])
        
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.network = CnnEncoder(hidden_dim=512, layer_init=layer_init)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None,log_writter=None, global_step=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
            
        # Log alpha
        if log_writter is not None and global_step is not None and self.alpha is not None:
            normalized_alpha = torch.softmax(self.alpha, dim=0)
            for i, alpha_i in enumerate(normalized_alpha):
                log_writter.add_scalar(
                    f"alpha/{i}", alpha_i.item(), global_step
                )
                
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.actor, f"{dirname}/actor.pt")
        torch.save(self.network, f"{dirname}/encoder.pt")
        torch.save(self.critic, f"{dirname}/critic.pt")

    def load(dirname, envs, load_critic=True, reset_actor=False, map_location=None):
        model = CnnFuse3Net(envs)
        model.network = torch.load(f"{dirname}/encoder.pt", map_location=map_location)
        if load_critic:
            model.critic = torch.load(f"{dirname}/critic.pt", map_location=map_location)
        if not reset_actor:
            model.actor = torch.load(f"{dirname}/actor.pt", map_location=map_location)            
        return model
