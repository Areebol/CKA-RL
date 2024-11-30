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

@torch.no_grad()
def merge_vectors(base, vectors):
    """
    Merge the base model and the task vectors
    base: state_dict
    vectors: list of state_dict
    """
    if len(vectors) == 0:
        return base.copy()
    coef = 1.0 / len(vectors)
    print("==> No vectors to merge")
    merge_model = base.copy()
    print("==> Base model return")
    print("==> Merging vectors")
    print("==> Merge coef: ", coef)
    for i, vector in enumerate(vectors):
        print(f"====> Merging vector-{i}")
        print(f"====> Vector-{i}'s keys: ", vector.keys())
        for key in base:
            if key not in vector:
                print(f'Warning, key {key} is not present in both task vectors.')
                continue
            merge_model[key] = merge_model[key] + coef * vector[key]
        print(f"====> Vector-{i} merged")
    print("==> Vectors merged")
    return merge_model

@torch.no_grad()
def get_task_vector(base, actor):
    """
    base: state_dict
    actor: state_dict
    """
    pretrained_state_dict = base
    finetuned_state_dict = actor
    vector = {}
    print("==> Computing task vector")
    for key in pretrained_state_dict:
        print(f"====> Computing {key} in task vector")
        if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
            print(f"====> {key} is int type, skip")
            continue
        vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
        print(f"====> {key} Compued")
    print("==> Task vector computed") 
    return vector

class CnnTvNetAgent(nn.Module):
    def __init__(self, envs, prevs_paths=[], finetune_encoder=False, map_location=None):
        """
        if self.is_first:
            init_base_model()
        else:
            load_base_model() + load_task_vector()
            model = base_model + weighted_task_vector()
        """
        super().__init__()
        hidden_dim = 512
        self.is_first = len(prevs_paths) == 0

        # 1. load encoder
        if not finetune_encoder or len(prevs_paths) == 0:
            self.encoder = CnnEncoder(hidden_dim=hidden_dim, layer_init=layer_init)
        else:
            self.encoder = torch.load(
                f"{prevs_paths[-1]}/encoder.pt", map_location=map_location
            )
            print("==> Encoder loaded from last TvNet module")

        # 2. reset critic
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1)

        # 3. init actor
        if not self.is_first:
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, envs.single_action_space.n),
            )
            self.task_vector = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, envs.single_action_space.n),
            )
            
            previous_actors_state_dict = [
                torch.load(f"{p}/actor.pt", map_location=map_location).state_dict()
                for p in prevs_paths
            ]
            self.base_state_dict = previous_actors_state_dict[0] # base model
            self.vectors_state_dict = previous_actors_state_dict[1:] # vectors
            self.actor.load_state_dict(merge_vectors(self.base_state_dict, self.vectors_state_dict))
            print("==> Actor initialized from base model and vectors")
        else:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, envs.single_action_space.n), std=0.01),
            )
            print("==> Actor initialized from sketch")

    def get_value(self, x):
        """
        Not need change
        """
        return self.critic(self.encoder(x))

    def get_action_and_value(
        self, x, action=None,
    ):
        """
        Not need change
        """
        hidden = self.encoder(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def save(self, dirname):
        """
        if self.is_first: # save as the base model
            save_base_model
        else:
            save_task_vectors
        """
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.encoder, f"{dirname}/encoder.pt")
        torch.save(self.critic, f"{dirname}/crititc.pt")
        if self.is_first:
            torch.save(self.actor, f"{dirname}/actor.pt")
            print("==> Save actor as base model")
        else:
            self.task_vector.load_state_dict(get_task_vector(self.base_state_dict, self.actor.state_dict()))
            torch.save(self.task_vector, f"{dirname}/actor.pt")
            print("==> Save task vector = actor - base")

    def load(dirname, envs, base_dir, map_location=None):
        """
        load base_model and task_vector
        """
        print("Loading base:", base_dir)

        model = CnnTvNetAgent(
            envs=envs, prevs_paths=[], map_location=map_location
        )
        model.encoder = torch.load(f"{dirname}/encoder.pt", map_location=map_location)
        model.critic = torch.load(f"{dirname}/crititc.pt", map_location=map_location)

        # load the state dict of the vector
        vector = torch.load(f"{dirname}/actor.pt", map_location=map_location)
        # load the state dict of the base
        base = torch.load(f"{base_dir}/actor.pt", map_location=map_location)
        
        curr = model.actor.state_dict()
        for k in base.state_dict():
            curr[k] = base[k] + vector[k]
        model.actor.load_state_dict(curr)

        return model
