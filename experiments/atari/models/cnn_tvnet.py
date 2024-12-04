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
        print("==> No vectors to merge")
        print("==> Base model return")
        return base.copy()
    coef = 1.0 / len(vectors)
    merge_model = base.copy()
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
    print(f"====> Computing {pretrained_state_dict.keys()} in task vector")
    for key in pretrained_state_dict:
        if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
            print(f"====> {key} is int type, skip")
            continue
        vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    print(f"====> {pretrained_state_dict.keys()} Compued")
    print("==> Task vector computed") 
    return vector

class CnnTvNetAgent(nn.Module):
    def __init__(self, envs, prevs_paths=[], map_location=None):
        """
        if self.is_first:
            init_base_model()
        else:
            load_base_model() + load_task_vector()
            model = base_model + weighted_task_vector()
        """
        super().__init__()
        self.hidden_dim = 512
        self.is_first = len(prevs_paths) == 0
        self.envs = envs
        # 1. load encoder
        if not self.is_first:
            self.encoder = CnnEncoder(hidden_dim=self.hidden_dim, layer_init=layer_init)
            
            previous_encoders_state_dict = [
                torch.load(f"{p}/encoder.pt", map_location=map_location).state_dict()
                for p in prevs_paths
            ]
            self.encoder_base_state_dict = previous_encoders_state_dict[0] # base model
            encoder_vectors_state_dict = previous_encoders_state_dict[1:] # vectors
            self.encoder.load_state_dict(merge_vectors(self.encoder_base_state_dict, encoder_vectors_state_dict))
            print("==> Encoder initialized from base model and vectors")
        else:
            self.encoder = CnnEncoder(hidden_dim=self.hidden_dim, layer_init=layer_init)
            print("==> Encoder linitialized from sketch")

        # 2. reset critic
        self.critic = layer_init(nn.Linear(self.hidden_dim, 1), std=1)

        # 3. init actor
        if not self.is_first:
            self.actor = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, envs.single_action_space.n),
            )
            
            previous_actors_state_dict = [
                torch.load(f"{p}/actor.pt", map_location=map_location).state_dict()
                for p in prevs_paths
            ]
            self.actor_base_state_dict = previous_actors_state_dict[0] # base model
            vectors_state_dict = previous_actors_state_dict[1:] # vectors
            self.actor.load_state_dict(merge_vectors(self.actor_base_state_dict, vectors_state_dict))
            print("==> Actor initialized from base model and vectors")
        else:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self.hidden_dim, envs.single_action_space.n), std=0.01),
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
        torch.save(self.critic, f"{dirname}/crititc.pt")
        if self.is_first:
            torch.save(self.actor, f"{dirname}/actor.pt")
            print("==> Save actor as base model")
            torch.save(self.encoder, f"{dirname}/encoder.pt")
            print("==> Save encoder as base model")
        else:
            actor_task_vector = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.envs.single_action_space.n),
            )
            actor_task_vector.load_state_dict(get_task_vector(self.actor_base_state_dict, self.actor.state_dict()))
            torch.save(actor_task_vector, f"{dirname}/actor.pt")
            print("==> Save task vector = actor - base")
            encoder_task_vector = CnnEncoder(hidden_dim=self.hidden_dim, layer_init=layer_init)
            encoder_task_vector.load_state_dict(get_task_vector(self.encoder_base_state_dict, self.encoder.state_dict()))
            torch.save(encoder_task_vector, f"{dirname}/encoder.pt")
            print("==> Save task vector = encoder - base")

    def load(dirname, envs, base_dir, map_location=None):
        """
        load base_model and task_vector
        """
        print("Loading base:", base_dir)

        model = CnnTvNetAgent(
            envs=envs, prevs_paths=[], map_location=map_location
        )
        model.critic = torch.load(f"{dirname}/crititc.pt", map_location=map_location)

        # load the state dict of the vector
        actor_vector = torch.load(f"{dirname}/actor.pt", map_location=map_location)
        encoder_vector = torch.load(f"{dirname}/encoder.pt", map_location=map_location)

        # load the state dict of the base
        actor_base = torch.load(f"{base_dir}/actor.pt", map_location=map_location)
        encoder_base = torch.load(f"{base_dir}/encoder.pt", map_location=map_location)
        
        model.actor.load_state_dict(merge_vectors(actor_base.state_dict(), [actor_vector.state_dict()]))
        model.encoder.load_state_dict(merge_vectors(encoder_base.state_dict(), [encoder_vector.state_dict()]))

        return model
    
    
class CnnTv2NetAgent(nn.Module):
    def __init__(self, envs, prevs_paths=[], map_location=None):
        super().__init__()
        self.hidden_dim = 512
        self.is_first = len(prevs_paths) == 0
        self.envs = envs
        # 1. load encoder
        if not self.is_first:
            self.encoder = CnnEncoder(hidden_dim=self.hidden_dim, layer_init=layer_init)
            
            previous_encoders_state_dict = [
                torch.load(f"{p}/encoder.pt", map_location=map_location).state_dict()
                for p in prevs_paths
            ]
            self.encoder_base_state_dict = previous_encoders_state_dict[0] # base model
            encoder_vectors_state_dict = previous_encoders_state_dict[1:] # vectors
            self.encoder.load_state_dict(merge_vectors(self.encoder_base_state_dict, encoder_vectors_state_dict))
            print("==> Encoder initialized from base model and vectors")
        else:
            self.encoder = CnnEncoder(hidden_dim=self.hidden_dim, layer_init=layer_init)
            print("==> Encoder linitialized from sketch")

        # 2. reset critic, actor
        self.critic = layer_init(nn.Linear(self.hidden_dim, 1), std=1)
        self.actor = nn.Sequential(
                layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self.hidden_dim, envs.single_action_space.n), std=0.01),
            )
        
    def get_value(self, x):
        return self.critic(self.encoder(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.encoder(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.critic, f"{dirname}/crititc.pt")
        torch.save(self.actor, f"{dirname}/actor.pt")
        if self.is_first:
            torch.save(self.encoder, f"{dirname}/encoder.pt")
            print("==> Save encoder as base model")
        else:
            encoder_task_vector = CnnEncoder(hidden_dim=self.hidden_dim, layer_init=layer_init)
            encoder_task_vector.load_state_dict(get_task_vector(self.encoder_base_state_dict, self.encoder.state_dict()))
            torch.save(encoder_task_vector, f"{dirname}/encoder.pt")
            print("==> Save task vector = encoder - base")

    def load(dirname, envs, base_dir, map_location=None):
        """
        load base_model and task_vector
        """
        print("Loading base:", base_dir)

        model = CnnTv2NetAgent(
            envs=envs, prevs_paths=[], map_location=map_location
        )
        model.critic = torch.load(f"{dirname}/crititc.pt", map_location=map_location)
        model.actor = torch.load(f"{dirname}/actor.pt", map_location=map_location)

        # load the state dict of the vector
        encoder_vector = torch.load(f"{dirname}/encoder.pt", map_location=map_location)

        # load the state dict of the base
        encoder_base = torch.load(f"{base_dir}/encoder.pt", map_location=map_location)
        
        model.encoder.load_state_dict(merge_vectors(encoder_base.state_dict(), [encoder_vector.state_dict()]))

        return model


