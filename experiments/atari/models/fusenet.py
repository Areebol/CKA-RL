import os
import torch
import numpy as np
import torch.nn as nn
from .cnn_encoder import CnnEncoder
from .fuse_modules import FuseActor, FuseEncoder, Actor
from torch.distributions.categorical import Categorical
from loguru import logger

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    
    if hasattr(layer, 'weights') and layer.weights is not None:
        torch.nn.init.zeros_(layer.weights)
    if hasattr(layer, 'biaes') and layer.biaes is not None:
        torch.nn.init.zeros_(layer.biaes)
    return layer

class FuseNetAgent(nn.Module):
    def __init__(self, envs, base_dir, prevs_paths=[], 
                 fix_alpha: bool = False,
                 alpha_factor: float = 1/100,
                 delta_theta_mode: str = "T",
                 fuse_encoder: bool = False,
                 fuse_actor: bool = True,
                 reset_actor: bool = False,
                 use_alpha_scale: bool = True,
                 map_location=None):
        super().__init__()
        self.delta_theta_mode = delta_theta_mode
        self.fuse_encoder = fuse_encoder
        self.fuse_actor = fuse_actor
        self.hidden_dim = 512
        self.envs = envs
        self.i = 0
        assert(fuse_encoder or fuse_actor)
        self.num_weights = len(prevs_paths)
        latest_dir = prevs_paths[-1] if self.num_weights > 0 else None
        
        # Alpha Setting
        if self.num_weights > 0:
            if fix_alpha: # Alpha is untrainable
                self.alpha = nn.Parameter(torch.zeros(self.num_weights), requires_grad=False)
                logger.info("Fix alpha to all 0")
            else: # Alpha is trainable
                self.alpha = nn.Parameter(torch.ones(self.num_weights) * alpha_factor, requires_grad=True)
                self.alpha_scale = nn.Parameter(torch.ones(1), requires_grad=True)
                logger.info("Train alpha")
            if not use_alpha_scale or fix_alpha:
                self.alpha_scale = nn.Parameter(torch.ones(1), requires_grad=False)
            logger.info(f"Alpha's shape: {self.alpha.shape}, Alpha: {self.alpha.data}, Alpha scale: {self.alpha_scale.data}")
        else:
            self.alpha = None
            self.alpha_scale = None
        
        # Actor 's fuse or not 
        if self.fuse_actor:
            logger.debug("FuseNet fuse actor")
            self.actor = FuseActor( hidden_dim=512,
                                    layer_init=layer_init,
                                    n_actions=envs.single_action_space.n,
                                    alpha=self.alpha,
                                    alpha_scale=self.alpha_scale,
                                    num_weights=self.num_weights )
            self.actor.set_base_and_vectors(base_dir, prevs_paths)
        else:
            if latest_dir is not None and reset_actor is False:
                logger.info(f"Loading actor from {latest_dir}")
                self.actor = torch.load(f"{latest_dir}/actor.pt", map_location=map_location)
            else:
                logger.info("Train actor from scratch")
                self.actor = Actor( hidden_dim=512,
                                    n_actions=envs.single_action_space.n,
                                    layer_init=layer_init)
        
        # Encoder's fuse or not
        if self.fuse_encoder:
            logger.debug("FuseNet fuse encoder")
            self.network = FuseEncoder(hidden_dim=512, 
                                       layer_init=layer_init, 
                                       alpha=self.alpha,
                                       alpha_scale=self.alpha_scale,
                                       num_weights=self.num_weights )
            self.network.set_base_and_vectors(base_dir, prevs_paths)
        else:
            if latest_dir is not None:
                logger.info(f"Loading encoder from {latest_dir}")
                self.network = torch.load(f"{latest_dir}/encoder.pt", map_location=map_location)
            else:
                logger.info("Train encoder from scratch")
                self.network = CnnEncoder(hidden_dim=512, layer_init=layer_init)

        # Critic 
        self.critic = layer_init(nn.Linear(512, 1), std=1)

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
        # for actor, merge `theta + alpha * tau` to `theta` if delta_theta_mode  == 'TAT'
        if self.delta_theta_mode == "TAT":
            if self.fuse_actor:
                logger.info(f"save actor weight as theta + alpha * tau")
                self.merge_actor_weight()
            if self.fuse_encoder:
                logger.info(f"save encoder weight as theta + alpha * tau")
                self.merge_encoder_weight()
        else:
            logger.info("save weight as theta")
        torch.save(self.actor, f"{dirname}/actor.pt")
        torch.save(self.network, f"{dirname}/encoder.pt")
        torch.save(self.critic, f"{dirname}/critic.pt")

    def load(dirname, envs, load_critic=True, reset_actor=False, map_location=None):
        model = FuseNetAgent(envs)
        model.network = torch.load(f"{dirname}/encoder.pt", map_location=map_location)
        if load_critic:
            model.critic = torch.load(f"{dirname}/critic.pt", map_location=map_location)
        if not reset_actor:
            model.actor = torch.load(f"{dirname}/actor.pt", map_location=map_location)            
        return model

    def merge_actor_weight(self):
        if self.alpha is None:
            logger.warning("Not alpha exist in FuseNetAgent, not merge")
            return 
        
        logger.info("merge actor's weight")
        self.actor.merge_weight()

    def merge_encoder_weight(self):
        if self.alpha is None or self.fuse_encoder is False:
            logger.warning("Not alpha exist in FuseNetAgent or Not Fuse Encoder, not merge")
            return 
        
        logger.info("merge encoder's weight")
        self.network.merge_weight()
