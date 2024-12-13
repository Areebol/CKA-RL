import os
import math
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from .cnn_encoder import CnnEncoder
from torch.nn import Parameter, ParameterList, init
from torch.distributions.categorical import Categorical
from loguru import logger

class FuseLinear(nn.Module):
    r"""Applies a linear transformation to the 
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, 
                 out_features: int, 
                 bias: bool = True, 
                 num_weights: int = 0, # 0 = train base weight， n = train base weight + alpha * tau
                 alpha: nn.Parameter = None,
                 alpha_scale: nn.Parameter = None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha # given by agent
        self.alpha_scale = alpha_scale # given by agnet
        self._bias = bias
        self.num_weights = num_weights # size of tau
        
        if self.num_weights > 0:
            # alpha need to match num_weights
            assert(self.alpha.shape[0] == self.num_weights)
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=True)
        assert self.num_weights >= 0, "num_weights must be non-negative"
        
        # tau = {theta_0,theta_1,...theta_n}
        if self.num_weights > 0:
            self.weights = Parameter(torch.stack([torch.empty((out_features, in_features)) for _ in range(num_weights)], dim=0), requires_grad=False)
        else:
            self.weights = None
        
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=True)

            # tau = {theta_0,theta_1,...theta_n}
            if self.num_weights > 0:
                self.biaes = Parameter(torch.stack([torch.empty(out_features) for _ in range(num_weights)], dim=0), requires_grad=False)
            else:
                self.biaes = None
        else:
            self.register_parameter('bias', None)
            if self.num_weights > 0:
                self.register_parameter('biases', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # alpha * tau {theta_0,theta_1,...theta_n} + base
        if self.alpha is not None:
            # logger.debug(f"Alpha is {self.alpha.data}, forward with alpha * tau")
            alphas_normalized = F.softmax(self.alpha * self.alpha_scale, dim=0)
            weight = self.weight + (alphas_normalized.view(-1, 1, 1) * self.weights).sum(dim = 0)
            if self._bias:
                bias = self.bias + (alphas_normalized.view(-1,1) * self.biaes).sum(dim=0)
            else:
                bias = None
        else:
            # logger.debug("Alpha is None, forward with base weight only")
            weight = self.weight
            if self._bias:
                bias = self.bias
                
        return F.linear(input, weight, bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, num_weights={self.num_weights}'   
        
    def set_base_and_vectors(self, base, vectors):
        # Set base weight
        if base is not None:
            logger.info(f"Seting base with tensor's shape = {base['weight'].shape}")
            assert('weight' in base and 'bias' in base)
            self.weight.data.copy_(base['weight'])
            self.bias.data.copy_(base['bias'])
        else:
            logger.info(f"Base is None, train base weight from scratch")
            
        # Set vectors weight
        if vectors is not None: 
            logger.info(f"Seting vectors with tensor's shape = {vectors['weight'].shape}")
            assert('weight' in vectors and 'bias' in vectors)
            assert base['weight'].shape == vectors['weight'].shape[1:], f"Shape of base {base['weight'].shape} weight and vectors weight {vectors['weight'].shape[1:]} must match"
            assert base['bias'].shape == vectors['bias'].shape[1:], f"Shape of base {base['bias'].shape} bias and vectors bias {vectors['bias'].shape[1:]} must match"
            
            self.weights.data.copy_(vectors['weight'])
            self.biaes.data.copy_(vectors['bias'])


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
                 fix_alpha: bool = False, encoder_dir=None,
                 alpha_factor: float = 1/100,
                 map_location=None):
        super().__init__()
        self.hidden_dim = 512
        self.envs = envs
        self.i = 0
        actor_base, actor_vectors, self.num_weights = self.load_actor_base_and_vectors(base_dir, prevs_paths)
        
        if self.num_weights > 0:
            if fix_alpha:
                self.alpha = nn.Parameter(torch.zeros(self.num_weights), requires_grad=False)
                self.alpha_scale = nn.Parameter(torch.ones(1), requires_grad=False)
                logger.info("Fix alpha to all 0")
            else:
                self.alpha = nn.Parameter(torch.ones(self.num_weights) * alpha_factor, requires_grad=True)
                self.alpha_scale = nn.Parameter(torch.ones(1), requires_grad=True)
                logger.info("Train alpha")
            logger.info(f"Alpha's shape: {self.alpha.shape}, Alpha: {self.alpha.data}, Alpha scale: {self.alpha_scale.data}")
        else:
            self.alpha = None
            self.alpha_scale = None
        
        self.actor = nn.Sequential(
            layer_init(FuseLinear(512, 512, num_weights=self.num_weights, 
                                  alpha=self.alpha, alpha_scale=self.alpha_scale)),
            nn.ReLU(),
            layer_init(FuseLinear(512, envs.single_action_space.n, num_weights=self.num_weights, 
                                  alpha=self.alpha, alpha_scale=self.alpha_scale),std=0.01),
        )
        self.actor[0].set_base_and_vectors(actor_base[0], actor_vectors[0])
        self.actor[2].set_base_and_vectors(actor_base[1], actor_vectors[1])
        
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        if encoder_dir is not None:
            logger.info(f"Loading encoder from {encoder_dir}")
            self.network = torch.load(f"{encoder_dir}/encoder.pt", map_location=map_location)
        else:
            logger.info("No Encoder exists, Train encoder from scratch")
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
        model = FuseNetAgent(envs)
        model.network = torch.load(f"{dirname}/encoder.pt", map_location=map_location)
        if load_critic:
            model.critic = torch.load(f"{dirname}/critic.pt", map_location=map_location)
        if not reset_actor:
            model.actor = torch.load(f"{dirname}/actor.pt", map_location=map_location)            
        return model

    def load_actor_base_and_vectors(self, base_dir, vector_dirs):
        """return theta_0, prevs_theta when base_dir exists
        base_dir : base weight's dir
        prevs_paths : prevs weight's dir (including base weight)
        """
        base = []
        vectors = []
        num_weights = 0
        if base_dir:
            # load base weight
            logger.info(f"Loading base from {base_dir}/actor.pt")
            base_state_dict = torch.load(f"{base_dir}/actor.pt").state_dict()
            for i in [0,2]:
                base.append({"weight":base_state_dict[f"{i}.weight"],"bias":base_state_dict[f"{i}.bias"]})
        else:
            return [None,None],[None,None], 0

        for i,j in zip([0,2],[0,1]):
            vector_weight = []
            vector_bias = []
            for p in vector_dirs:
                logger.info(f"Loading vector from {p}/actor.pt")
                # load theta_i + base weight from prevs
                vector_state_dict = torch.load(f"{p}/actor.pt").state_dict()
                # get theta_i
                vector_weight.append(base[j]['weight'] - vector_state_dict[f"{i}.weight"])
                vector_bias.append(base[j]['bias'] - vector_state_dict[f"{i}.bias"])
            vectors.append({"weight":torch.stack(vector_weight),
                            "bias":torch.stack(vector_bias)})
        num_weights += vectors[0]["weight"].shape[0] if vectors else 0
        return base, vectors, num_weights