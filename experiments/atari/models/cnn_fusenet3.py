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
                 num_weights: int = 0, # 0表示刚开始训练， n表示有n个weight可用 (n>0)
                 alpha: nn.Parameter = None,
                 alpha_scale: nn.Parameter = None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.alpha_scale = alpha_scale
        self._bias = bias
        self.num_weights = num_weights
        
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False)
        self.weight_eps = Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=True)
        assert self.num_weights >= 0, "num_weights must be non-negative"
        if self.num_weights > 0:
            self.weights = Parameter(torch.stack([torch.empty((out_features, in_features)) for _ in range(num_weights)], dim=0), requires_grad=False)
        
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
            self.bias_eps = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=True)
            if self.num_weights > 0:
                self.biaes = Parameter(torch.stack([torch.empty(out_features) for _ in range(num_weights)], dim=0), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_eps', None)
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
        if self.num_weights == 0:
            """base + eps"""
            return self._forward_with_eps_only(input)
        else:
            """base + weights + eps"""
            return self._forward_with_fuse_eps_vectors(input)
    
    def _forward_with_fuse_eps_vectors(self, input: Tensor) -> Tensor:
        alphas_normalized = F.softmax(self.alpha * self.alpha_scale, dim=0)
        weight = self.weight + (alphas_normalized.view(-1, 1, 1) * self.weights).sum(dim = 0) + self.weight_eps
        if self._bias:
            bias = self.bias + (alphas_normalized.view(-1,1) * self.biaes).sum(dim=0) + self.bias_eps
        else:
            bias = None
        return F.linear(input, weight, bias)
    
    def _forward_with_eps_only(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight_eps, self.bias_eps)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'   
        
    def set_base_and_vectors(self, base, vectors):
        # Set weight and vectors
        if base is not None:
            print("Seting base")
            assert('weight' in base and 'bias' in base)
            self.weight.data.copy_(base['weight'])
            self.bias.data.copy_(base['bias'])
            
            # weight eps从0开始学习
            self.weight_eps.data.zero_()
            self.bias_eps.data.zero_()
        if vectors is not None: 
            print("Seting vectors")
            assert('weight' in vectors and 'bias' in vectors)
            assert base['weight'].shape == vectors['weight'].shape[1:], f"Shape of base {base['weight'].shape} weight and vectors weight {vectors['weight'].shape[1:]} must match"
            assert base['bias'].shape == vectors['bias'].shape[1:], f"Shape of base {base['bias'].shape} bias and vectors bias {vectors['bias'].shape[1:]} must match"
            self.weights.data.copy_(vectors['weight'])
            self.biaes.data.copy_(vectors['bias'])


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
    """return theta_0, prevs_theta when base_dir exists"""
    base = []
    vectors = []
    num_weights = 0
    if base_dir:
        print("Loading base from", f"{base_dir}/actor.pt")
        base_state_dict = torch.load(f"{base_dir}/actor.pt").state_dict()
        for i in [0,2]:
            base.append({"weight":base_state_dict[f"{i}.weight_eps"],"bias":base_state_dict[f"{i}.bias_eps"]})
    else:
        return [None,None],[None,None], 0

    for i,j in zip([0,2],[0,1]):
        """初始化 Delta theta = 0"""
        vector_weight = [torch.zeros_like(base[j]["weight"])]
        vector_bias = [torch.zeros_like(base[j]["bias"])]
        for p in prevs_paths:
            print("Loading vector from", f"{p}/actor.pt")
            vector_state_dict = torch.load(f"{p}/actor.pt").state_dict()
            vector_weight.append(vector_state_dict[f"{i}.weight_eps"])
            vector_bias.append(vector_state_dict[f"{i}.bias_eps"])
        vectors.append({"weight":torch.stack(vector_weight),
                        "bias":torch.stack(vector_bias)})
    num_weights += vectors[0]["weight"].shape[0] if vectors else 0
    return base, vectors, num_weights

class CnnFuse3Net(nn.Module):
    def __init__(self, envs, base_dir, prevs_paths=[], map_location=None):
        super().__init__()
        self.hidden_dim = 512
        self.envs = envs
        self.i = 0
        base, vectors, self.num_weights = load_actor_base_and_vectors(base_dir, prevs_paths)
        
        if self.num_weights > 0:
            self.alpha = nn.Parameter(torch.randn(self.num_weights) / self.num_weights, requires_grad=True)
            print("Alpha's shape:", self.alpha.shape)
            self.alpha_scale = nn.Parameter(torch.ones(1), requires_grad=True)
        else:
            self.alpha = None
            self.alpha_scale = None
        
        self.actor = nn.Sequential(
            layer_init(FuseLinear(512, 512, num_weights=self.num_weights, 
                                  alpha=self.alpha, alpha_scale=self.alpha_scale)),
            nn.ReLU(),
            layer_init(FuseLinear(512, envs.single_action_space.n, num_weights=self.num_weights, 
                                  alpha=self.alpha, alpha_scale=self.alpha_scale)),
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
