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
from torch.nn.modules.conv import _ConvNd
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from typing import Optional, List, Tuple, Union


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

    @torch.no_grad()
    def merge_weight(self):
        if self.num_weights <= 0:
            logger.debug("Not weights or alpha exists, return original weight")
            return
        # logger.debug(f"Merging FuseLinear: {self.weight.shape} + {self.weights.shape} * {self.alpha.shape}")
        alphas_normalized = F.softmax(self.alpha * self.alpha_scale, dim=0)
        # weight = self.weight.data # debug
        self.weight.data = self.weight.data + (alphas_normalized.view(-1, 1, 1) * self.weights.data).sum(dim = 0)
        if self._bias:
            self.bias.data = self.bias.data + (alphas_normalized.view(-1,1) * self.biaes.data).sum(dim=0)
        # logger.debug(weight == self.weight.data) # debug

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
            # logger.debug(f"Setting base with tensor's shape = {base['weight'].shape}")
            assert('weight' in base and 'bias' in base)
            self.weight.data.copy_(base['weight'])
            self.bias.data.copy_(base['bias'])
        else:
            # logger.debug(f"Base is None, train base weight from scratch")
            return
            
        # Set vectors weight
        if vectors is not None: 
            # logger.debug(f"Setting vectors with tensor's shape = {vectors['weight'].shape}")
            assert('weight' in vectors and 'bias' in vectors)
            assert base['weight'].shape == vectors['weight'].shape[1:], f"Shape of base {base['weight'].shape} weight and vectors weight {vectors['weight'].shape[1:]} must match"
            assert base['bias'].shape == vectors['bias'].shape[1:], f"Shape of base {base['bias'].shape} bias and vectors bias {vectors['bias'].shape[1:]} must match"
            
            self.weights.data.copy_(vectors['weight'])
            self.biaes.data.copy_(vectors['bias'])

class FuseConv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        num_weights: int = 0, # 0 = train base weight， n = train base weight + alpha * tau
        alpha: nn.Parameter = None,
        alpha_scale: nn.Parameter = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
        self.alpha = alpha # given by agent
        self.alpha_scale = alpha_scale # given by agnet
        self._bias = bias
        self.num_weights = num_weights # size of tau

        if self.num_weights > 0:
            # alpha need to match num_weights
            assert(self.alpha.shape[0] == self.num_weights)
        assert self.num_weights >= 0, "num_weights must be non-negative"
        
        # tau = {theta_0,theta_1,...theta_n}
        if self.num_weights > 0:
            self.weights = Parameter(torch.stack([torch.zeros_like((self.weight)) for _ in range(num_weights)], dim=0), requires_grad=False)
        else:
            self.weights = None
        
        if bias:
            # tau = {theta_0,theta_1,...theta_n}
            if self.num_weights > 0:
                self.biaes = Parameter(torch.stack([torch.zeros_like(self.bias) for _ in range(num_weights)], dim=0), requires_grad=False)
            else:
                self.biaes = None

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        # alpha * tau {theta_0,theta_1,...theta_n} + base
        if self.alpha is not None:
            # logger.debug(f"Alpha is {self.alpha.data}, forward with alpha * tau")
            alphas_normalized = F.softmax(self.alpha * self.alpha_scale, dim=0)
            weight = self.weight + (alphas_normalized.view(-1, 1, 1, 1, 1) * self.weights).sum(dim = 0)
            if self._bias:
                bias = self.bias + (alphas_normalized.view(-1,1) * self.biaes).sum(dim=0)
            else:
                bias = None
        else:
            # logger.debug("Alpha is None, forward with base weight only")
            weight = self.weight
            if self._bias:
                bias = self.bias
                
        return self._conv_forward(input, weight, bias)

    @torch.no_grad()
    def merge_weight(self):
        if self.num_weights <= 0:
            logger.debug("Not weights or alpha exists, return original weight")
            return
        # logger.debug(f"Merging FuseConv: {self.weight.shape} + {self.weights.shape} * {self.alpha.shape}")
        alphas_normalized = F.softmax(self.alpha * self.alpha_scale, dim=0)
        # weight = self.weight.data # debug
        self.weight.data = self.weight.data + (alphas_normalized.view(-1, 1, 1, 1, 1) * self.weights.data).sum(dim = 0)
        if self._bias:
            self.bias.data = self.bias.data + (alphas_normalized.view(-1,1) * self.biaes.data).sum(dim=0)
        # logger.debug(weight == self.weight.data) # debug

    def set_base_and_vectors(self, base, vectors):
        # Set base weight
        if base is not None:
            # logger.debug(f"Setting base with tensor's shape = {base['weight'].shape}")
            assert('weight' in base and 'bias' in base)
            self.weight.data.copy_(base['weight'])
            self.bias.data.copy_(base['bias'])
        else:
            logger.debug(f"Base is None, train base weight from scratch")
            
        # Set vectors weight
        if vectors is not None: 
            # logger.debug(f"Setting vectors with tensor's shape = {vectors['weight'].shape}")
            assert('weight' in vectors and 'bias' in vectors)
            assert base['weight'].shape == vectors['weight'].shape[1:], f"Shape of base {base['weight'].shape} weight and vectors weight {vectors['weight'].shape[1:]} must match"
            assert base['bias'].shape == vectors['bias'].shape[1:], f"Shape of base {base['bias'].shape} bias and vectors bias {vectors['bias'].shape[1:]} must match"
            
            self.weights.data.copy_(vectors['weight'])
            self.biaes.data.copy_(vectors['bias'])

class FuseEncoder(nn.Module):
    def __init__(self, hidden_dim=512, layer_init=lambda x, **kwargs: x,
                    num_weights: int = 0, # 0 = train base weight， n = train base weight + alpha * tau
                    alpha: nn.Parameter = None,
                    alpha_scale: nn.Parameter = None,):
        super().__init__()
        self.fuse_layers = [0,2,4,7]
        self.network = nn.Sequential(
            layer_init(FuseConv2d(4, 32, 8, stride=4, alpha=alpha, alpha_scale=alpha_scale,num_weights=num_weights)), # 0
            nn.ReLU(),
            layer_init(FuseConv2d(32, 64, 4, stride=2, alpha=alpha, alpha_scale=alpha_scale,num_weights=num_weights)), # 2
            nn.ReLU(),
            layer_init(FuseConv2d(64, 64, 3, stride=1, alpha=alpha, alpha_scale=alpha_scale,num_weights=num_weights)), # 4
            nn.ReLU(),
            nn.Flatten(),
            layer_init(FuseLinear(64 * 7 * 7, hidden_dim,alpha=alpha, alpha_scale=alpha_scale,num_weights=num_weights)), # 7
            nn.ReLU(),
        )
     
    def load_base_and_vectors(self, base_dir, vector_dirs):   
        base = []
        vectors = []
        num_weights = 0
        if base_dir:
            # load base weight
            logger.info(f"Loading base from {base_dir}/encoder.pt")
            base_state_dict = torch.load(f"{base_dir}/encoder.pt").state_dict()
            prefix = list(base_state_dict.keys())[0].split('.')[0]
            # logger.debug(prefix)
            for i in self.fuse_layers:
                base.append({"weight":base_state_dict[f"{prefix}.{i}.weight"],"bias":base_state_dict[f"{prefix}.{i}.bias"]})
        else:
            return [None,None],[None,None]

        for idx,i in enumerate(self.fuse_layers):
            vector_weight = []
            vector_bias = []
            for p in vector_dirs:
                if idx == 0:
                    logger.debug(f"Loading vectors from {p}/encoder.pt")
                # load theta_i + base weight from prevs
                vector_state_dict = torch.load(f"{p}/encoder.pt").state_dict()
                # get theta_i
                vector_weight.append(base[idx]['weight'] - vector_state_dict[f"{prefix}.{i}.weight"])
                vector_bias.append(base[idx]['bias'] - vector_state_dict[f"{prefix}.{i}.bias"])
            vectors.append({"weight":torch.stack(vector_weight),
                            "bias":torch.stack(vector_bias)})
        num_weights += vectors[0]["weight"].shape[0] if vectors else 0
        return base, vectors 
        
    def set_base_and_vectors(self, base_dir, prevs_paths):
        base, vectors = self.load_base_and_vectors(base_dir, prevs_paths)
        if base[0] is None:
            logger.warning("Not base or vectors exist")
            return 
        logger.debug("Setting FuseEncoder's weight and vectors")
        for idx,i in enumerate(self.fuse_layers):
            self.network[i].set_base_and_vectors(base[idx],vectors[idx])
        
    def forward(self, x):
        return self.network(x)
    
    def merge_weight(self):
        for i in self.fuse_layers:
            self.network[i].merge_weight()

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
                 delta_theta_mode: str = "T",
                 fuse_encoder: bool = False,
                 fuse_actor: bool = True,
                 map_location=None):
        super().__init__()
        self.delta_theta_mode = delta_theta_mode
        self.fuse_encoder = fuse_encoder
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
        
        # Actor 's fuse 
        self.actor = nn.Sequential(
            layer_init(FuseLinear(512, 512, num_weights=self.num_weights, 
                                  alpha=self.alpha, alpha_scale=self.alpha_scale)),
            nn.ReLU(),
            layer_init(FuseLinear(512, envs.single_action_space.n, num_weights=self.num_weights, 
                                  alpha=self.alpha, alpha_scale=self.alpha_scale),std=0.01),
        )
        self.actor[0].set_base_and_vectors(actor_base[0], actor_vectors[0])
        self.actor[2].set_base_and_vectors(actor_base[1], actor_vectors[1])
        
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
            if encoder_dir is not None:
                logger.info(f"Loading encoder from {encoder_dir}")
                self.network = torch.load(f"{encoder_dir}/encoder.pt", map_location=map_location)
            else:
                logger.info("No Encoder exists, Train encoder from scratch")
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
        for j in [0,2]:
            self.actor[j].merge_weight()

    def merge_encoder_weight(self):
        if self.alpha is None or self.fuse_encoder is False:
            logger.warning("Not alpha exist in FuseNetAgent or Not Fuse Encoder, not merge")
            return 
        
        logger.info("merge encoder's weight")
        self.network.merge_weight()

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

        for idx,i in enumerate([0,2]):
            vector_weight = []
            vector_bias = []
            for p in vector_dirs:
                if idx == 0:
                    logger.debug(f"Loading vectors from {p}/actor.pt")
                # load theta_i + base weight from prevs
                vector_state_dict = torch.load(f"{p}/actor.pt").state_dict()
                # get theta_i
                vector_weight.append(base[idx]['weight'] - vector_state_dict[f"{i}.weight"])
                vector_bias.append(base[idx]['bias'] - vector_state_dict[f"{i}.bias"])
            vectors.append({"weight":torch.stack(vector_weight),
                            "bias":torch.stack(vector_bias)})
        num_weights += vectors[0]["weight"].shape[0] if vectors else 0
        return base, vectors, num_weights