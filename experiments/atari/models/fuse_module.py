import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter, ParameterList, init
from torch import Tensor
import math

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class FuseLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, 
                 out_features: int, 
                 bias: bool = True, 
                 num_weights: int = -1, # -1 表示刚开始训练， 0表示没有weight可用， n表示有n个weight可用 (n>0)
                 alpha: nn.Parameter = None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self._bias = bias
        self.num_weights = num_weights
        
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False)
        self.weight_eps = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if self.num_weights > 0:
            self.weights = Parameter(torch.stack([torch.empty((out_features, in_features)) for _ in range(num_weights)], dim=0), requires_grad=False)
        
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
            self.bias_eps = Parameter(torch.empty(out_features, **factory_kwargs))
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
        if self.num_weights == -1:
            """eps"""
            return self._forward_with_eps_only(input)
        elif self.num_weights == 0:
            """base + eps"""
            return self._forward_with_fuse_eps(input)
        else:
            """base + weights + eps"""
            return self._forward_with_fuse_eps_vectors(input)
    
    def _forward_with_fuse_eps_vectors(self, input: Tensor) -> Tensor:
        alphas_normalized = F.softmax(self.alpha, dim=0)
        weight = self.weight + (alphas_normalized.view(-1, 1, 1) * self.weights).sum(dim = 0) + self.weight_eps
        if self._bias:
            bias = self.bias + (alphas_normalized.view(-1,1) * self.biaes).sum(dim=0) + self.bias_eps
        else:
            bias = None
        return F.linear(input, weight, bias)
    
    def _forward_with_fuse_eps(self, input: Tensor) -> Tensor:
        weight = self.weight + self.weight_eps
        if self._bias:
            bias = self.bias + self.bias_eps
        else:
            bias = None
        return F.linear(input, weight, bias)
    
    def _forward_with_eps_only(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight_eps, self.bias_eps)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'   
        
    def set_base_and_vectors(self, base):
        # TODO: set weight and vectors
        ...