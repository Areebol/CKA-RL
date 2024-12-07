import os
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from cnn_encoder import CnnEncoder
import sys, os

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class IncreParamLinearLayer(nn.Module):
    """
    $\theta_{new} = \theta_{old} + \alpha \cdot \tau + \Delta \theta$
    1. learnable parameters $\Delta \theta$, $\alpha$
    2. fixed parameters $\theta_{old}$, $\tau$
    """
    def __init__(self, model, vectors):
        """initialize IncreParamLayer with model = theta_old, vectors = tau"""
        super().__init__()
        assert(model.weight is not None)
        assert(model.bias is not None)
        self.theta_old = {"weight":model.weight.clone().detach(), "bias":model.bias.clone().detach()} # model's weights, bias
        self.tau = [{"weight" : v.weight.clone().detach(), "bias" : v.bias.clone().detach()} for v in vectors] # model's weights, bias
        # self.theta_old.requires_grad = False
        # for t in self.tau:
        #     t.requires_grad = False
        
        self.delta_theta = {"weight":nn.Parameter(torch.zeros_like(self.theta_old["weight"])), "bias":nn.Parameter(torch.zeros_like(self.theta_old["bias"]))}
        self.alpha = nn.Parameter(torch.zeros(len(vectors)))
        
    def forward(self, x):
        """return $\theta_{new}$"""
        rho = nn.functional.softmax(self.alpha)
        theta_new_weight = self.theta_old["weight"] + sum(a * t["weight"] for a, t in zip(rho, self.tau)) + self.delta_theta["weight"]
        theta_new_bias = self.theta_old["bias"] + sum(a * t["bias"] for a, t in zip(rho, self.tau)) + self.delta_theta["bias"]
        return nn.functional.linear(x, theta_new_weight, theta_new_bias)
    
    
class CnnFuseNet(nn.Module):
    def __init__(self, envs, prevs_paths=[], map_location=None):
        super().__init__()
        self.hidden_dim = 512
        self.is_first = len(prevs_paths) == 0
        self.envs = envs
        # 1. load encoder
        if not self.is_first:
            del_thetas = [
                torch.load(f"{p}/del_theta.pt", map_location=map_location)
                for p in prevs_paths[1:]
            ]
            theta_old = torch.load(f"{prevs_paths[0]}/theta_old.pt", map_location=map_location)
            # self.actor = IncreParamMLP(theta_old,del_thetas)
            print("==> Actor initialized from theta_old and del_thetas, len(del_thetas):", len(del_thetas))
        else:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self.hidden_dim, envs.single_action_space.n), std=0.01),
            )
            print("==> Encoder linitialized from sketch")
            
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.network = CnnEncoder(hidden_dim=512, layer_init=layer_init)

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
        model = CnnFuseNet(envs)
        model.network = torch.load(f"{dirname}/encoder.pt", map_location=map_location)
        if not reset_actor:
            model.actor = torch.load(f"{dirname}/actor.pt", map_location=map_location)
        if load_critic:
            model.critic = torch.load(f"{dirname}/critic.pt", map_location=map_location)
        return model


import unittest
class TestIncreParamLayer(unittest.TestCase):
    def setUp(self):
        # Initialize a simple model and vectors for testing
        self.model = nn.Linear(10, 5)
        self.vectors = [self.model for _ in range(3)]
        # self.incre_param_layer = IncreParamLinearLayer(self.model, self.vectors)

    def test_initialization(self):
        # Check if the parameters are initialized correctly
        self.assertTrue(hasattr(self.incre_param_layer, 'theta_old'))
        self.assertTrue(hasattr(self.incre_param_layer, 'tau'))
        self.assertTrue(hasattr(self.incre_param_layer, 'delta_theta'))
        self.assertTrue(hasattr(self.incre_param_layer, 'alpha'))

    def test_forward(self):
        # Test the forward pass
        x = torch.randn(1, 10)
        output = self.incre_param_layer(x)
        self.assertEqual(output.shape, (1, 5))

    def test_learnable_parameters(self):
        # Check if delta_theta and alpha are learnable parameters
        self.assertTrue(self.incre_param_layer.delta_theta["weight"].requires_grad)
        self.assertTrue(self.incre_param_layer.delta_theta["bias"].requires_grad)
        self.assertTrue(self.incre_param_layer.alpha.requires_grad)
        
        # for t in self.incre_param_layer.tau:
        #     self.assertTrue(not t.requires_grad)
        # self.assertTrue(not self.incre_param_layer.theta_old.requires_grad)
        
    def test_backward(self):
        # Test if the model can perform backward propagation
        x = torch.randn(1, 10)
        output = self.incre_param_layer(x)
        loss = output.sum()
        self.incre_param_layer.zero_grad()  # Reset gradients
        loss.backward()
        self.assertIsNotNone(self.incre_param_layer.delta_theta["weight"].grad)
        print(self.incre_param_layer.delta_theta["weight"].grad)
        self.assertIsNotNone(self.incre_param_layer.delta_theta["bias"].grad)
        print(self.incre_param_layer.delta_theta["bias"].grad)
        self.assertIsNotNone(self.incre_param_layer.alpha.grad)
        print(self.incre_param_layer.alpha.grad)

if __name__ == '__main__':
    unittest.main()