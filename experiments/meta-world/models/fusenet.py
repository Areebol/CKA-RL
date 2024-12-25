import torch
import torch.nn as nn
import os
from .fuse_module import FuseShared, FuseLinear
from .shared_arch import shared
from loguru import logger
import numpy as np

class FuseNetAgent(nn.Module):
    def __init__(self, 
                 obs_dim, 
                 act_dim, 
                 base_dir, 
                 prevs_paths, 
                 delta_theta_mode = "T", 
                 global_alpha = True, 
                 alpha_init = "Randn", 
                 alpha_major = 0.6, 
                 alpha_factor = 1e-3,
                 fix_alpha = False,
                 reset_heads = False,
                 use_alpha_scale = True,
                 fuse_shared = True, 
                 fuse_heads = True,):
        super().__init__()
        self.delta_theta_mode = delta_theta_mode
        self.global_alpha = global_alpha
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.num_weights = len(prevs_paths)
        self.fuse_shared = fuse_shared
        self.fuse_heads = fuse_heads

        assert(fuse_heads or fuse_shared)
       # Alpha Setting
        if self.num_weights > 0:
            if fix_alpha: # Alpha is untrainable
                self.alpha = nn.Parameter(torch.zeros(self.num_weights), requires_grad=False)
                logger.info("Fix alpha to all 0")
            else: # Alpha is trainable
                logger.info(f"alpha_init, {alpha_init}")
                logger.info(f"alpha_major, {alpha_major}")
                if alpha_init == "Uniform" or self.num_weights == 1:
                    self.alpha = nn.Parameter(torch.ones(self.num_weights) * alpha_factor, requires_grad=True)
                elif alpha_init == "Randn":
                    self.alpha = nn.Parameter(torch.randn(self.num_weights) / self.num_weights, requires_grad=True)
                elif alpha_init == "Major" and self.num_weights > 1:
                    alpha = [np.log((1-alpha_major)/(self.num_weights-1)) for _ in range(self.num_weights-1)]
                    alpha.append(np.log(alpha_major))
                    self.alpha = nn.Parameter(torch.tensor(alpha,dtype=torch.float), requires_grad=True)
                    logger.info(self.alpha)
                elif alpha_init not in ["Uniform", "Randn", "Major"]:
                    raise NotImplementedError
                self.alpha_scale = nn.Parameter(torch.ones(1), requires_grad=True)
                logger.info("Train alpha")
            if not use_alpha_scale or fix_alpha:
                self.alpha_scale = nn.Parameter(torch.ones(1), requires_grad=False)
            # logger.info(f"Alpha's shape: {self.alpha.shape}, Alpha: {self.alpha.data}, Alpha scale: {self.alpha_scale.data}")
        else:
            self.alpha = None
            self.alpha_scale = None
        if self.fuse_shared:
            self.fc = FuseShared(input_dim=obs_dim, alpha=self.alpha, alpha_scale=self.alpha_scale, num_weights=self.num_weights)
        else:
            self.fc = shared(input_dim=obs_dim)

        self.reset_heads()
        self.set_base_and_vectors(base_dir, prevs_paths)

    def reset_heads(self):
        # will be created when calling `reset_heads`
        if self.fuse_heads:
            self.fc_mean = FuseLinear(256, 
                                    self.act_dim, alpha=self.alpha, 
                                    alpha_scale=self.alpha_scale, 
                                    num_weights=self.num_weights)
            self.fc_logstd = FuseLinear(256, 
                                        self.act_dim, alpha=self.alpha, 
                                        alpha_scale=self.alpha_scale, 
                                        num_weights=self.num_weights)
        else:
            self.fc_mean = nn.Linear(256, self.act_dim)
            self.fc_logstd = nn.Linear(256, self.act_dim)
        

    def load_base_and_vectors(self, base_dir, vector_dirs, module_name):
        num_weights = 0
        base = None
        vectors = None
        if base_dir:
            # load base weight
            logger.info(f"Loading base from {base_dir}/model.pt")
            base_state_dict = torch.load(f"{base_dir}/model.pt").state_dict()
            base = {"weight":base_state_dict[f"{module_name}.weight"],"bias":base_state_dict[f"{module_name}.bias"]}
        else:
            return None, None

        vector_weight = []
        vector_bias = []
        for p in vector_dirs:
            logger.debug(f"Loading vectors from {p}/model.pt")
            # load theta_i + base weight from prevs
            vector_state_dict = torch.load(f"{p}/model.pt").state_dict()
            # get theta_i
            vector_weight.append(base['weight'] - vector_state_dict[f"{module_name}.weight"])
            vector_bias.append(base['bias'] - vector_state_dict[f"{module_name}.bias"])
        vectors = {"weight":torch.stack(vector_weight),
                        "bias":torch.stack(vector_bias)}
        num_weights += vectors["weight"].shape[0] if vectors else 0
        return base, vectors

    def heads_set_base_and_vectors(self, base_dir, prevs_paths):
        for module_name in ["fc_mean", "fc_logstd"]:
            base, vectors = self.load_base_and_vectors(base_dir, prevs_paths, module_name)
            if base is None:
                continue
            getattr(self, module_name).set_base_and_vectors(base, vectors)

    def set_base_and_vectors(self, base_dir, prevs_paths):
        if self.fuse_shared:
            self.fc.set_base_and_vectors(base_dir, prevs_paths)
        if self.fuse_heads:
            self.heads_set_base_and_vectors(base_dir, prevs_paths)

    def forward(self, x):
        x = self.fc(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        return mean, log_std

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        # for actor, merge `theta + alpha * tau` to `theta` if delta_theta_mode  == 'TAT'
        if self.delta_theta_mode == "TAT":
            self.merge_weight()
        else:
            logger.info("save weight as theta")
        torch.save(self, f"{dirname}/model.pt")

    def load(dirname, map_location=None, reset_heads=False):
        model = torch.load(f"{dirname}/model.pt", map_location=map_location)
        if reset_heads:
            model.reset_heads()
        return model

    def merge_weight(self):
        if self.fuse_shared:
            self.fc.merge_weight()
        if self.fuse_heads:
            self.fc_mean.merge_weight()
            self.fc_logstd.merge_weight()