# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCriticPointNet(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        proprioception_space = None,
        latent_dim: int = 32,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        if proprioception_space is None:
            raise ValueError("proprioception_space must be provided")
        critic_space = num_actor_obs - num_critic_obs
        
        # Policy
        self.actor = AgentPointNetMLP(proprioception_space, latent_dim, num_actions)

        # Value function
        self.critic = AgentPointNetMLP(proprioception_space + critic_space, latent_dim, 1)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True


class AgentPointNetMLP(nn.Module):
    def __init__(self, input_size, latent_dim, output_size):
        super().__init__()
        self.input_size = input_size
        
        self.pointnet = PointNetEncoder(latent_dim=latent_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size + latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        proprioception = x[:, :self.input_size]
        points = x[:, self.input_size:]
        latent_points = self.pointnet(points)
        mlp_input = torch.cat((proprioception, latent_points), dim=1)
        
        return self.mlp(mlp_input)

class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.pointnet_mlps = nn.Sequential(
            nn.Linear(6, 64),  # Input: (B, N, 6)
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU()
        )
        
        self.final = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, pts):  
        pts = pts.view(pts.shape[0], -1, 4) 
        xyz = pts[:, :, :3] # (Num_envs, num_pts, 3)
        mask = pts[:, :, 3].long()
        mask_one_hot = nn.functional.one_hot(mask, num_classes=3).float()  
        pts_input = torch.cat([xyz, mask_one_hot], dim=-1)  # (Num_envs, num_pts, 6)

        x = self.pointnet_mlps(pts_input)
        padding_mask = (mask == 0).unsqueeze(-1)
        x = x.masked_fill(padding_mask, -1e9)
        x, _ = torch.max(x, dim=1)    # (Num_envs,256)
        return self.final(x)  # (Num_envs,latent_dim)