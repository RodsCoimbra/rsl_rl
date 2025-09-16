# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from rsl_rl.networks import Memory
from torch.distributions import Normal



class StudentTeacherRecurrentPointNet(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        encoder_lidar_dims=[64, 32, 32],
        student_hidden_dims=[400, 200],
        init_noise_std=0.1,
        proprioception_space_student=None,
        lidar_space_student=None,
        proprioception_space_teacher=None,
        latent_dim: int = 32,        
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        rnn_type="lstm",
        teacher_recurrent=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacherRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys()),
            )
        super().__init__()
        self.loaded_teacher = False  # indicates if teacher has been loaded
        self.teacher_recurrent = teacher_recurrent
        
        # student
        self.student = StudentPointNetMLP(
            proprioception_space_student, 
            encoder_lidar_dims, 
            student_hidden_dims, 
            num_actions,
            lidar_space_student,
            mlp_input_dim = rnn_hidden_dim
        )

        # teacher
        self.teacher = AgentPointNetMLP(proprioception_space_teacher, latent_dim, num_actions)
        self.teacher.eval()

        self.memory_s = Memory(self.student.rnn_input_size, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        if self.teacher_recurrent:
            self.memory_t = Memory(
                num_teacher_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim
            )

        print(f"Student MLP: {self.student}")
        print(f"Teacher MLP: {self.teacher}")

        print(f"Student RNN: {self.memory_s}")
        if self.teacher_recurrent:
            print(f"Teacher RNN: {self.memory_t}")
            
        # action noise
        if isinstance(init_noise_std, list):
            if len(init_noise_std) != num_actions:
                raise ValueError(f"init_noise_std should be a scalar or a list of length {num_actions}")
            self.std = nn.Parameter(torch.tensor(init_noise_std, dtype=torch.float32))
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
    def reset(self, dones=None, hidden_states=None):
        if hidden_states is None:
            hidden_states = (None, None)
        self.memory_s.reset(dones, hidden_states[0])
        if self.teacher_recurrent:
            self.memory_t.reset(dones, hidden_states[1])
    
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
        concat_obs = self.student.concat_input_encoder(observations)
        rnn_obs = self.memory_s(concat_obs)
        mean = self.student.forward_mlp(rnn_obs.squeeze(0))
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        concat_obs = self.student.concat_input_encoder(observations)
        rnn_obs = self.memory_s(concat_obs)
        actions_mean = self.student.forward_mlp(rnn_obs.squeeze(0))
        return actions_mean

    def evaluate(self, teacher_observations):
        if self.teacher_recurrent:
            teacher_observations = self.memory_t(teacher_observations)
        
        with torch.no_grad():
            actions = self.teacher(teacher_observations)
        return actions

    def get_hidden_states(self):
        if self.teacher_recurrent:
            return self.memory_s.hidden_states, self.memory_t.hidden_states
        else:
            return self.memory_s.hidden_states, None

    def detach_hidden_states(self, dones=None):
        self.memory_s.detach_hidden_states(dones)
        if self.teacher_recurrent:
            self.memory_t.detach_hidden_states(dones)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters.
        """

        # check if state_dict contains teacher and student or just teacher parameters
        if any("actor" in key for key in state_dict.keys()):  # loading parameters from rl training
            # rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            # also load recurrent memory if teacher is recurrent
            if self.is_recurrent and self.teacher_recurrent:
                raise NotImplementedError("Loading recurrent memory for the teacher is not implemented yet")  # TODO
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return False
        elif any("student" in key for key in state_dict.keys()):  # loading parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")


class AgentPointNetMLP(nn.Module):
    def __init__(self, input_size, latent_dim, output_size):
        super().__init__()
        self.input_size = input_size
        
        self.pointnet = PointNetEncoder(latent_dim=latent_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size + latent_dim, 400),
            nn.ELU(),
            nn.Linear(400, 200),
            nn.ELU(),
            nn.Linear(200, output_size)
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
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.LayerNorm(128),
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

class StudentPointNetMLP(nn.Module):
    def __init__(self, input_size, encoder_lidar_dims, student_hidden_dims, num_actions, lidar_space_student, mlp_input_dim):
        super().__init__()
        self.input_size = input_size
        activation_function = nn.ELU()
        
        # Lidar Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(lidar_space_student, encoder_lidar_dims[0]))
        encoder_layers.append(activation_function)
        for layer_idx in range(len(encoder_lidar_dims)):
            if layer_idx == len(encoder_lidar_dims) - 1:
                encoder_layers.append(nn.LayerNorm(encoder_lidar_dims[layer_idx]))
                continue
            encoder_layers.append(nn.Linear(encoder_lidar_dims[layer_idx], encoder_lidar_dims[layer_idx + 1]))
            encoder_layers.append(activation_function)
            
        self.lidar_encoder = nn.Sequential(*encoder_layers)
        
        
        # Student MLP
        self.rnn_input_size = input_size + encoder_lidar_dims[-1]
        student_layers = []
        student_layers.append(nn.Linear(mlp_input_dim, student_hidden_dims[0]))
        student_layers.append(activation_function)
        
        for layer_index in range(len(student_hidden_dims)):
            if layer_index == len(student_hidden_dims) - 1:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], num_actions))
            else:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                student_layers.append(activation_function)
        
        self.student_mlp = nn.Sequential(*student_layers)
    
    def forward(self, x):
        return self.student_mlp(self.concat_input_encoder(x))
    
    def forward_mlp(self, x):
        return self.student_mlp(x)
    
    def forward_encoder(self, x):
        distances_lidar = x[:, self.input_size:]
        return self.lidar_encoder(distances_lidar)
    
    def concat_input_encoder(self, x):
        proprioception = x[:, :self.input_size]
        lidar_latent = self.forward_encoder(x)
        return torch.cat([proprioception, lidar_latent], dim=1)