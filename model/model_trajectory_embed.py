import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class ResBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        activation=Swish(),
        layer_norm=True,
        with_residual=True,
        dropout=0.0
    ):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.with_residual = with_residual
    
    def forward(self, x):
        y = self.activation(self.linear(x))
        if self.dropout is not None:
            y = self.dropout(y)
        if self.with_residual:
            y = x + y
        if self.layer_norm is not None:
            y = self.layer_norm(y)
        return y

class RobotGraspModel(nn.Module):
    def __init__(self, image_shape: tuple, embed_dim: int, window_size: int, qpos_dim: int, output_type: str, action_chunk: int):
        super(RobotGraspModel, self).__init__()
        
        self.in_channels = image_shape[0]
        self.image_size = image_shape[1:]
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.qpos_dim = qpos_dim
        self.output_dim = 7 if output_type == 'position' else 8
        self.action_chunk = action_chunk
        self.output_dim = self.output_dim * self.action_chunk
        
        if output_type == 'position':
            self.out_min = torch.tensor([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 0.0])
            self.out_max = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.08])
        elif output_type == 'ee_pose':
            self.out_min = torch.tensor([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.out_max = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.out_min = self.out_min.repeat(self.action_chunk, 1).view(-1)
        self.out_max = self.out_max.repeat(self.action_chunk, 1).view(-1)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels*self.window_size, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )
        
        with torch.no_grad():
            dummy_input = torch.randn(1, self.in_channels*self.window_size, self.image_size[0], self.image_size[1])
            h = self.cnn(dummy_input)
            self.h_dim = h.view(1, -1).size(1)
        
        self.fc1 = nn.Sequential(
            nn.Linear(self.h_dim, self.embed_dim, bias=True),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.qpos_dim*self.window_size, self.embed_dim, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # Output projection to position dimension
        self.output_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(inplace=True),
            ResBlock(self.embed_dim, self.embed_dim),
            ResBlock(self.embed_dim, self.embed_dim),
            nn.Linear(self.embed_dim, self.output_dim),
            nn.Tanh()
        )
        
    def forward(self, frames, qpos):
        """
        Args:
            frames: (batch_size, window_size, channels, height, width)
            qpos: (batch_size, window_size, qpos_dim)
        Returns:
            output: (batch_size, output_dim)
        """
        self.out_min = self.out_min.to(frames.device)
        self.out_max = self.out_max.to(frames.device)
        
        batch_size, window_size, channels, height, width = frames.shape
        frames = frames.view(batch_size, window_size*channels, height, width)
        qpos = qpos.view(batch_size, window_size*self.qpos_dim)
        
        h = self.cnn(frames)
        h = h.view(batch_size, self.h_dim)
        h1 = self.fc1(h)
        h2 = self.fc2(qpos)
        
        last_input = torch.cat([h1, h2], dim=-1)
        output = self.output_proj(last_input)
        
        output = (output + 1) / 2 * (self.out_max - self.out_min) + self.out_min
        return output


class RobotGraspModelWithTrajectoryEmbed(nn.Module):
    def __init__(self, image_shape: tuple, embed_dim: int, window_size: int, qpos_dim: int, 
                 output_type: str, action_chunk: int, num_trajectories: int = 1000):
        super(RobotGraspModelWithTrajectoryEmbed, self).__init__()
        
        self.in_channels = image_shape[0]
        self.image_size = image_shape[1:]
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.qpos_dim = qpos_dim
        self.output_type = output_type
        self.action_chunk = action_chunk
        self.num_trajectories = num_trajectories
        
        # Calculate output dimension
        self.output_dim = 7 if output_type == 'position' else 8
        self.output_dim = self.output_dim * self.action_chunk
        
        if output_type == 'position':
            self.out_min = torch.tensor([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 0.0])
            self.out_max = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.08])
        elif output_type == 'ee_pose':
            self.out_min = torch.tensor([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.out_max = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.out_min = self.out_min.repeat(self.action_chunk, 1).view(-1)
        self.out_max = self.out_max.repeat(self.action_chunk, 1).view(-1)
        
        # Trajectory embedding matrix: (num_trajectories, 4) - [mean_x, mean_y, std_x, std_y]
        # Initialize with small random values
        self.trajectory_embeddings = nn.Parameter(
            torch.cat([
                torch.randn(self.num_trajectories, 2) * 0.1,  # mean
                torch.ones(self.num_trajectories, 2) * 0.1 + 0.05  # std (ensure positive)
            ], dim=1)
        )
        
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels*self.window_size, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )
        
        with torch.no_grad():
            dummy_input = torch.randn(1, self.in_channels*self.window_size, self.image_size[0], self.image_size[1])
            h = self.cnn(dummy_input)
            self.h_dim = h.view(1, -1).size(1)
        
        self.fc1 = nn.Sequential(
            nn.Linear(self.h_dim, self.embed_dim, bias=True),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.qpos_dim*self.window_size, self.embed_dim, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # Output projection (now includes 2D trajectory embedding)
        self.output_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + 2, self.embed_dim),  # +2 for trajectory embedding
            nn.ReLU(inplace=True),
            ResBlock(self.embed_dim, self.embed_dim),
            ResBlock(self.embed_dim, self.embed_dim),
            nn.Linear(self.embed_dim, self.output_dim),
            nn.Tanh()
        )
        
    def sample_trajectory_embedding(self, trajectory_indices: torch.Tensor) -> torch.Tensor:
        """
        Sample 2D trajectory embeddings based on trajectory indices
        Args:
            trajectory_indices: (batch_size,) trajectory indices
        Returns:
            sampled_embeddings: (batch_size, 2) sampled 2D embeddings
        """
        # Get embedding parameters for the trajectories
        traj_params = self.trajectory_embeddings[trajectory_indices]  # (batch_size, 4)
        
        # Split into mean and std
        mean = traj_params[:, :2]  # (batch_size, 2)
        std = torch.abs(traj_params[:, 2:]) + 1e-6  # (batch_size, 2), ensure positive
        
        # Sample using reparameterization trick
        eps = 2*torch.randn_like(mean)
        sampled = mean + eps * std
        
        # Clip to [-1, 1]
        # sampled = torch.tanh(sampled)
        sampled = torch.clamp(sampled, -1, 1)

        
        return sampled
    
    def forward(self, frames, qpos, trajectory_indices=None):
        """
        Args:
            frames: (batch_size, window_size, channels, height, width)
            qpos: (batch_size, window_size, qpos_dim)
            trajectory_indices: (batch_size,) trajectory indices
        Returns:
            output: (batch_size, output_dim)
        """
        self.out_min = self.out_min.to(frames.device)
        self.out_max = self.out_max.to(frames.device)
        
        batch_size, window_size, channels, height, width = frames.shape
        frames = frames.view(batch_size, window_size*channels, height, width)
        qpos = qpos.view(batch_size, window_size*self.qpos_dim)
        
        # CNN feature extraction
        h = self.cnn(frames)
        h = h.view(batch_size, self.h_dim)
        h1 = self.fc1(h)
        h2 = self.fc2(qpos)
        
        # Sample trajectory embedding
        if trajectory_indices is not None:
            traj_embed = self.sample_trajectory_embedding(trajectory_indices)  # (batch_size, 2)
        else:
            traj_embed = 2 * torch.randn(batch_size, 2) - 1.0
        
        # Concatenate all features
        last_input = torch.cat([h1, h2, traj_embed], dim=-1)
        
        # Output projection
        output = self.output_proj(last_input)
        
        # Apply output scaling
        output = (output + 1) / 2 * (self.out_max - self.out_min) + self.out_min
        return output
    
    def get_trajectory_embeddings_info(self, trajectory_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get mean and std for trajectory embeddings
        Returns:
            mean: (batch_size, 2)
            std: (batch_size, 2)
        """
        traj_params = self.trajectory_embeddings[trajectory_indices]
        mean = traj_params[:, :2]
        std = torch.abs(traj_params[:, 2:]) + 1e-6
        return mean, std 