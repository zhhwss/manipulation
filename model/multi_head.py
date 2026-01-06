import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        activation=None,
        layer_norm=True,
        with_residual=True,
        dropout=0.0
    ):
        super().__init__()
        if activation is None:
            activation = Swish()

        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.with_residual = with_residual and (input_dim == output_dim)
    
    def forward(self, x):
        y = self.activation(self.linear(x))
        if self.dropout is not None:
            y = self.dropout(y)
        if self.with_residual:
            y = x + y
        if self.layer_norm is not None:
            y = self.layer_norm(y)
        return y

class MultiModalRobotGraspModel(nn.Module):
    """
    Multi-modal model that processes:
    - Point cloud data (first 3 channels)
    - Grayscale image (4th channel)
    
    Dynamically weights outputs based on point cloud availability.
    """
    def __init__(self, image_shape: tuple, embed_dim: int, window_size: int, qpos_dim: int, output_type: str, action_chunk: int):
        super(MultiModalRobotGraspModel, self).__init__()
        
        assert image_shape[0] == 4, "Expected 4 channels: 3 for point cloud + 1 for grayscale"
        
        self.image_size = image_shape[1:]
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.qpos_dim = qpos_dim
        self.action_dim = 7 if output_type == 'position' else 8
        self.action_chunk = action_chunk
        self.output_dim = self.action_dim * self.action_chunk
        
        # Output bounds
        if output_type == 'position':
            self.register_buffer('out_min', torch.tensor([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 0.0]).repeat(self.action_chunk))
            self.register_buffer('out_max', torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.08]).repeat(self.action_chunk))
        elif output_type == 'ee_pose':
            self.register_buffer('out_min', torch.tensor([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).repeat(self.action_chunk))
            self.register_buffer('out_max', torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).repeat(self.action_chunk))
        
        # Point cloud branch (3 channels)
        self.pointcloud_cnn = self._create_cnn_branch(3 * self.window_size, 'pointcloud')
        
        # Grayscale branch (1 channel)  
        self.grayscale_cnn = self._create_cnn_branch(1 * self.window_size, 'grayscale')
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_pc = torch.randn(1, 3 * self.window_size, self.image_size[0], self.image_size[1])
            dummy_gray = torch.randn(1, 1 * self.window_size, self.image_size[0], self.image_size[1])
            
            pc_feat = self.pointcloud_cnn(dummy_pc)
            gray_feat = self.grayscale_cnn(dummy_gray)
            
            self.pc_feat_dim = pc_feat.view(1, -1).size(1)
            self.gray_feat_dim = gray_feat.view(1, -1).size(1)
        
        # qpos processing (shared)
        self.qpos_fc = nn.Sequential(
            nn.Linear(self.qpos_dim * self.window_size, self.embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Point cloud branch feature processing
        self.pc_fc = nn.Sequential(
            nn.Linear(self.pc_feat_dim, self.embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Grayscale branch feature processing
        self.gray_fc = nn.Sequential(
            nn.Linear(self.gray_feat_dim, self.embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Point cloud output head
        self.pc_head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),  # vision + qpos
            nn.ReLU(inplace=True),
            ResBlock(self.embed_dim, self.embed_dim),
            ResBlock(self.embed_dim, self.embed_dim),
            nn.Linear(self.embed_dim, self.output_dim),
            nn.Tanh()
        )
        
        # Grayscale output head
        self.gray_head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),  # vision + qpos
            nn.ReLU(inplace=True),
            ResBlock(self.embed_dim, self.embed_dim),
            ResBlock(self.embed_dim, self.embed_dim),
            nn.Linear(self.embed_dim, self.output_dim),
            nn.Tanh()
        )
        
        # Temporal smoothing layers for action chunks
        # These layers operate on the time dimension to make outputs smoother
        self.temporal_smoother = nn.Sequential(
            # First layer: smooth local fluctuations with small kernel
            nn.Conv1d(self.action_dim, self.action_dim, kernel_size=3, padding=1, groups=self.action_dim, bias=False),
            nn.BatchNorm1d(self.action_dim),
            nn.ReLU(inplace=True),
            
            # Second layer: smooth medium-range dependencies
            nn.Conv1d(self.action_dim, self.action_dim, kernel_size=5, padding=2, groups=self.action_dim, bias=False),
            nn.BatchNorm1d(self.action_dim),
            nn.ReLU(inplace=True),
            
            # Third layer: smooth long-range trends with larger kernel
            nn.Conv1d(self.action_dim, self.action_dim, kernel_size=7, padding=3, groups=self.action_dim, bias=False),
            nn.BatchNorm1d(self.action_dim),
        )
        
        # Initialize smoothing kernels to act as low-pass filters
        self._init_smoothing_kernels()
    
    def _init_smoothing_kernels(self):
        """Initialize smoothing convolution kernels as Gaussian low-pass filters"""
        import math
        
        with torch.no_grad():
            conv_layers = [self.temporal_smoother[0], self.temporal_smoother[3], self.temporal_smoother[6]]
            sigmas = [1.0, 1.5, 0.8]  # Different sigma values for different layers
            
            for conv_layer, sigma in zip(conv_layers, sigmas):
                if hasattr(conv_layer, 'weight'):
                    kernel_size = conv_layer.kernel_size[0]
                    # Create Gaussian kernel
                    x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
                    gaussian_kernel = torch.exp(-0.5 * (x / sigma) ** 2)
                    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
                    
                    # Apply to each channel (groups=action_dim)
                    for i in range(self.action_dim):
                        conv_layer.weight[i, 0, :] = gaussian_kernel
                sigma = kernel_size / 6.0  # Standard deviation for Gaussian
                center = kernel_size // 2
                
                with torch.no_grad():
                    for i in range(module.weight.size(0)):  # For each output channel
                        # Create 1D Gaussian kernel
                        kernel = torch.zeros(kernel_size)
                        for j in range(kernel_size):
                            kernel[j] = math.exp(-0.5 * ((j - center) / sigma) ** 2)
                        kernel = kernel / kernel.sum()  # Normalize
                        
                        # Set the kernel for this channel (depthwise convolution)
                        module.weight[i, 0] = kernel
    
    def _create_cnn_branch(self, input_channels: int, branch_name: str):
        """Create CNN branch optimized for GPU inference"""
        return nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth block
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Final block
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # Fixed spatial size for consistent output
        )
    
    def _compute_pointcloud_mask_ratio(self, pointcloud_data):
        """
        Efficiently compute non-zero ratio for point cloud data.
        Only checks the first channel since all 3 channels share the same mask.
        
        Args:
            pointcloud_data: (batch_size, 3*window_size, height, width)
        Returns:
            ratios: (batch_size,) - non-zero ratio for each sample
        """
        # Take only the first channel (assuming all 3 PC channels share the same mask)
        first_channel = pointcloud_data[:, 0:1]  # (batch_size, 1, height, width)
        
        # Count non-zero pixels efficiently
        non_zero_mask = (first_channel != 0.0)  # (batch_size, 1, height, width)
        non_zero_count = torch.sum(non_zero_mask, dim=(1, 2, 3), dtype=torch.float32)  # (batch_size,)
        
        # Total pixels per sample
        total_pixels = first_channel.shape[1] * first_channel.shape[2] * first_channel.shape[3]
        
        # Compute ratio
        ratios = non_zero_count / total_pixels  # (batch_size,)
        
        return ratios
    
    def forward(self, frames, qpos):
        """
        Args:
            frames: (batch_size, window_size, 4, height, width) - first 3 channels: point cloud, 4th: grayscale
            qpos: (batch_size, window_size, qpos_dim)
        Returns:
            output: (batch_size, output_dim)
        """
        batch_size, window_size, channels, height, width = frames.shape
        
        # Reshape for CNN processing
        frames_flat = frames.view(batch_size, window_size * channels, height, width)
        qpos_flat = qpos.view(batch_size, window_size * self.qpos_dim)
        
        # Split into point cloud and grayscale
        pointcloud_data = frames_flat[:, :3*window_size]  # (batch_size, 3*window_size, height, width)
        grayscale_data = frames_flat[:, 3*window_size:]   # (batch_size, 1*window_size, height, width)
        
        # Compute point cloud mask ratio efficiently
        pc_ratios = self._compute_pointcloud_mask_ratio(pointcloud_data)  # (batch_size,)
        
        # Process qpos (shared)
        qpos_features = self.qpos_fc(qpos_flat)  # (batch_size, embed_dim)
        
        # Process point cloud branch
        pc_cnn_features = self.pointcloud_cnn(pointcloud_data)  # (batch_size, 256, 4, 4)
        pc_cnn_features = pc_cnn_features.view(batch_size, -1)  # (batch_size, pc_feat_dim)
        pc_features = self.pc_fc(pc_cnn_features)  # (batch_size, embed_dim)
        pc_combined = torch.cat([pc_features, qpos_features], dim=-1)  # (batch_size, embed_dim*2)
        pc_output = self.pc_head(pc_combined)  # (batch_size, action_dim * action_chunk)
        
        # Apply temporal smoothing to point cloud output
        pc_output_reshaped = pc_output.view(batch_size, self.action_dim, self.action_chunk)  # (batch_size, action_dim, action_chunk)
        pc_output_smooth = self.temporal_smoother(pc_output_reshaped)  # (batch_size, action_dim, action_chunk)
        pc_output_smooth = pc_output_smooth.view(batch_size, -1)  # (batch_size, output_dim)
        
        # Process grayscale branch  
        gray_cnn_features = self.grayscale_cnn(grayscale_data)  # (batch_size, 256, 4, 4)
        gray_cnn_features = gray_cnn_features.view(batch_size, -1)  # (batch_size, gray_feat_dim)
        gray_features = self.gray_fc(gray_cnn_features)  # (batch_size, embed_dim)
        gray_combined = torch.cat([gray_features, qpos_features], dim=-1)  # (batch_size, embed_dim*2)
        gray_output = self.gray_head(gray_combined)  # (batch_size, action_dim * action_chunk)
        
        # Apply temporal smoothing to grayscale output
        gray_output_reshaped = gray_output.view(batch_size, self.action_dim, self.action_chunk)  # (batch_size, action_dim, action_chunk)
        gray_output_smooth = self.temporal_smoother(gray_output_reshaped)  # (batch_size, action_dim, action_chunk)
        gray_output_smooth = gray_output_smooth.view(batch_size, -1)  # (batch_size, output_dim)
        
        # Compute weights based on point cloud availability
        # If ratio > 10%, use average. Otherwise, use ratio/0.1 as point cloud weight
        use_average = pc_ratios > 0.1  # (batch_size,)
        pc_weights = torch.where(use_average, 
                                torch.tensor(0.5, device=frames.device), 
                                pc_ratios / 0.1)  # (batch_size,)
        gray_weights = 1.0 - pc_weights  # (batch_size,)
        
        # Expand weights to match output dimensions
        pc_weights = pc_weights.unsqueeze(1)    # (batch_size, 1)
        gray_weights = gray_weights.unsqueeze(1)  # (batch_size, 1)
        
        # Weighted combination using smoothed outputs
        combined_output = pc_weights * pc_output_smooth + gray_weights * gray_output_smooth  # (batch_size, output_dim)
        
        # Apply output bounds
        combined_output = (combined_output + 1) / 2 * (self.out_max - self.out_min) + self.out_min
        
        return combined_output

# Legacy model class for backwards compatibility
class RobotGraspModel(MultiModalRobotGraspModel):
    """Backwards compatible wrapper"""
    def __init__(self, image_shape: tuple, embed_dim: int, window_size: int, qpos_dim: int, output_type: str, action_chunk: int):
        # If image_shape has 4 channels, use the new multi-modal model
        if image_shape[0] == 4:
            super().__init__(image_shape, embed_dim, window_size, qpos_dim, output_type, action_chunk)
        else:
            # Fallback to original single-modal implementation
            raise NotImplementedError("Original single-modal model not implemented in this version. Use MultiModalRobotGraspModel directly.")