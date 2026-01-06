import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

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
    def __init__(self, image_shape: tuple, embed_dim: int, window_size: int, qpos_dim: int, output_type: str, action_chunk: int, unet_layers: int = 2):
        super(RobotGraspModel, self).__init__()
        
        self.in_channels = image_shape[0]
        self.image_size = image_shape[1:]
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.qpos_dim = qpos_dim
        self.output_type = output_type
        self.action_chunk = action_chunk
        self.unet_layers = unet_layers
        
        # Calculate action output dimension
        self.action_output_dim = 7 if output_type == 'position' else 8
        self.output_dim = self.action_output_dim * self.action_chunk
        
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
        
        # Pre-UNet processing
        self.pre_unet_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(inplace=True),
            ResBlock(self.embed_dim, self.embed_dim),
            ResBlock(self.embed_dim, self.embed_dim),
        )
        
        # Project to action chunks: (batch_size, embed_dim) -> (batch_size, action_output_dim * action_chunk)
        self.to_action_chunks = nn.Linear(self.embed_dim, self.action_output_dim * self.action_chunk)
        
        # Build U-Net encoder layers - real downsampling
        self.unet_encoder = nn.ModuleList()
        current_channels = self.action_output_dim
        
        # Store the sizes at each level for exact reconstruction
        self.encoder_sizes = []
        
        for i in range(self.unet_layers):
            next_channels = current_channels * 2
            
            # Real downsampling with stride=2
            layer = nn.Sequential(
                nn.Conv1d(current_channels, next_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(next_channels, next_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            self.unet_encoder.append(layer)
            current_channels = next_channels
        
        # Build U-Net decoder layers - real upsampling
        self.unet_decoder = nn.ModuleList()
        
        # Track encoder output channels for skip connections
        encoder_channels = []
        temp_channels = self.action_output_dim
        for i in range(self.unet_layers):
            temp_channels *= 2
            encoder_channels.append(temp_channels)
        
        for i in range(self.unet_layers):
            prev_channels = current_channels
            next_channels = current_channels // 2
            
            # Ensure the final decoder layer outputs the correct number of channels
            if i == self.unet_layers - 1:
                next_channels = self.action_output_dim
            
            # Get skip connection channels from corresponding encoder layer
            skip_idx = self.unet_layers - 1 - i
            skip_channels = encoder_channels[skip_idx]
            input_channels = prev_channels + skip_channels
            
            # Real upsampling
            layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                nn.Conv1d(input_channels, next_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(next_channels, next_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
            self.unet_decoder.append(layer)
            current_channels = next_channels
        
        # Final output projection
        self.final_proj = nn.Sequential(
            nn.Conv1d(self.action_output_dim, self.action_output_dim, kernel_size=1, bias=True),
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
        
        # CNN feature extraction
        h = self.cnn(frames)
        h = h.view(batch_size, self.h_dim)
        h1 = self.fc1(h)
        h2 = self.fc2(qpos)
        
        # Concatenate features
        last_input = torch.cat([h1, h2], dim=-1)
        
        # Pre-UNet processing
        features = self.pre_unet_proj(last_input)  # (batch_size, embed_dim)
        
        # Project to action chunks
        action_features = self.to_action_chunks(features)  # (batch_size, action_output_dim * action_chunk)
        
        # Reshape for Conv1D: (batch_size, action_output_dim, action_chunk)
        # This ensures pooling only operates on the action_chunk dimension
        x = action_features.view(batch_size, self.action_output_dim, self.action_chunk)
        original_size = x.shape[2]  # Store original action_chunk size
        
        # Store skip connections for U-Net
        skip_connections = []
        
        # U-Net Encoder (real downsampling in action_chunk dimension)
        for encoder_layer in self.unet_encoder:
            # Store feature map before pooling for skip connection
            conv_out = encoder_layer[:-1](x)  # All layers except MaxPool1d
            skip_connections.append(conv_out)
            x = encoder_layer[-1](conv_out)  # Apply MaxPool1d
        
        # U-Net Decoder (real upsampling in action_chunk dimension) with skip connections
        for i, decoder_layer in enumerate(self.unet_decoder):
            # Upsample
            x = decoder_layer[0](x)  # Upsample layer
            
            # Get corresponding skip connection
            skip_idx = len(skip_connections) - 1 - i
            if skip_idx >= 0:
                skip_connection = skip_connections[skip_idx]
                
                # Ensure sizes match after upsampling (handle odd sizes)
                if x.shape[2] != skip_connection.shape[2]:
                    x = nn.functional.interpolate(x, size=skip_connection.shape[2], mode='linear', align_corners=False)
                
                # Concatenate skip connection
                x = torch.cat([x, skip_connection], dim=1)
            
            # Apply remaining conv layers
            for conv_layer in decoder_layer[1:]:
                x = conv_layer(x)
        
        # Ensure final output matches original size exactly
        if x.shape[2] != original_size:
            x = nn.functional.interpolate(x, size=original_size, mode='linear', align_corners=False)
        
        # Final projection
        output = self.final_proj(x)  # (batch_size, action_output_dim, action_chunk)
        
        # Reshape back to (batch_size, output_dim)
        output = output.view(batch_size, -1)
        
        # Apply output scaling
        output = (output + 1) / 2 * (self.out_max - self.out_min) + self.out_min
        return output 


# ============= MDN Loss Functions and Utilities =============

def mdn_negative_log_likelihood(mdn_params, target):
    """
    计算 MDN 的负对数似然损失
    
    Args:
        mdn_params: dict with 'log_pi', 'mu', 'sigma'
            log_pi: (batch_size, num_mixtures) - log mixture weights
            mu: (batch_size, num_mixtures, output_dim) - means
            sigma: (batch_size, num_mixtures, output_dim) - standard deviations
        target: (batch_size, output_dim) - target values
    
    Returns:
        nll_loss: scalar - negative log likelihood loss
    """
    log_pi = mdn_params['log_pi']  # (batch_size, num_mixtures)
    mu = mdn_params['mu']          # (batch_size, num_mixtures, output_dim)
    sigma = mdn_params['sigma']    # (batch_size, num_mixtures, output_dim)
    
    batch_size, num_mixtures, output_dim = mu.shape
    
    # Expand target for broadcasting: (batch_size, 1, output_dim)
    target_expanded = target.unsqueeze(1)  # (batch_size, 1, output_dim)
    
    # Compute log probability for each Gaussian component
    # log p(x|μ_k, σ_k) = -0.5 * log(2π) - log(σ_k) - 0.5 * ((x - μ_k) / σ_k)²
    log_2pi = math.log(2 * math.pi)
    
    # (batch_size, num_mixtures, output_dim)
    log_gaussian = -0.5 * log_2pi - torch.log(sigma) - 0.5 * ((target_expanded - mu) / sigma) ** 2
    
    # Sum over output dimensions: (batch_size, num_mixtures)
    log_gaussian_sum = log_gaussian.sum(dim=2)
    
    # Add mixture weights: log(π_k) + log p(x|μ_k, σ_k)
    log_weighted_gaussian = log_pi + log_gaussian_sum  # (batch_size, num_mixtures)
    
    # Log-sum-exp to get log p(x) = log(Σ π_k * p(x|μ_k, σ_k))
    log_likelihood = torch.logsumexp(log_weighted_gaussian, dim=1)  # (batch_size,)
    
    # Return negative log likelihood
    nll_loss = -log_likelihood.mean()
    return nll_loss


def plot_mdn_predictions(mdn_params, target=None, title="MDN Predictions", max_components=5):
    """
    可视化 MDN 的预测分布 (仅适用于 2D 输出)
    
    Args:
        mdn_params: dict with 'log_pi', 'mu', 'sigma'
        target: optional target values for comparison
        title: plot title
        max_components: maximum number of components to visualize
    """
    log_pi = mdn_params['log_pi'].detach().cpu()  # (batch_size, num_mixtures)
    mu = mdn_params['mu'].detach().cpu()          # (batch_size, num_mixtures, output_dim)
    sigma = mdn_params['sigma'].detach().cpu()    # (batch_size, num_mixtures, output_dim)
    
    if mu.shape[2] != 2:
        print(f"Visualization only supports 2D output, got {mu.shape[2]}D")
        return
    
    batch_size, num_mixtures = log_pi.shape
    pi = torch.exp(log_pi)  # Convert to probabilities
    
    plt.figure(figsize=(12, 8))
    
    # Plot for first few samples in batch
    num_samples_to_plot = min(4, batch_size)
    
    for i in range(num_samples_to_plot):
        plt.subplot(2, 2, i + 1)
        
        # Plot each mixture component
        for k in range(min(num_mixtures, max_components)):
            weight = pi[i, k].item()
            center = mu[i, k].numpy()  # (2,)
            std = sigma[i, k].numpy()  # (2,)
            
            # Draw ellipse for 2-sigma confidence region
            if weight > 0.01:  # Only plot significant components
                circle = plt.Circle(center, 2 * std.mean(), 
                                  alpha=weight * 0.7, 
                                  color=f'C{k}',
                                  label=f'Mix {k}: π={weight:.3f}')
                plt.gca().add_patch(circle)
                
                # Mark center
                plt.plot(center[0], center[1], 'o', color=f'C{k}', markersize=8)
        
        # Plot target if provided
        if target is not None:
            target_point = target[i].detach().cpu().numpy()
            plt.plot(target_point[0], target_point[1], 'rx', markersize=10, 
                    markeredgewidth=3, label='Target')
        
        plt.title(f'Sample {i + 1}')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def sample_from_mdn(mdn_params, num_samples=1):
    """
    从 MDN 中采样
    
    Args:
        mdn_params: dict with 'log_pi', 'mu', 'sigma'
        num_samples: number of samples to generate per batch item
    
    Returns:
        samples: (batch_size, num_samples, output_dim)
    """
    log_pi = mdn_params['log_pi']  # (batch_size, num_mixtures)
    mu = mdn_params['mu']          # (batch_size, num_mixtures, output_dim)
    sigma = mdn_params['sigma']    # (batch_size, num_mixtures, output_dim)
    
    batch_size, num_mixtures, output_dim = mu.shape
    device = mu.device
    
    pi = torch.exp(log_pi)  # Convert to probabilities
    
    samples = []
    for _ in range(num_samples):
        # Sample mixture component for each batch item
        mixture_idx = torch.multinomial(pi, 1).squeeze(-1)  # (batch_size,)
        
        # Select corresponding mu and sigma
        batch_indices = torch.arange(batch_size, device=device)
        selected_mu = mu[batch_indices, mixture_idx]      # (batch_size, output_dim)
        selected_sigma = sigma[batch_indices, mixture_idx] # (batch_size, output_dim)
        
        # Sample from Gaussian
        noise = torch.randn_like(selected_mu)
        sample = selected_mu + selected_sigma * noise
        samples.append(sample)
    
    return torch.stack(samples, dim=1)  # (batch_size, num_samples, output_dim)


def visualize_trajectory_predictions(mean, std, title="Trajectory Embeddings"):
    """
    可视化轨迹嵌入的均值和标准差（以圆的形式）
    
    Args:
        mean: (num_trajectories, 2) - 圆心坐标
        std: (num_trajectories, 2) - 圆的半径（标准差）
        title: 图标题
    """
    plt.figure(figsize=(10, 8))
    
    num_trajectories = mean.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_trajectories, 10)))
    
    for i in range(num_trajectories):
        center = mean[i]  # (2,)
        radius = std[i]   # (2,)
        
        # 使用平均半径画圆
        avg_radius = radius.mean() if hasattr(radius, 'mean') else np.mean(radius)
        
        # 画圆
        circle = plt.Circle(center, avg_radius, 
                          alpha=0.3, 
                          color=colors[i % len(colors)],
                          label=f'Traj {i}')
        plt.gca().add_patch(circle)
        
        # 标记圆心
        plt.scatter(center[0], center[1], 
                   color=colors[i % len(colors)], 
                   s=30, alpha=0.7, zorder=5)
        
        # 添加轨迹编号
        plt.text(center[0], center[1], str(i), 
                ha='center', va='center', fontsize=8, 
                color='white', weight='bold')
    
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



class RobotGraspModel_v2(nn.Module):
    def __init__(self, image_shape: tuple, embed_dim: int, window_size: int, qpos_dim: int, output_type: str, action_chunk: int, unet_layers: int = 1, num_mixtures: int = 5):
        super(RobotGraspModel_v2, self).__init__()
        
        self.in_channels = image_shape[0]
        self.image_size = image_shape[1:]
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.qpos_dim = qpos_dim
        self.output_type = output_type
        self.action_chunk = action_chunk
        self.unet_layers = unet_layers
        self.num_mixtures = num_mixtures
        
        # Calculate action output dimension
        self.action_output_dim = 7 if output_type == 'position' else 8
        self.output_dim = self.action_output_dim * self.action_chunk
        
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
        
        # Pre-UNet processing
        self.pre_unet_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(inplace=True),
            ResBlock(self.embed_dim, self.embed_dim),
            ResBlock(self.embed_dim, self.embed_dim),
        )
        
        # Project to action chunks: (batch_size, embed_dim) -> (batch_size, action_output_dim * action_chunk)
        self.to_action_chunks = nn.Linear(self.embed_dim, self.action_output_dim * self.action_chunk)
        
        # Build U-Net encoder layers - real downsampling
        self.unet_encoder = nn.ModuleList()
        current_channels = self.action_output_dim
        
        # Store the sizes at each level for exact reconstruction
        self.encoder_sizes = []
        
        for i in range(self.unet_layers):
            next_channels = current_channels * 2
            
            # Real downsampling with stride=2
            layer = nn.Sequential(
                nn.Conv1d(current_channels, next_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                # nn.Conv1d(next_channels, next_channels, kernel_size=3, padding=1, bias=True),
                # nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            self.unet_encoder.append(layer)
            current_channels = next_channels
        
        # Build U-Net decoder layers - real upsampling
        self.unet_decoder = nn.ModuleList()
        
        # Track encoder output channels for skip connections
        encoder_channels = []
        temp_channels = self.action_output_dim
        for i in range(self.unet_layers):
            temp_channels *= 2
            encoder_channels.append(temp_channels)
        
        for i in range(self.unet_layers):
            prev_channels = current_channels
            next_channels = current_channels // 2
            
            # Ensure the final decoder layer outputs the correct number of channels
            if i == self.unet_layers - 1:
                next_channels = self.action_output_dim
            
            # Get skip connection channels from corresponding encoder layer
            skip_idx = self.unet_layers - 1 - i
            skip_channels = encoder_channels[skip_idx]
            input_channels = prev_channels + skip_channels
            
            # Real upsampling
            layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                nn.Conv1d(input_channels, next_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                # nn.Conv1d(next_channels, next_channels, kernel_size=3, padding=1, bias=True),
                # nn.ReLU(inplace=True)
            )
            self.unet_decoder.append(layer)
            current_channels = next_channels
        
        # MDN output heads - separate projections for each mixture component parameter
        # π_k: mixture weights (K components)
        self.pi_head = nn.Sequential(
            nn.Conv1d(self.action_output_dim, self.num_mixtures, kernel_size=1, bias=True),
            nn.LogSoftmax(dim=1)  # Log probabilities for numerical stability
        )
        
        # μ_k: means (K components × output_dim)
        self.mu_head = nn.Conv1d(self.action_output_dim, self.num_mixtures * self.output_dim, kernel_size=1, bias=True)
        
        # σ_k: standard deviations (K components × output_dim) 
        self.sigma_head = nn.Sequential(
            nn.Conv1d(self.action_output_dim, self.num_mixtures * self.output_dim, kernel_size=1, bias=True),
            nn.Softplus()  # Ensure positive values for std
        )
        
    def forward(self, frames, qpos, mode='max_prob'):
        """
        Args:
            frames: (batch_size, window_size, channels, height, width)
            qpos: (batch_size, window_size, qpos_dim)
            mode: 'sample', 'max_prob', or 'mdn_params' for training
        Returns:
            If mode='mdn_params': dict with 'pi', 'mu', 'sigma' for training
            If mode='sample': sampled output (batch_size, output_dim)
            If mode='max_prob': output from highest probability component (batch_size, output_dim)
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
        
        # Concatenate features
        last_input = torch.cat([h1, h2], dim=-1)
        
        # Pre-UNet processing
        features = self.pre_unet_proj(last_input)  # (batch_size, embed_dim)
        
        # Project to action chunks
        action_features = self.to_action_chunks(features)  # (batch_size, action_output_dim * action_chunk)
        
        # Reshape for Conv1D: (batch_size, action_output_dim, action_chunk)
        # This ensures pooling only operates on the action_chunk dimension
        x = action_features.view(batch_size, self.action_output_dim, self.action_chunk)
        original_size = x.shape[2]  # Store original action_chunk size
        
        # Store skip connections for U-Net
        skip_connections = []
        
        # U-Net Encoder (real downsampling in action_chunk dimension)
        for encoder_layer in self.unet_encoder:
            # Store feature map before pooling for skip connection
            conv_out = encoder_layer[:-1](x)  # All layers except MaxPool1d
            skip_connections.append(conv_out)
            x = encoder_layer[-1](conv_out)  # Apply MaxPool1d
        
        # U-Net Decoder (real upsampling in action_chunk dimension) with skip connections
        for i, decoder_layer in enumerate(self.unet_decoder):
            # Upsample
            x = decoder_layer[0](x)  # Upsample layer
            
            # Get corresponding skip connection
            skip_idx = len(skip_connections) - 1 - i
            if skip_idx >= 0:
                skip_connection = skip_connections[skip_idx]
                
                # Ensure sizes match after upsampling (handle odd sizes)
                if x.shape[2] != skip_connection.shape[2]:
                    x = nn.functional.interpolate(x, size=skip_connection.shape[2], mode='linear', align_corners=False)
                
                # Concatenate skip connection
                x = torch.cat([x, skip_connection], dim=1)
            
            # Apply remaining conv layers
            for conv_layer in decoder_layer[1:]:
                x = conv_layer(x)
        
        # Ensure final output matches original size exactly
        if x.shape[2] != original_size:
            x = nn.functional.interpolate(x, size=original_size, mode='linear', align_corners=False)
        
        # MDN output heads
        # π_k: (batch_size, num_mixtures, action_chunk) -> (batch_size, num_mixtures)
        log_pi = self.pi_head(x).mean(dim=2)  # Average over action_chunk dimension
        
        # μ_k: (batch_size, num_mixtures * output_dim, action_chunk) -> (batch_size, num_mixtures, output_dim)
        mu = self.mu_head(x).view(batch_size, self.num_mixtures, self.output_dim, self.action_chunk)
        mu = mu.mean(dim=3)  # Average over action_chunk dimension
        
        # σ_k: (batch_size, num_mixtures * output_dim, action_chunk) -> (batch_size, num_mixtures, output_dim)
        sigma = self.sigma_head(x).view(batch_size, self.num_mixtures, self.output_dim, self.action_chunk)
        sigma = sigma.mean(dim=3) + 1e-6  # Add small epsilon for numerical stability
        
        # Apply output scaling to means
        mu_scaled = (torch.tanh(mu) + 1) / 2 * (self.out_max - self.out_min) + self.out_min
        
        if mode == 'mdn_params':
            # Return parameters for training
            return {
                'log_pi': log_pi,  # (batch_size, num_mixtures)
                'mu': mu_scaled,   # (batch_size, num_mixtures, output_dim) 
                'sigma': sigma     # (batch_size, num_mixtures, output_dim)
            }
        
        elif mode == 'sample':
            # Sample from the mixture
            pi = torch.exp(log_pi)  # Convert log probabilities to probabilities
            # Sample mixture component for each batch item
            mixture_idx = torch.multinomial(pi, 1).squeeze(-1)  # (batch_size,)
            
            # Select corresponding mu and sigma
            batch_indices = torch.arange(batch_size, device=frames.device)
            selected_mu = mu_scaled[batch_indices, mixture_idx]     # (batch_size, output_dim)
            selected_sigma = sigma[batch_indices, mixture_idx]      # (batch_size, output_dim)
            
            # Sample from Gaussian
            noise = torch.randn_like(selected_mu)
            output = selected_mu + selected_sigma * noise
            return output
            
        elif mode == 'max_prob':
            # Return output from highest probability component
            pi = torch.exp(log_pi)  # Convert log probabilities to probabilities
            max_mixture_idx = torch.argmax(pi, dim=1)  # (batch_size,)
            
            # Select corresponding mu
            batch_indices = torch.arange(batch_size, device=frames.device)
            output = mu_scaled[batch_indices, max_mixture_idx]  # (batch_size, output_dim)
            return output
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'sample', 'max_prob', or 'mdn_params'.")


class RobotGraspModel_v3(nn.Module):
    def __init__(self, image_shape: tuple, embed_dim: int, window_size: int, qpos_dim: int, output_type: str, action_chunk: int, unet_layers: int = 1):
        super(RobotGraspModel_v3, self).__init__()
        
        self.in_channels = image_shape[0]
        self.image_size = image_shape[1:]
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.qpos_dim = qpos_dim
        self.output_type = output_type
        self.action_chunk = action_chunk
        self.unet_layers = unet_layers
        
        # Calculate action output dimension
        self.action_output_dim = 7 if output_type == 'position' else 8
        self.output_dim = self.action_output_dim * self.action_chunk
        
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
        
        # Pre-UNet processing
        self.pre_unet_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(inplace=True),
            ResBlock(self.embed_dim, self.embed_dim),
            ResBlock(self.embed_dim, self.embed_dim),
        )
        
        # Project to action chunks: (batch_size, embed_dim) -> (batch_size, action_output_dim * action_chunk)
        self.to_action_chunks = nn.Linear(self.embed_dim, self.action_output_dim * self.action_chunk)
        
        # Build U-Net encoder layers - use dilated convolutions to increase receptive field without changing size
        self.unet_encoder = nn.ModuleList()
        current_channels = self.action_output_dim
        
        for i in range(self.unet_layers):
            next_channels = current_channels * 2
            dilation = 2 ** i  # Increasing dilation for larger receptive field
            
            layer = nn.Sequential(
                nn.Conv1d(current_channels, next_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=True),
                nn.ReLU(inplace=True)
            )
            self.unet_encoder.append(layer)
            current_channels = next_channels
        
        # Build U-Net decoder layers - mirror the encoder but reduce channels
        self.unet_decoder = nn.ModuleList()
        
        for i in range(self.unet_layers):
            next_channels = current_channels // 2
            dilation = 2 ** (self.unet_layers - 1 - i)  # Decreasing dilation
            
            # Ensure the final decoder layer outputs the correct number of channels
            if i == self.unet_layers - 1:
                next_channels = self.action_output_dim
                dilation = 1  # Final layer uses normal convolution
            
            layer = nn.Sequential(
                nn.Conv1d(current_channels, next_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=True),
                nn.ReLU(inplace=True)
            )
            self.unet_decoder.append(layer)
            current_channels = next_channels
        
        # Final output projection
        self.final_proj = nn.Sequential(
            nn.Conv1d(self.action_output_dim, self.action_output_dim, kernel_size=1, bias=True),
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
        
        # CNN feature extraction
        h = self.cnn(frames)
        h = h.view(batch_size, self.h_dim)
        h1 = self.fc1(h)
        h2 = self.fc2(qpos)
        
        # Concatenate features
        last_input = torch.cat([h1, h2], dim=-1)
        
        # Pre-UNet processing
        features = self.pre_unet_proj(last_input)  # (batch_size, embed_dim)
        
        # Project to action chunks
        action_features = self.to_action_chunks(features)  # (batch_size, action_output_dim * action_chunk)
        
        # Reshape for Conv1D: (batch_size, action_output_dim, action_chunk)
        # This ensures processing only operates on the action_chunk dimension
        x = action_features.view(batch_size, self.action_output_dim, self.action_chunk)
        
        # Store skip connections for U-Net
        skip_connections = []
        
        # U-Net Encoder (dilated convolutions - no size change)
        for encoder_layer in self.unet_encoder:
            skip_connections.append(x)
            x = encoder_layer(x)
        
        # U-Net Decoder (dilated convolutions - no size change) with skip connections
        for i, decoder_layer in enumerate(self.unet_decoder):
            x = decoder_layer(x)
            # Add skip connection if dimensions match (they should since no pooling)
            skip_idx = len(skip_connections) - 1 - i
            if skip_idx >= 0 and x.shape == skip_connections[skip_idx].shape:
                x = x + skip_connections[skip_idx]
        
        # Final projection
        output = self.final_proj(x)  # (batch_size, action_output_dim, action_chunk)
        
        # Reshape back to (batch_size, output_dim)
        output = output.view(batch_size, -1)
        
        # Apply output scaling
        output = (output + 1) / 2 * (self.out_max - self.out_min) + self.out_min
        return output 