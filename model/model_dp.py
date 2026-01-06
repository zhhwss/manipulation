import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional
import copy
from einops import rearrange, reduce

try:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers not available, using custom scheduler")

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

# ============= Simple PointNet Encoder for Point Cloud Images =============

class SimplePointNetEncoder(nn.Module):
    """
    Simple PointNet-like encoder for point cloud images
    Processes image-format point clouds (depth-converted to xyz)
    """
    def __init__(self, input_channels: int, output_dim: int = 256):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        
        # Point-wise MLPs (using 1x1 convolutions)
        self.point_net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        # Final projection
        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_channels, height, width) - point cloud image
        Returns:
            features: (batch_size, output_dim)
        """
        # Point-wise feature extraction
        features = self.point_net(x)  # (batch_size, 256, height, width)
        
        # Global max pooling
        global_features = self.global_pool(features)  # (batch_size, 256, 1, 1)
        global_features = global_features.view(global_features.size(0), -1)  # (batch_size, 256)
        
        # Final projection
        output = self.fc(global_features)  # (batch_size, output_dim)
        
        return output


# ============= ConditionalUnet1D (Simplified) =============

class ConditionalUnet1D(nn.Module):
    """
    Simplified version of ConditionalUnet1D for action sequence denoising
    """
    def __init__(self, 
                 input_dim: int,
                 global_cond_dim: Optional[int] = None,
                 diffusion_step_embed_dim: int = 256,
                 down_dims: tuple = (256, 512, 1024),
                 kernel_size: int = 5,
                 n_groups: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.global_cond_dim = global_cond_dim
        self.down_dims = down_dims
        
        # Diffusion timestep embedding
        self.time_embed_dim = diffusion_step_embed_dim // 4
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, diffusion_step_embed_dim),
            nn.ReLU(),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim)
        )
        
        # Global conditioning projection
        if global_cond_dim is not None:
            self.global_cond_proj = nn.Linear(global_cond_dim, diffusion_step_embed_dim)
        else:
            self.global_cond_proj = None
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, down_dims[0], 1)
        
        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        in_dim = down_dims[0]
        for out_dim in down_dims:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size//2),
                    nn.GroupNorm(n_groups, out_dim),
                    nn.ReLU(),
                    nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size//2),
                    nn.GroupNorm(n_groups, out_dim),
                    nn.ReLU(),
                    nn.MaxPool1d(2)
                )
            )
            in_dim = out_dim
        
        # Middle
        self.middle = nn.Sequential(
            nn.Conv1d(down_dims[-1], down_dims[-1], kernel_size, padding=kernel_size//2),
            nn.GroupNorm(n_groups, down_dims[-1]),
            nn.ReLU(),
            nn.Conv1d(down_dims[-1], down_dims[-1], kernel_size, padding=kernel_size//2),
            nn.GroupNorm(n_groups, down_dims[-1]),
            nn.ReLU(),
        )
        
        # Decoder (upsampling)
        self.decoder = nn.ModuleList()
        up_dims = list(reversed(down_dims))
        for i, out_dim in enumerate(up_dims[1:] + [input_dim]):
            in_dim = up_dims[i]
            self.decoder.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                    nn.Conv1d(in_dim * 2, out_dim, kernel_size, padding=kernel_size//2),  # *2 for skip connection
                    nn.GroupNorm(n_groups, out_dim) if out_dim != input_dim else nn.Identity(),
                    nn.ReLU() if out_dim != input_dim else nn.Identity(),
                    nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size//2),
                    nn.GroupNorm(n_groups, out_dim) if out_dim != input_dim else nn.Identity(),
                    nn.ReLU() if out_dim != input_dim else nn.Identity(),
                )
            )
        
        # Conditioning fusion layers
        self.cond_fusion = nn.ModuleList()
        for dim in down_dims:
            self.cond_fusion.append(
                nn.Sequential(
                    nn.Linear(diffusion_step_embed_dim, dim),
                    nn.ReLU(),
                    nn.Linear(dim, dim)
                )
            )
    
    def _timestep_embedding(self, timesteps, dim):
        """Sinusoidal timestep embeddings"""
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        freqs = freqs.to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, sample, timestep, global_cond=None):
        """
        Args:
            sample: (batch_size, input_dim, sequence_length)
            timestep: (batch_size,) or int
            global_cond: (batch_size, global_cond_dim)
        Returns:
            output: (batch_size, input_dim, sequence_length)
        """
        # Timestep embedding
        if isinstance(timestep, int):
            timestep = torch.full((sample.shape[0],), timestep, device=sample.device)
        time_emb = self._timestep_embedding(timestep, self.time_embed_dim)
        time_emb = self.time_mlp(time_emb)  # (batch_size, diffusion_step_embed_dim)
        
        # Global conditioning
        if global_cond is not None and self.global_cond_proj is not None:
            global_emb = self.global_cond_proj(global_cond)
            cond_emb = time_emb + global_emb
        else:
            cond_emb = time_emb
        
        # Input projection
        x = self.input_proj(sample)
        
        # Encoder with skip connections
        skip_connections = []
        for i, layer in enumerate(self.encoder):
            # Apply conditioning
            cond_proj = self.cond_fusion[i](cond_emb)  # (batch_size, dim)
            cond_proj = cond_proj.unsqueeze(-1)  # (batch_size, dim, 1)
            
            # Store skip connection before pooling
            conv_out = layer[:-1](x)  # All layers except MaxPool1d
            skip_connections.append(conv_out + cond_proj)
            x = layer[-1](conv_out)  # Apply MaxPool1d
        
        # Middle
        x = self.middle(x)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            # Upsample
            x = layer[0](x)
            
            # Skip connection
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]
                # Match sizes
                if x.shape[-1] != skip.shape[-1]:
                    x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            
            # Apply remaining layers
            for sublayer in layer[1:]:
                x = sublayer(x)
        
        return x


# ============= Enhanced Diffusion Policy =============

class DiffusionPolicy(nn.Module):
    def __init__(self, 
                 image_shape: tuple, 
                 embed_dim: int, 
                 window_size: int, 
                 qpos_dim: int, 
                 output_type: str, 
                 action_chunk: int, 
                 unet_layers: int = 2, 
                 num_diffusion_steps: int = 100,
                 encoder_type: str = "cnn",  # "cnn" or "pointnet"
                 use_diffusers_scheduler: bool = False,
                 scheduler_type: str = "ddpm",  # "ddpm" or "ddim"
                 beta_schedule: str = "squaredcos_cap_v2",  # for diffusers scheduler
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 prediction_type: str = "sample",
                 condition_type: str = "film",
                 down_dims: tuple = (256, 512, 1024),
                 kernel_size: int = 5,
                 n_groups: int = 8):
        super(DiffusionPolicy, self).__init__()
        
        self.in_channels = image_shape[0]
        self.image_size = image_shape[1:]
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.qpos_dim = qpos_dim
        self.output_type = output_type
        self.action_chunk = action_chunk
        self.encoder_type = encoder_type
        self.use_diffusers_scheduler = use_diffusers_scheduler
        self.condition_type = condition_type
        
        # Calculate action output dimension
        self.action_output_dim = 7 if output_type == 'position' else 8
        self.output_dim = self.action_output_dim * self.action_chunk
        
        # Initialize scheduler (following DP3 configuration)
        if use_diffusers_scheduler and DIFFUSERS_AVAILABLE:
            if scheduler_type == "ddim":
                self.noise_scheduler = DDIMScheduler(
                    num_train_timesteps=num_diffusion_steps,
                    beta_start=beta_start,
                    beta_end=beta_end,
                    beta_schedule=beta_schedule,
                    clip_sample=True,
                    set_alpha_to_one=True,
                    steps_offset=0,
                    prediction_type=prediction_type
                )
            else:  # ddpm
                self.noise_scheduler = DDPMScheduler(
                    num_train_timesteps=num_diffusion_steps,
                    beta_start=beta_start,
                    beta_end=beta_end,
                    beta_schedule=beta_schedule,
                    prediction_type=prediction_type
                )
            self.num_diffusion_steps = num_diffusion_steps
            self.use_ddim = (scheduler_type == "ddim")
            print(f"Using {scheduler_type.upper()} scheduler with {beta_schedule} schedule")
        else:
            # Use custom scheduler
            self.num_diffusion_steps = num_diffusion_steps
            self.register_buffer('betas', self._cosine_beta_schedule(num_diffusion_steps))
            self.register_buffer('alphas', 1.0 - self.betas)
            self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
            self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
            self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
            self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        
        # Vision encoder selection
        if encoder_type == "pointnet":
            print(f"Using PointNet encoder for point cloud images")
            self.vision_encoder = SimplePointNetEncoder(
                input_channels=self.in_channels * self.window_size,
                output_dim=embed_dim
            )
            self.vision_feature_dim = embed_dim
        else:  # "cnn"
            print(f"Using CNN encoder")
            self.vision_encoder = nn.Sequential(
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
            
            # Calculate CNN output dimension
            with torch.no_grad():
                dummy_input = torch.randn(1, self.in_channels*self.window_size, self.image_size[0], self.image_size[1])
                h = self.vision_encoder(dummy_input)
                self.vision_feature_dim = h.view(1, -1).size(1)
        
        # Vision feature projection
        self.vision_proj = nn.Sequential(
            nn.Linear(self.vision_feature_dim, self.embed_dim, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # QPos encoder
        self.qpos_encoder = nn.Sequential(
            nn.Linear(self.qpos_dim*self.window_size, self.embed_dim, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # Combined observation feature dimension
        obs_feature_dim = self.embed_dim * 2  # vision + qpos
        
        # ConditionalUnet1D for denoising
        # input_dim should be action_output_dim since we reshape to (batch, action_output_dim, action_chunk)
        self.denoising_model = ConditionalUnet1D(
            input_dim=self.action_output_dim,
            global_cond_dim=obs_feature_dim,
            diffusion_step_embed_dim=embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups
        )
    
    def _encode_observations(self, frames, qpos):
        """
        Encode observations (frames + qpos) into conditioning features
        
        Args:
            frames: (batch_size, window_size, channels, height, width)
            qpos: (batch_size, window_size, qpos_dim)
            
        Returns:
            obs_features: (batch_size, obs_feature_dim)
        """
        batch_size = frames.shape[0]
        
        # Flatten frames for vision encoder
        frames_flat = frames.view(batch_size, self.window_size * self.in_channels, *self.image_size)
        
        # Vision encoding
        if self.encoder_type == "pointnet":
            vision_features = self.vision_encoder(frames_flat)  # (batch_size, embed_dim)
        else:  # "cnn"
            vision_features = self.vision_encoder(frames_flat)  # (batch_size, vision_feature_dim)
            vision_features = vision_features.view(batch_size, -1)
        
        vision_features = self.vision_proj(vision_features)  # (batch_size, embed_dim)
        
        # QPos encoding
        qpos_flat = qpos.view(batch_size, self.window_size * self.qpos_dim)
        qpos_features = self.qpos_encoder(qpos_flat)  # (batch_size, embed_dim)
        
        # Combine features
        obs_features = torch.cat([vision_features, qpos_features], dim=-1)  # (batch_size, embed_dim * 2)
        
        return obs_features
        
    def forward(self, frames, qpos, num_denoising_steps=5):
        """
        Forward inference method for diffusion model
        
        Args:
            frames: (batch_size, window_size, channels, height, width)
            qpos: (batch_size, window_size, qpos_dim)
            num_denoising_steps: number of denoising steps, default 20
            
        Returns:
            actions: (batch_size, action_output_dim * action_chunk) predicted action sequence
        """
        # Use fast sampling for inference
        with torch.no_grad():
            samples = self.conditional_sample(frames, qpos, num_denoising_steps=num_denoising_steps)
            return samples 

    def _cosine_beta_schedule(self, timesteps):
        """Cosine noise schedule for diffusion"""
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, actions, timesteps, noise=None):
        """
        Add noise to actions for diffusion training
        
        Args:
            actions: (batch_size, action_output_dim * action_chunk)
            timesteps: (batch_size,) or int
            noise: optional noise tensor
            
        Returns:
            noisy_actions: (batch_size, action_output_dim * action_chunk)
            noise: (batch_size, action_output_dim * action_chunk)
        """
        if self.use_diffusers_scheduler and hasattr(self, 'noise_scheduler'):
            # Use diffusers scheduler
            if noise is None:
                noise = torch.randn_like(actions)
            return self.noise_scheduler.add_noise(actions, noise, timesteps), noise
        else:
            # Use custom scheduler
            if noise is None:
                noise = torch.randn_like(actions)
            
            if isinstance(timesteps, int):
                timesteps = torch.full((actions.shape[0],), timesteps, device=actions.device)
                
            sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1)
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1)
            
            noisy_actions = sqrt_alphas_cumprod_t * actions + sqrt_one_minus_alphas_cumprod_t * noise
            return noisy_actions, noise
    
    def predict_noise(self, noisy_actions, frames, qpos, timesteps, global_cond=None):
        """
        Predict noise using the denoising model
        
        Args:
            noisy_actions: (batch_size, action_output_dim * action_chunk)
            frames: (batch_size, window_size, channels, height, width)
            qpos: (batch_size, window_size, qpos_dim)
            timesteps: (batch_size,) or int
            global_cond: optional pre-computed conditioning features
            
        Returns:
            predicted_noise: (batch_size, action_output_dim * action_chunk)
        """
        batch_size = noisy_actions.shape[0]
        
        # Get conditioning features
        if global_cond is None:
            global_cond = self._encode_observations(frames, qpos)
        
        # Reshape noisy actions for ConditionalUnet1D
        # (batch_size, action_output_dim * action_chunk) -> (batch_size, action_chunk, action_output_dim) -> (batch_size, action_output_dim, action_chunk)
        # First reshape to separate timesteps and dimensions, then transpose to ConditionalUnet1D format
        noisy_actions_seq = noisy_actions.view(batch_size, self.action_chunk, self.action_output_dim)
        noisy_actions_seq = noisy_actions_seq.transpose(1, 2)  # (batch_size, action_output_dim, action_chunk)
        
        # Predict noise using ConditionalUnet1D
        predicted_noise_seq = self.denoising_model(
            sample=noisy_actions_seq,
            timestep=timesteps,
            global_cond=global_cond
        )
        
        # Reshape back to (batch_size, action_output_dim * action_chunk)
        # First transpose back: (batch_size, action_output_dim, action_chunk) -> (batch_size, action_chunk, action_output_dim)
        predicted_noise_seq = predicted_noise_seq.transpose(1, 2)
        # Then flatten: (batch_size, action_chunk, action_output_dim) -> (batch_size, action_output_dim * action_chunk)
        predicted_noise = predicted_noise_seq.contiguous().view(batch_size, -1)
        
        return predicted_noise
    
    def conditional_sample(self, frames, qpos, num_denoising_steps=20, noisy_init=None):
        """
        Sample actions using conditional diffusion process
        
        Args:
            frames: (batch_size, window_size, channels, height, width)
            qpos: (batch_size, window_size, qpos_dim)
            num_denoising_steps: number of denoising steps
            noisy_init: (batch_size, action_output_dim * action_chunk) optional noisy initialization, if None use random noise
            
        Returns:
            sampled_actions: (batch_size, action_output_dim * action_chunk)
        """
        batch_size = frames.shape[0]
        device = frames.device
        
        # Pre-compute conditioning features
        global_cond = self._encode_observations(frames, qpos)
        
        if self.use_diffusers_scheduler and hasattr(self, 'noise_scheduler'):
            # Use diffusers scheduler
            self.noise_scheduler.set_timesteps(num_denoising_steps)
            
            # Start from noise (random or provided)
            if noisy_init is not None:
                actions = noisy_init.clone()
            else:
                actions = torch.randn(batch_size, self.output_dim, device=device)
            
            # Denoising loop
            for t in self.noise_scheduler.timesteps:
                timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
                
                # Predict noise
                predicted_noise = self.predict_noise(actions, frames, qpos, timestep, global_cond)
                
                # Scheduler step
                actions = self.noise_scheduler.step(predicted_noise, t, actions).prev_sample
            
            return actions
        else:
            # Use custom fast sampling (DDIM-like)
            if noisy_init is not None:
                actions = noisy_init.clone()
            else:
                actions = torch.randn(batch_size, self.output_dim, device=device)
            
            # Calculate timesteps for fast sampling
            step_size = max(1, self.num_diffusion_steps // num_denoising_steps)
            timesteps = list(range(0, self.num_diffusion_steps, step_size))[:num_denoising_steps]
            
            # Fast denoising loop
            for i, t in enumerate(reversed(timesteps)):
                timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
                
                # Predict noise
                predicted_noise = self.predict_noise(actions, frames, qpos, timestep, global_cond)
                
                # DDIM update rule (deterministic, faster)
                if i < len(timesteps) - 1:  # Not the last step
                    alpha_cumprod_t = self.alphas_cumprod[t]
                    sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
                    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
                    
                    # Predict original sample
                    pred_original = (actions - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
                    
                    # Next timestep
                    next_t = timesteps[len(timesteps) - 2 - i] if i < len(timesteps) - 1 else 0
                    alpha_cumprod_prev = self.alphas_cumprod[next_t]
                    
                    # DDIM update (deterministic)
                    pred_dir = torch.sqrt(1 - alpha_cumprod_prev) * predicted_noise
                    actions = torch.sqrt(alpha_cumprod_prev) * pred_original + pred_dir
                else:
                    # Last step
                    alpha_cumprod_t = self.alphas_cumprod[t]
                    sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
                    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
                    actions = (actions - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
            
            return actions
    
    def compute_loss(self, batch):
        """
        Compute diffusion training loss (similar to DP3)
        
        Args:
            batch: dict with 'frames', 'qpos' and 'actions' keys
            
        Returns:
            loss: scalar loss
            loss_dict: dict with loss components
        """
        # Extract batch data
        frames = batch['frames']  # (batch_size, window_size, channels, height, width)
        qpos = batch['qpos']      # (batch_size, window_size, qpos_dim)
        actions = batch['actions'] # (batch_size, action_output_dim * action_chunk)
        
        batch_size = actions.shape[0]
        device = actions.device
        
        # Sample random timesteps
        if self.use_diffusers_scheduler and hasattr(self, 'noise_scheduler'):
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=device)
        else:
            timesteps = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(actions)
        
        # Add noise to actions
        noisy_actions, _ = self.add_noise(actions, timesteps, noise)
        
        # Predict noise
        predicted_noise = self.predict_noise(noisy_actions, frames, qpos, timesteps)
        
        # Compute loss
        if self.use_diffusers_scheduler and hasattr(self, 'noise_scheduler'):
            pred_type = self.noise_scheduler.config.prediction_type
            if pred_type == 'epsilon':
                target = noise
            elif pred_type == 'sample':
                target = actions
            elif pred_type == 'v_prediction':
                # V-parameterization
                alpha_t = self.noise_scheduler.alphas_cumprod[timesteps]
                sigma_t = (1 - alpha_t) ** 0.5
                alpha_t = alpha_t ** 0.5
                target = alpha_t[:, None] * noise - sigma_t[:, None] * actions
            else:
                raise ValueError(f"Unsupported prediction type {pred_type}")
        else:
            # Default: predict epsilon (noise)
            target = noise
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, target)
        
        loss_dict = {
            'diffusion_loss': loss.item(),
        }
        
        return loss, loss_dict
    

# ============= Diffusion Policy Utilities =============

def diffusion_loss(predicted_noise, true_noise):
    """
    Simple MSE loss for predicted vs true noise
    """
    return F.mse_loss(predicted_noise, true_noise) 