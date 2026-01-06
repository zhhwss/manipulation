import torch
import torch.nn as nn

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



class RobotGraspModel_v2(nn.Module):
    def __init__(self, image_shape: tuple, embed_dim: int, window_size: int, qpos_dim: int, output_type: str, action_chunk: int, unet_layers: int = 1):
        super(RobotGraspModel_v2, self).__init__()
        
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


class RobotGraspModelLowRank(nn.Module):
    def __init__(self, image_shape: tuple, embed_dim: int, window_size: int, qpos_dim: int, output_type: str, action_chunk: int, rank: int = 50):
        super(RobotGraspModelLowRank, self).__init__()
        
        self.in_channels = image_shape[0]
        self.image_size = image_shape[1:]
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.qpos_dim = qpos_dim
        self.output_type = output_type
        self.action_chunk = action_chunk
        self.rank = rank  # Low-rank dimension
        
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
        
        # Feature processing
        self.feature_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(inplace=True),
            ResBlock(self.embed_dim, self.embed_dim),
            ResBlock(self.embed_dim, self.embed_dim),
        )
        
        # Low-rank decomposition components
        # Output coefficients (batch_size,) -> (batch_size, rank)
        self.coeff_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim // 2, self.rank),
            nn.Tanh()  # Normalize coefficients
        )
        
        # Basis vectors matrix: (rank, output_dim)
        # This is shared across all trajectories but learnable
        self.basis_vectors = nn.Parameter(
            torch.randn(self.rank, self.output_dim) * 0.1
        )
        
        # Smoothing Conv1D layers
        # Reshape output for Conv1D: (batch_size, action_output_dim, action_chunk)
        self.smoothing_conv = nn.Sequential(
            nn.Conv1d(self.action_output_dim, self.action_output_dim, kernel_size=3, padding=1, bias=True),
            # nn.ReLU(inplace=True),
            nn.Conv1d(self.action_output_dim, self.action_output_dim, kernel_size=3, padding=1, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(self.action_output_dim, self.action_output_dim, kernel_size=5, padding=2, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(self.action_output_dim, self.action_output_dim, kernel_size=1, padding=0, bias=True),
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
        
        # Feature processing
        features = self.feature_proj(last_input)  # (batch_size, embed_dim)
        
        # Generate coefficients
        coefficients = self.coeff_proj(features)  # (batch_size, rank)
        
        # Low-rank reconstruction: coefficients @ basis_vectors
        # (batch_size, rank) @ (rank, output_dim) -> (batch_size, output_dim)
        output = torch.matmul(coefficients, self.basis_vectors)
        
        # Reshape for Conv1D smoothing: (batch_size, action_output_dim, action_chunk)
        output_reshaped = output.view(batch_size, self.action_output_dim, self.action_chunk)
        
        # Apply smoothing Conv1D layers
        smoothed_output = self.smoothing_conv(output_reshaped)
        
        # Reshape back to (batch_size, output_dim)
        output = smoothed_output.view(batch_size, -1)
        
        # Apply output scaling
        output = (output + 1) / 2 * (self.out_max - self.out_min) + self.out_min
        return output 


class RobotGraspModelLowRankSVD(nn.Module):
    def __init__(self, image_shape: tuple, embed_dim: int, window_size: int, qpos_dim: int, output_type: str, action_chunk: int, rank: int = 50, svd_basis_vectors=None):
        super(RobotGraspModelLowRankSVD, self).__init__()
        
        self.in_channels = image_shape[0]
        self.image_size = image_shape[1:]
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.qpos_dim = qpos_dim
        self.output_type = output_type
        self.action_chunk = action_chunk
        self.rank = rank  # Low-rank dimension
        
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
        
        # Feature processing
        self.feature_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(inplace=True),
            ResBlock(self.embed_dim, self.embed_dim),
            ResBlock(self.embed_dim, self.embed_dim),
        )
        
        # Low-rank decomposition components
        # Output coefficients (batch_size,) -> (batch_size, rank)
        self.coeff_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim // 2, self.rank),
            nn.Tanh()  # Normalize coefficients
        )
        
        # SVD-based basis vectors: (rank, output_dim)
        if svd_basis_vectors is not None:
            # Use provided SVD basis vectors (NOT trainable)
            self.register_buffer('basis_vectors', torch.tensor(svd_basis_vectors, dtype=torch.float32))
        else:
            # Initialize randomly if SVD vectors not provided (trainable)
            self.basis_vectors = nn.Parameter(torch.randn(self.rank, self.output_dim) * 0.1)
        
        # Smoothing Conv1D layers
        # Reshape output for Conv1D: (batch_size, action_output_dim, action_chunk)
        self.smoothing_conv = nn.Sequential(
            nn.Conv1d(self.action_output_dim, self.action_output_dim, kernel_size=3, padding=1, bias=True),
            nn.Conv1d(self.action_output_dim, self.action_output_dim, kernel_size=3, padding=1, bias=True),
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
        
        # Feature processing
        features = self.feature_proj(last_input)  # (batch_size, embed_dim)
        
        # Generate coefficients
        coefficients = self.coeff_proj(features)  # (batch_size, rank)
        
        # Low-rank reconstruction: coefficients @ basis_vectors
        # (batch_size, rank) @ (rank, output_dim) -> (batch_size, output_dim)
        output = torch.matmul(coefficients, self.basis_vectors)
        
        # Reshape for Conv1D smoothing: (batch_size, action_output_dim, action_chunk)
        output_reshaped = output.view(batch_size, self.action_output_dim, self.action_chunk)
        
        # Apply smoothing Conv1D layers
        smoothed_output = self.smoothing_conv(output_reshaped)
        
        # Reshape back to (batch_size, output_dim)
        output = smoothed_output.view(batch_size, -1)
        
        # Apply output scaling
        output = (output + 1) / 2 * (self.out_max - self.out_min) + self.out_min
        return output 
    

class RobotGraspModelSmooth(nn.Module):
    def __init__(self, image_shape: tuple, embed_dim: int, window_size: int, qpos_dim: int, output_type: str, action_chunk: int):
        super(RobotGraspModelSmooth, self).__init__()
        
        self.in_channels = image_shape[0]
        self.image_size = image_shape[1:]
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.qpos_dim = qpos_dim
        self.output_type = output_type
        self.action_chunk = action_chunk
        
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
        
        # Feature processing before action generation
        self.feature_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(inplace=True),
            ResBlock(self.embed_dim, self.embed_dim),
            ResBlock(self.embed_dim, self.embed_dim),
        )
        
        # Project to action chunks
        self.to_action_chunks = nn.Linear(self.embed_dim, self.action_output_dim * self.action_chunk)
        
        # Smoothing Conv1D layers for temporal consistency
        # These layers operate on the action_chunk dimension to ensure smooth transitions
        self.smoothing_layers = nn.Sequential(
            # First smoothing layer with larger kernel for global temporal smoothing
            nn.Conv1d(self.action_output_dim, self.action_output_dim, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            
            # Second smoothing layer with smaller kernel for local refinement
            nn.Conv1d(self.action_output_dim, self.action_output_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            
            # Third smoothing layer for final refinement
            nn.Conv1d(self.action_output_dim, self.action_output_dim, kernel_size=3, padding=1, bias=True),
        )
        
        # Residual connection weight for controlling smoothing strength
        self.smooth_weight = nn.Parameter(torch.tensor(0.7))  # Learnable smoothing strength
        
        # Final activation
        self.final_activation = nn.Tanh()
        
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
        
        # Feature processing
        features = self.feature_proj(last_input)  # (batch_size, embed_dim)
        
        # Project to action chunks
        action_features = self.to_action_chunks(features)  # (batch_size, action_output_dim * action_chunk)
        
        # Reshape for Conv1D smoothing: (batch_size, action_output_dim, action_chunk)
        # This ensures smoothing operates along the action_chunk dimension
        x = action_features.view(batch_size, self.action_output_dim, self.action_chunk)
        
        # Apply smoothing layers
        x = self.smoothing_layers(x)
        
        # Apply residual connection with learnable weight
        # This allows the model to learn how much smoothing to apply
        # x = self.smooth_weight * smoothed + (1 - self.smooth_weight) * x
        
        # Apply final activation
        x = self.final_activation(x)
        
        # Reshape back to (batch_size, output_dim)
        output = x.view(batch_size, -1)
        
        # Apply output scaling
        output = (output + 1) / 2 * (self.out_max - self.out_min) + self.out_min
        return output


class RobotGraspModelRNN(nn.Module):
    """使用 RNN/LSTM 进行时序平滑的模型"""
    def __init__(self, image_shape: tuple, embed_dim: int, window_size: int, qpos_dim: int, output_type: str, action_chunk: int):
        super(RobotGraspModelRNN, self).__init__()
        
        self.in_channels = image_shape[0]
        self.image_size = image_shape[1:]
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.qpos_dim = qpos_dim
        self.output_type = output_type
        self.action_chunk = action_chunk
        
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
        
        # Feature processing
        self.feature_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(inplace=True),
            ResBlock(self.embed_dim, self.embed_dim),
            ResBlock(self.embed_dim, self.embed_dim),
        )
        
        # Project to initial action representation
        self.to_initial_actions = nn.Linear(self.embed_dim, self.action_output_dim * self.action_chunk)
        
        # LSTM for temporal smoothing
        # Input: (batch_size, action_chunk, action_output_dim)
        # Output: (batch_size, action_chunk, action_output_dim)
        self.lstm = nn.LSTM(
            input_size=self.action_output_dim,
            hidden_size=self.action_output_dim * 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Final projection after LSTM
        self.final_proj = nn.Sequential(
            nn.Linear(self.action_output_dim * 4, self.action_output_dim * 2),  # 4 = 2 * bidirectional
            nn.ReLU(inplace=True),
            nn.Linear(self.action_output_dim * 2, self.action_output_dim),
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
        
        # Feature processing
        features = self.feature_proj(last_input)  # (batch_size, embed_dim)
        
        # Project to initial action representation
        action_features = self.to_initial_actions(features)  # (batch_size, action_output_dim * action_chunk)
        
        # Reshape for LSTM: (batch_size, action_chunk, action_output_dim)
        x = action_features.view(batch_size, self.action_chunk, self.action_output_dim)
        
        # Apply LSTM for temporal smoothing
        lstm_out, _ = self.lstm(x)  # (batch_size, action_chunk, action_output_dim * 4)
        
        # Final projection
        smoothed = self.final_proj(lstm_out)  # (batch_size, action_chunk, action_output_dim)
        
        # Reshape back to (batch_size, output_dim)
        output = smoothed.view(batch_size, -1)
        
        # Apply output scaling
        output = (output + 1) / 2 * (self.out_max - self.out_min) + self.out_min
        return output


class RobotGraspModelTransformer(nn.Module):
    """使用 Transformer 进行时序平滑的模型"""
    def __init__(self, image_shape: tuple, embed_dim: int, window_size: int, qpos_dim: int, output_type: str, action_chunk: int):
        super(RobotGraspModelTransformer, self).__init__()
        
        self.in_channels = image_shape[0]
        self.image_size = image_shape[1:]
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.qpos_dim = qpos_dim
        self.output_type = output_type
        self.action_chunk = action_chunk
        
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
        
        # Feature processing
        self.feature_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(inplace=True),
            ResBlock(self.embed_dim, self.embed_dim),
            ResBlock(self.embed_dim, self.embed_dim),
        )
        
        # Project to action tokens
        self.to_action_tokens = nn.Linear(self.embed_dim, self.action_output_dim * self.action_chunk)
        
        # Positional encoding for action sequence
        self.pos_encoding = nn.Parameter(torch.randn(1, self.action_chunk, self.action_output_dim) * 0.1)
        
        # Transformer encoder for temporal smoothing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.action_output_dim,
            nhead=4,  # Multi-head attention
            dim_feedforward=self.action_output_dim * 2,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(self.action_output_dim, self.action_output_dim),
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
        
        # Feature processing
        features = self.feature_proj(last_input)  # (batch_size, embed_dim)
        
        # Project to action tokens
        action_features = self.to_action_tokens(features)  # (batch_size, action_output_dim * action_chunk)
        
        # Reshape for Transformer: (batch_size, action_chunk, action_output_dim)
        x = action_features.view(batch_size, self.action_chunk, self.action_output_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply Transformer for temporal smoothing with self-attention
        smoothed = self.transformer(x)  # (batch_size, action_chunk, action_output_dim)
        
        # Final projection
        output_tokens = self.final_proj(smoothed)  # (batch_size, action_chunk, action_output_dim)
        
        # Reshape back to (batch_size, output_dim)
        output = output_tokens.view(batch_size, -1)
        
        # Apply output scaling
        output = (output + 1) / 2 * (self.out_max - self.out_min) + self.out_min
        return output


class RobotGraspModelGaussian(nn.Module):
    """使用高斯滤波进行平滑的模型"""
    def __init__(self, image_shape: tuple, embed_dim: int, window_size: int, qpos_dim: int, output_type: str, action_chunk: int):
        super(RobotGraspModelGaussian, self).__init__()
        
        self.in_channels = image_shape[0]
        self.image_size = image_shape[1:]
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.qpos_dim = qpos_dim
        self.output_type = output_type
        self.action_chunk = action_chunk
        
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
        
        # Feature processing
        self.feature_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(inplace=True),
            ResBlock(self.embed_dim, self.embed_dim),
            ResBlock(self.embed_dim, self.embed_dim),
        )
        
        # Project to action chunks
        self.to_action_chunks = nn.Linear(self.embed_dim, self.action_output_dim * self.action_chunk)
        
        # Learnable Gaussian kernel parameters
        self.sigma = nn.Parameter(torch.tensor(1.0))  # Standard deviation
        self.kernel_size = 7  # Fixed kernel size (should be odd)
        
        # Final activation
        self.final_activation = nn.Tanh()
        
    def create_gaussian_kernel(self, sigma, kernel_size, device):
        """Create 1D Gaussian kernel"""
        x = torch.arange(kernel_size, dtype=torch.float32, device=device)
        x = x - (kernel_size - 1) / 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        return kernel
        
    def gaussian_smooth_1d(self, x, kernel):
        """Apply 1D Gaussian smoothing along the last dimension"""
        batch_size, channels, length = x.shape
        
        # Pad the input
        padding = len(kernel) // 2
        x_padded = torch.nn.functional.pad(x, (padding, padding), mode='reflect')
        
        # Apply convolution
        kernel = kernel.view(1, 1, -1).expand(channels, 1, -1)
        smoothed = torch.nn.functional.conv1d(
            x_padded.view(batch_size * channels, 1, -1),
            kernel,
            groups=channels,
            padding=0
        )
        
        return smoothed.view(batch_size, channels, length)
        
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
        
        # Feature processing
        features = self.feature_proj(last_input)  # (batch_size, embed_dim)
        
        # Project to action chunks
        action_features = self.to_action_chunks(features)  # (batch_size, action_output_dim * action_chunk)
        
        # Reshape for smoothing: (batch_size, action_output_dim, action_chunk)
        x = action_features.view(batch_size, self.action_output_dim, self.action_chunk)
        
        # Create Gaussian kernel
        gaussian_kernel = self.create_gaussian_kernel(
            torch.abs(self.sigma) + 0.1,  # Ensure positive sigma
            self.kernel_size,
            frames.device
        )
        
        # Apply Gaussian smoothing along action_chunk dimension
        smoothed = self.gaussian_smooth_1d(x, gaussian_kernel)
        
        # Apply final activation
        smoothed = self.final_activation(smoothed)
        
        # Reshape back to (batch_size, output_dim)
        output = smoothed.view(batch_size, -1)
        
        # Apply output scaling
        output = (output + 1) / 2 * (self.out_max - self.out_min) + self.out_min
        return output 