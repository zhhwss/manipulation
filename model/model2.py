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