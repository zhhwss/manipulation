import torch
import torch.nn as nn
import torch.nn.functional as F

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

class RobotGraspModelBN(nn.Module):
    def __init__(self, image_shape: tuple, embed_dim: int, window_size: int, qpos_dim: int, output_type: str, action_chunk: int, unet_layers: int = 2, num_heads: int = 3):
        super(RobotGraspModel, self).__init__()
        
        self.in_channels = image_shape[0]
        self.image_size = image_shape[1:]
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.qpos_dim = qpos_dim
        self.output_type = output_type
        self.action_chunk = action_chunk
        self.num_heads = num_heads
        
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
        
        # Head selection tracking
        self.register_buffer('head_selection_counts', torch.zeros(num_heads, dtype=torch.long))
        self.register_buffer('total_selections', torch.tensor(0, dtype=torch.long))
        self.min_selection_threshold = 1.0 / 6.0  # 1/6 threshold
        
        # Shared CNN backbone
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
        
        # Shared feature extraction layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.h_dim, self.embed_dim, bias=True),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.qpos_dim*self.window_size, self.embed_dim, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # Multiple output projections (one for each head) - each head gets its own output_proj
        self.output_projections = nn.ModuleList()
        for _ in range(num_heads):
            output_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(inplace=True),
            ResBlock(self.embed_dim, self.embed_dim),
            ResBlock(self.embed_dim, self.embed_dim),
                nn.Linear(self.embed_dim, self.output_dim),
                nn.Tanh()
            )
            self.output_projections.append(output_proj)
        
    def get_valid_heads(self):
        """Get heads that have selection rate >= 1/6"""
        if self.total_selections == 0:
            return list(range(self.num_heads))  # All heads valid initially
        
        selection_rates = self.head_selection_counts.float() / self.total_selections.float()
        valid_heads = (selection_rates >= self.min_selection_threshold).nonzero(as_tuple=True)[0].tolist()
        
        # Ensure at least one head is valid
        if len(valid_heads) == 0:
            valid_heads = [0]  # Fallback to first head
            
        return valid_heads
    
    def update_head_selection(self, selected_head):
        """Update head selection statistics"""
        if 0 <= selected_head < self.num_heads:
            self.head_selection_counts[selected_head] += 1
            self.total_selections += 1
    
    def forward_shared(self, frames, qpos):
        """Forward pass through shared network components"""
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
        
        return last_input
    
    def forward(self, frames, qpos, head_idx=None, training=False):
        """
        Args:
            frames: (batch_size, window_size, channels, height, width)
            qpos: (batch_size, window_size, qpos_dim)
            head_idx: specific head index to use (if None, use all heads or random valid head)
            training: whether in training mode
        Returns:
            If training: list of outputs from all heads
            If inference: single output tensor from selected head
        """
        # Forward through shared network
        shared_features = self.forward_shared(frames, qpos)  # (batch_size, embed_dim * 2)
        
        if training:
            # Training mode: compute all head outputs
            outputs = []
            for i, output_proj in enumerate(self.output_projections):
                output = output_proj(shared_features)  # (batch_size, output_dim)
                # Apply output scaling
                output = (output + 1) / 2 * (self.out_max - self.out_min) + self.out_min
                outputs.append(output)
            return outputs
        else:
            # Inference mode: use specific head or random valid head
            if head_idx is None:
                valid_heads = self.get_valid_heads()
                head_idx = torch.randint(0, len(valid_heads), (1,)).item()
                head_idx = valid_heads[head_idx]
            
            # Ensure head_idx is valid
            head_idx = max(0, min(head_idx, self.num_heads - 1))
            
            output = self.output_projections[head_idx](shared_features)  # (batch_size, output_dim)
            # Apply output scaling
            output = (output + 1) / 2 * (self.out_max - self.out_min) + self.out_min
            return output 

    def get_head_statistics(self):
        """Get head selection statistics"""
        if self.total_selections == 0:
            return {
                'selection_counts': self.head_selection_counts.tolist(),
                'selection_rates': [0.0] * self.num_heads,
                'total_selections': 0,
                'valid_heads': list(range(self.num_heads))
            }
        
        selection_rates = (self.head_selection_counts.float() / self.total_selections.float()).tolist()
        valid_heads = self.get_valid_heads()
        
        return {
            'selection_counts': self.head_selection_counts.tolist(),
            'selection_rates': selection_rates,
            'total_selections': self.total_selections.item(),
            'valid_heads': valid_heads
        } 
