# models/dinov2_backbone.py
import torch
import torch.nn as nn
from transformers import Dinov2Model  # Ensure your transformers version supports DINOv2
from ..utils import add_lora_to_module

class DINOv2Backbone(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", lora_r=4, lora_alpha=1.0, target_dim=None):
        super().__init__()
        # Load pretrained DINOv2 backbone
        self.dino = Dinov2Model.from_pretrained(model_name)
        
        # Store model variant information for dimension handling
        self.model_variant = model_name.split('/')[-1]  # Extract model variant (tiny, small, base, large)
        
        # Determine hidden dimension based on model variant
        if 'small' in self.model_variant:
            self.hidden_dim = 384  # dinov2-small hidden dimension
        elif 'base' in self.model_variant:
            self.hidden_dim = 768  # dinov2-base hidden dimension
        elif 'large' in self.model_variant:
            self.hidden_dim = 1024  # dinov2-large hidden dimension
        elif 'giant' in self.model_variant:
            self.hidden_dim = 1536  # dinov2-giant hidden dimension
        else:
            # Default to base model dimension
            self.hidden_dim = 768
        
        print(f"DINOv2 backbone variant: {self.model_variant}, hidden dimension: {self.hidden_dim}")
        
        # Create projection layer if target dimension is specified and differs from backbone's hidden dim
        self.target_dim = target_dim
        if target_dim is not None and target_dim != self.hidden_dim:
            print(f"Creating projection layer from dimension {self.hidden_dim} to {target_dim}")
            self.projection = nn.Linear(self.hidden_dim, target_dim)
        else:
            self.projection = None
        
        # Freeze backbone weights (they remain frozen except for LoRA parameters)
        for param in self.dino.parameters():
            param.requires_grad = False
            
        # Apply LoRA adapters selectively only to the last 2 encoder blocks
        # This significantly reduces the number of trainable parameters
        if hasattr(self.dino, 'encoder') and hasattr(self.dino.encoder, 'layer'):
            # Apply LoRA only to the last two transformer blocks
            num_layers = len(self.dino.encoder.layer)
            last_layers = min(2, num_layers)  # Ensure we don't exceed the number of layers
            for i in range(num_layers - last_layers, num_layers):
                print(f"Applying LoRA to encoder layer {i}")
                add_lora_to_module(self.dino.encoder.layer[i], r=lora_r, alpha=lora_alpha)
        else:
            # Fallback to applying LoRA to output projection if encoder layers not accessible
            print("Encoder layers not directly accessible, applying LoRA to output projection only")
            if hasattr(self.dino, 'pooler'):
                add_lora_to_module(self.dino.pooler, r=lora_r, alpha=lora_alpha)

    def forward(self, pixel_values):
        # Forward pass through DINOv2 model; we use the last hidden state as features
        outputs = self.dino(pixel_values)
        features = outputs.last_hidden_state  # shape: [batch, seq_len, hidden_dim]
        
        # Project features if needed
        if self.projection is not None:
            features = self.projection(features)
            
        return features