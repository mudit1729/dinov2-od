# models/detector.py
import torch
import torch.nn as nn
from .dinov2_backbone import DINOv2Backbone
from .detr_decoder import DETRDecoder
from .. import config

class DINOv2ObjectDetector(nn.Module):
    def __init__(self, 
                 num_classes=config.num_classes,
                 dino_model_name=config.dino_model_name,
                 lora_r=config.lora_r,
                 lora_alpha=config.lora_alpha,
                 hidden_dim=config.hidden_dim,
                 num_queries=config.num_queries,
                 nheads=config.nheads,
                 num_decoder_layers=config.num_decoder_layers,
                 dim_feedforward=config.dim_feedforward,
                 dropout=config.dropout,
                 n_points=config.n_points,
                 use_deformable=config.use_deformable):
        super().__init__()
        
        # Get default hidden dimension based on model variant if not specified
        if hidden_dim is None:
            if 'small' in dino_model_name:
                hidden_dim = 384
            elif 'base' in dino_model_name:
                hidden_dim = 768
            elif 'large' in dino_model_name:
                hidden_dim = 1024
            elif 'giant' in dino_model_name:
                hidden_dim = 1536
            else:
                hidden_dim = 768  # Default to base model dimension
        
        # Initialize the frozen DINOv2 backbone with LoRA
        self.backbone = DINOv2Backbone(
            model_name=dino_model_name, 
            lora_r=lora_r, 
            lora_alpha=lora_alpha,
            target_dim=hidden_dim  # Project to the desired hidden dimension if needed
        )
        
        # Initialize the decoder with optional deformable attention
        self.decoder = DETRDecoder(
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            nheads=nheads,
            num_decoder_layers=num_decoder_layers,
            num_classes=num_classes,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            n_points=n_points,
            use_deformable=use_deformable
        )

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: Input images tensor of shape [batch, 3, H, W]
        Returns:
            A dict with detection outputs.
        """
        # Extract features using the backbone
        features = self.backbone(pixel_values)  # shape: [batch, seq_len, hidden_dim]
        # Pass the features through the decoder to obtain detection predictions
        outputs = self.decoder(features)
        return outputs