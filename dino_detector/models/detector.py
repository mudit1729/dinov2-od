# models/detector.py
import torch
import torch.nn as nn
from .dinov2_backbone import DINOv2Backbone
from .detr_decoder import DETRDecoder
from .. import config

class DINOv2ObjectDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize the frozen DINOv2 backbone with LoRA
        self.backbone = DINOv2Backbone(
            model_name=config.dino_model_name, 
            lora_r=config.lora_r, 
            lora_alpha=config.lora_alpha
        )
        # Initialize the decoder with optional deformable attention
        self.decoder = DETRDecoder(
            num_queries=config.num_queries,
            hidden_dim=config.hidden_dim,
            nheads=config.nheads,
            num_decoder_layers=config.num_decoder_layers,
            num_classes=config.num_classes,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            n_points=config.n_points,
            use_deformable=config.use_deformable
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