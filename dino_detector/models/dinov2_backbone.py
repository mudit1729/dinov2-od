# models/dinov2_backbone.py
import torch
import torch.nn as nn
from transformers import Dinov2Model  # Ensure your transformers version supports DINOv2
from ..utils import add_lora_to_module

class DINOv2Backbone(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", lora_r=4, lora_alpha=1.0):
        super().__init__()
        # Load pretrained DINOv2 backbone
        self.dino = Dinov2Model.from_pretrained(model_name)
        # Freeze backbone weights (they remain frozen except for LoRA parameters)
        for param in self.dino.parameters():
            param.requires_grad = False
        # Apply LoRA adapters to all linear layers in the backbone
        add_lora_to_module(self.dino, r=lora_r, alpha=lora_alpha)

    def forward(self, pixel_values):
        # Forward pass through DINOv2 model; we use the last hidden state as features
        outputs = self.dino(pixel_values)
        # outputs.last_hidden_state shape: [batch, seq_len, hidden_dim]
        return outputs.last_hidden_state