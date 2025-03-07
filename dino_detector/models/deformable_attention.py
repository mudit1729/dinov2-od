# models/deformable_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


class DeformableAttention(nn.Module):
    """
    Deformable Attention Module for object detection.
    
    This module implements a simplified version of Deformable Attention as described in
    "Deformable DETR: Deformable Transformers for End-to-End Object Detection".
    """
    def __init__(self, d_model=256, n_heads=8, n_points=4):
        """
        Initialize the deformable attention module.
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            n_points: Number of sampling points per attention head per query
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        
        # Projection matrices
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Initialize sampling offsets to 0
        constant_(self.sampling_offsets.weight.data, 0.)
        constant_(self.sampling_offsets.bias.data, 0.)
        
        # Initialize attention weights
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        
        # Initialize projections with Xavier
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
        
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes):
        """
        Forward pass for deformable attention.
        
        Args:
            query: [batch_size, num_queries, d_model]
            reference_points: [batch_size, num_queries, 2] normalized reference points (x, y)
            input_flatten: [batch_size, h*w, d_model] flattened feature maps
            input_spatial_shapes: [h, w] spatial shapes of feature maps
            
        Returns:
            output: [batch_size, num_queries, d_model]
        """
        batch_size, num_queries, _ = query.shape
        h, w = input_spatial_shapes
        
        # Calculate sampling offsets for each query
        # Shape: [batch_size, num_queries, n_heads, n_points, 2]
        offsets = self.sampling_offsets(query).view(
            batch_size, num_queries, self.n_heads, self.n_points, 2
        )
        
        # Calculate attention weights
        # Shape: [batch_size, num_queries, n_heads, n_points]
        weights = self.attention_weights(query).view(
            batch_size, num_queries, self.n_heads, self.n_points
        ).softmax(dim=-1)
        
        # Project values
        values = self.value_proj(input_flatten)
        
        # Expand reference points to all heads
        # Shape: [batch_size, num_queries, n_heads, 1, 2]
        reference = reference_points.unsqueeze(2).unsqueeze(3)
        
        # Add offsets to reference points and clip to valid range [0, 1]
        # Shape: [batch_size, num_queries, n_heads, n_points, 2]
        sampling_locations = reference + offsets
        sampling_locations = torch.clamp(sampling_locations, 0, 1)
        
        # Convert normalized coordinates to pixel coordinates
        # Shape: [batch_size, num_queries, n_heads, n_points, 2]
        sampling_locations_x = sampling_locations[..., 0] * (w - 1)
        sampling_locations_y = sampling_locations[..., 1] * (h - 1)
        
        # Get integer sampling coordinates
        x0 = torch.floor(sampling_locations_x).long()
        y0 = torch.floor(sampling_locations_y).long()
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Clip to feature map boundaries
        x0 = torch.clamp(x0, 0, w - 1)
        x1 = torch.clamp(x1, 0, w - 1)
        y0 = torch.clamp(y0, 0, h - 1)
        y1 = torch.clamp(y1, 0, h - 1)
        
        # Calculate interpolation weights
        wx1 = (sampling_locations_x - x0.float())
        wx0 = 1.0 - wx1
        wy1 = (sampling_locations_y - y0.float())
        wy0 = 1.0 - wy1
        
        # Reshape values for bilinear sampling
        # Shape: [batch_size, h, w, d_model]
        values = values.view(batch_size, h, w, self.d_model)
        
        # Perform bilinear sampling
        # Shape: [batch_size, num_queries, n_heads, n_points, d_model/n_heads]
        d_head = self.d_model // self.n_heads
        values_per_head = values.view(batch_size, h, w, self.n_heads, d_head)
        
        # Initialize result tensor
        result = torch.zeros(batch_size, num_queries, self.n_heads, 
                           self.n_points, d_head, device=query.device)
        
        # Sample values using bilinear interpolation
        for b in range(batch_size):
            for q in range(num_queries):
                for h_idx in range(self.n_heads):
                    for p in range(self.n_points):
                        x0_idx, y0_idx = x0[b, q, h_idx, p], y0[b, q, h_idx, p]
                        x1_idx, y1_idx = x1[b, q, h_idx, p], y1[b, q, h_idx, p]
                        
                        # Get interpolation weights
                        w00 = wx0[b, q, h_idx, p] * wy0[b, q, h_idx, p]
                        w01 = wx0[b, q, h_idx, p] * wy1[b, q, h_idx, p]
                        w10 = wx1[b, q, h_idx, p] * wy0[b, q, h_idx, p]
                        w11 = wx1[b, q, h_idx, p] * wy1[b, q, h_idx, p]
                        
                        # Perform bilinear interpolation
                        result[b, q, h_idx, p] = (
                            values_per_head[b, y0_idx, x0_idx, h_idx] * w00 +
                            values_per_head[b, y1_idx, x0_idx, h_idx] * w01 +
                            values_per_head[b, y0_idx, x1_idx, h_idx] * w10 +
                            values_per_head[b, y1_idx, x1_idx, h_idx] * w11
                        )
        
        # Apply attention weights
        # Shape: [batch_size, num_queries, n_heads, d_head]
        output = torch.sum(result * weights.unsqueeze(-1), dim=3)
        
        # Reshape output
        # Shape: [batch_size, num_queries, d_model]
        output = output.view(batch_size, num_queries, self.d_model)
        
        # Project output
        output = self.output_proj(output)
        
        return output


class DeformableDecoderLayer(nn.Module):
    """
    Deformable Decoder Layer combining self-attention, deformable cross-attention, and FFN.
    """
    def __init__(self, d_model=256, n_heads=8, dim_feedforward=2048, 
                 dropout=0.1, n_points=4):
        super().__init__()
        
        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Deformable cross-attention
        self.cross_attn = DeformableAttention(d_model, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Reference point projection (for generating sampling reference points)
        self.reference_points_proj = nn.Linear(d_model, 2)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Forward pass for the deformable decoder layer.
        
        Args:
            tgt: Target tensor [batch_size, num_queries, d_model]
            memory: Memory tensor from encoder [batch_size, h*w, d_model]
            tgt_mask: Mask for target tensor
            memory_mask: Mask for memory tensor
            
        Returns:
            Output tensor [batch_size, num_queries, d_model]
        """
        # Self-attention
        q = k = v = tgt.transpose(0, 1)  # [num_queries, batch_size, d_model]
        tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2.transpose(0, 1))
        tgt = self.norm1(tgt)
        
        # Calculate reference points from queries
        reference_points = self.reference_points_proj(tgt).sigmoid()  # [batch_size, num_queries, 2]
        
        # Spatial shapes of the feature map
        batch_size, hw, d_model = memory.shape
        h = w = int(hw ** 0.5)  # Assuming square feature maps
        
        # Deformable cross-attention
        tgt2 = self.cross_attn(tgt, reference_points, memory, (h, w))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class DeformableTransformerDecoder(nn.Module):
    """
    Deformable Transformer Decoder for object detection.
    """
    def __init__(self, decoder_layer, num_layers):
        """
        Initialize the deformable transformer decoder.
        
        Args:
            decoder_layer: Decoder layer instance to use
            num_layers: Number of decoder layers
        """
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Forward pass for the deformable transformer decoder.
        
        Args:
            tgt: Target tensor [batch_size, num_queries, d_model]
            memory: Memory tensor from encoder [batch_size, h*w, d_model]
            tgt_mask: Mask for target tensor
            memory_mask: Mask for memory tensor
            
        Returns:
            Output tensor [batch_size, num_queries, d_model]
        """
        output = tgt
        intermediate = []
        
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)
            intermediate.append(output)
        
        return output, intermediate