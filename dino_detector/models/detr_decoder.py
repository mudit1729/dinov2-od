# models/detr_decoder.py
import torch
import torch.nn as nn
from ..utils import MLP
from .deformable_attention import DeformableDecoderLayer, DeformableTransformerDecoder

class DETRDecoder(nn.Module):
    def __init__(self, num_queries, hidden_dim, nheads, num_decoder_layers, num_classes, 
                 dim_feedforward=2048, dropout=0.1, n_points=4, use_deformable=True):
        super().__init__()
        self.num_queries = num_queries
        self.use_deformable = use_deformable
        
        # Learned object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Decoder - either standard transformer or deformable
        if use_deformable:
            print("Using Deformable Attention in Decoder")
            decoder_layer = DeformableDecoderLayer(
                d_model=hidden_dim, 
                n_heads=nheads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                n_points=n_points
            )
            self.decoder = DeformableTransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim, 
                nhead=nheads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Prediction heads: one for class scores and one for bounding boxes
        # Reduce hidden_dim for MLP to reduce parameters
        reduced_dim = hidden_dim // 2
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, reduced_dim, 4, num_layers=2)  # Reduced from 3 layers
        
        # Optional: additional reference points for deformable attention
        if use_deformable:
            self.reference_points = nn.Linear(hidden_dim, 2)

    def forward(self, src):
        """
        Args:
            src: Feature maps from backbone, shape [batch, seq_len, hidden_dim]
        Returns:
            A dict with keys:
              - "pred_logits": class predictions [batch, num_queries, num_classes]
              - "pred_boxes": bounding box predictions [batch, num_queries, 4] (normalized)
        """
        batch_size = src.size(0)
        
        # Prepare object queries: shape [batch, num_queries, hidden_dim]
        tgt = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # For standard transformer decoder
        if not self.use_deformable:
            # Transformer expects memory shape: [seq_len, batch, hidden_dim]
            memory = src.permute(1, 0, 2)  # [seq_len, batch, hidden_dim]
            tgt_input = tgt.permute(1, 0, 2)  # [num_queries, batch, hidden_dim]
            
            # Decode: output shape [num_queries, batch, hidden_dim]
            hs = self.decoder(tgt_input, memory)
            hs = hs.transpose(0, 1)  # shape: [batch, num_queries, hidden_dim]
        
        # For deformable transformer decoder
        else:
            # Memory is already [batch, seq_len, hidden_dim]
            memory = src
            
            # Decode using deformable attention
            hs, _ = self.decoder(tgt, memory)
        
        # Predict classes and boxes
        outputs_class = self.class_embed(hs)
        outputs_bbox = self.bbox_embed(hs).sigmoid()  # use sigmoid to normalize bbox coordinates
        
        return {"pred_logits": outputs_class, "pred_boxes": outputs_bbox}