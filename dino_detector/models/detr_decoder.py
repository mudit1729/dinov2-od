# models/detr_decoder.py
import torch
import torch.nn as nn
from ..utils import MLP

class DETRDecoder(nn.Module):
    def __init__(self, num_queries, hidden_dim, nheads, num_decoder_layers, num_classes):
        super().__init__()
        self.num_queries = num_queries
        # Learned object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Prediction heads: one for class scores and one for bounding boxes
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

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
        # Prepare object queries: shape [num_queries, batch, hidden_dim]
        tgt = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        # Transformer expects memory shape: [seq_len, batch, hidden_dim]
        memory = src.permute(1, 0, 2)
        # Decode: output shape [num_queries, batch, hidden_dim]
        hs = self.decoder(tgt, memory)
        hs = hs.transpose(0, 1)  # shape: [batch, num_queries, hidden_dim]
        outputs_class = self.class_embed(hs)
        outputs_bbox = self.bbox_embed(hs).sigmoid()  # use sigmoid to normalize bbox coordinates
        return {"pred_logits": outputs_class, "pred_boxes": outputs_bbox}