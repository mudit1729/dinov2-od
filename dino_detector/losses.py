# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .utils import box_cxcywh_to_xyxy, generalized_box_iou


class FocalLoss(nn.Module):
    """
    Focal Loss for Object Detection.
    Implements the focal loss from "Focal Loss for Dense Object Detection" paper.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    where p_t is the predicted probability of the target class.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for the rare class (typically foreground)
            gamma: Focusing parameter, reduces the loss contribution from easy examples
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs: Tensor of shape [N, C] with raw logits
            targets: Tensor of shape [N] with class labels
            
        Returns:
            Loss tensor according to the reduction method
        """
        probs = inputs.sigmoid()
        
        # Create one-hot encoding for targets
        num_classes = inputs.shape[1]
        target_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Compute the focal loss
        pt = target_one_hot * probs + (1 - target_one_hot) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = target_one_hot * self.alpha + (1 - target_one_hot) * (1 - self.alpha)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(inputs, target_one_hot, reduction='none')
        
        # Compute weighted loss
        loss = alpha_weight * focal_weight * bce
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class SetCriterion(nn.Module):
    """
    Loss criterion for object detection based on DETR-style bipartite matching.
    
    The criterion performs a bipartite matching between predicted and ground-truth objects
    and optimizes a set-based loss that considers classification, bounding boxes, and GIoU.
    """
    def __init__(self, matcher, num_classes, weight_dict, focal_alpha=0.25, focal_gamma=2.0):
        """
        Initialize the criterion.
        
        Args:
            matcher: Module that computes the matching between predictions and targets
            num_classes: Number of object categories (without special tokens)
            weight_dict: Dict containing weights for different losses
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss using focal loss.
        
        Args:
            outputs: Dict with 'pred_logits' of shape [batch_size, num_queries, num_classes]
            targets: List of dicts containing 'labels'
            indices: List of tuples with (prediction_indices, target_indices)
            num_boxes: Total number of boxes (normalization factor)
            
        Returns:
            Dict of classification losses
        """
        src_logits = outputs['pred_logits']
        
        # Extract matched indices
        idx = self._get_src_permutation_idx(indices)
        
        # Get target classes for matched predictions
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        # Compute focal loss for classification
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                           dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        
        pt = src_logits.sigmoid() * target_classes_onehot + (1 - src_logits.sigmoid()) * (1 - target_classes_onehot)
        focal_weight = (1 - pt) ** self.focal_gamma
        alpha_weight = self.focal_alpha * target_classes_onehot + (1 - self.focal_alpha) * (1 - target_classes_onehot)
        
        loss_ce = F.binary_cross_entropy_with_logits(
            src_logits, target_classes_onehot, reduction='none'
        )
        loss_ce = alpha_weight * focal_weight * loss_ce
        loss_ce = loss_ce.sum() / num_boxes
        
        return {'loss_ce': loss_ce}
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Bounding box regression losses: L1 loss and GIoU loss.
        
        Args:
            outputs: Dict with 'pred_boxes' of shape [batch_size, num_queries, 4]
            targets: List of dicts containing 'boxes'
            indices: List of tuples with (prediction_indices, target_indices)
            num_boxes: Total number of boxes (normalization factor)
            
        Returns:
            Dict of box regression losses
        """
        assert 'pred_boxes' in outputs
        
        # Get the indices of matched predictions
        idx = self._get_src_permutation_idx(indices)
        
        # Get predicted boxes for matched predictions
        src_boxes = outputs['pred_boxes'][idx]
        
        # Get target boxes for matched predictions (from all target boxes)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # Compute L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes
        
        # Compute GIoU loss
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        loss_giou = loss_giou.sum() / num_boxes
        
        return {
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou
        }
    
    def _get_src_permutation_idx(self, indices):
        """
        Helper function to get indices of matched predictions.
        """
        # Concatenate all source (prediction) indices with their corresponding batch indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices):
        """
        Helper function to get indices of matched ground-truth objects.
        """
        # Concatenate all target indices with their corresponding batch indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
        
    def forward(self, outputs, targets):
        """
        Compute the detection losses.
        
        Args:
            outputs: Dict with 'pred_logits' and 'pred_boxes'
            targets: List of dicts containing 'labels' and 'boxes'
            
        Returns:
            Dict of losses with applied weights
        """
        # Compute the optimal assignment between predictions and GT
        indices = self.matcher(outputs, targets)
        
        # Calculate the number of matched boxes (to normalize the losses)
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # Handle distributed training
        if dist.is_available() and dist.is_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes, min=1).item()
        
        # Compute all the losses
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        
        # Apply weights to the losses
        weighted_losses = {k: self.weight_dict[k] * losses[k] if k in self.weight_dict else losses[k]
                          for k in losses}
        
        return weighted_losses


def build_criterion(matcher, num_classes, weight_dict, focal_alpha=0.25, focal_gamma=2.0):
    """
    Build and return the criterion based on the matcher.
    """
    return SetCriterion(
        matcher=matcher,
        num_classes=num_classes,
        weight_dict=weight_dict,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma
    )