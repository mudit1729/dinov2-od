# matching.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .utils import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for Object Detection.
    This class computes an assignment between predicted and ground truth boxes
    using the Hungarian algorithm (linear_sum_assignment from scipy).
    
    The matching is done with a weighted sum of classification and bbox costs:
        Cost = λ_class * class_cost + λ_bbox * bbox_cost + λ_giou * giou_cost
        
    Where:
    - class_cost is the focal loss between predicted class probabilities and ground truth labels
    - bbox_cost is the L1 loss between predicted box coordinates and ground truth box coordinates
    - giou_cost is the generalized IoU loss between predicted boxes and ground truth boxes
    """
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2, focal_alpha=0.25, focal_gamma=2.0):
        """
        Initialize the Hungarian Matcher with cost weights
        
        Args:
            cost_class: Weight for classification cost
            cost_bbox: Weight for bbox L1 cost
            cost_giou: Weight for generalized IoU cost
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "at least one cost should be non-zero"
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Compute optimal assignment between predictions and ground truth
        
        Args:
            outputs: Dict with 'pred_logits' [batch_size, num_queries, num_classes] and
                     'pred_boxes' [batch_size, num_queries, 4]
            targets: List of dicts containing:
                    'labels': [num_gt_objects] class labels as integers
                    'boxes': [num_gt_objects, 4] boxes in (cx,cy,w,h) format
                    
        Returns:
            List[Tuple[Tensor, Tensor]]: List of tuples, each containing
                                         (index_i, index_j) where:
                                         - index_i is the indices of the selected predictions
                                         - index_j is the indices of the GT for each prediction
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # List of indices for the predictions and GT boxes for all batches
        indices = []
        
        # Process each item in the batch
        for b in range(bs):
            target = targets[b]
            if len(target) == 0:  # No GT boxes for this image
                indices.append((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)))
                continue
                
            tgt_ids = target["labels"]
            tgt_bbox = target["boxes"]
            
            # Classification cost using focal loss formulation
            alpha = self.focal_alpha
            gamma = self.focal_gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            
            # Cost matrix for classification: pick cost for the target class for each GT object
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            
            # Bbox cost: L1 distance between predicted and GT boxes (in (cx,cy,w,h) format)
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            
            # GIoU cost: negative GIoU as we want to maximize IoU
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox)
            )
            
            # Final cost matrix = weighted sum of all costs
            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            
            # Take only first num_queries predictions for matching
            # This ensures we only consider valid indices
            C_valid = C[:num_queries]
            
            # Hungarian algorithm to find optimal assignment (minimize cost)
            indices_i, indices_j = linear_sum_assignment(C_valid.cpu().numpy())
            
            # Convert to tensors and ensure indices_i are within valid range (num_queries)
            indices_i = torch.as_tensor(indices_i, dtype=torch.int64)
            indices_j = torch.as_tensor(indices_j, dtype=torch.int64)
            
            # Filter out invalid prediction indices
            valid_mask = indices_i < num_queries
            if not valid_mask.all():
                print(f"WARNING: Some prediction indices are out of range: max={indices_i.max()}, num_queries={num_queries}")
                indices_i = indices_i[valid_mask]
                indices_j = indices_j[valid_mask]
                
            indices.append((indices_i, indices_j))
        
        # Return batch indices, prediction indices, and GT indices
        return [(torch.as_tensor(i, dtype=torch.int64), 
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    """
    Build and return the Hungarian matcher based on arguments.
    """
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma
    )