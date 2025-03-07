# utils.py
import torch
import torch.nn as nn
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from tqdm import tqdm

class MLP(nn.Module):
    """
    Very simple multi-layer perceptron (used for bounding box prediction).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def add_lora_to_module(module, r=4, alpha=1.0):
    """
    Recursively replace nn.Linear layers with a LoRA-adapted version.
    Only replace layers that are instances of nn.Linear.
    """
    for name, child in module.named_children():
        # Recursively apply on child modules first
        add_lora_to_module(child, r=r, alpha=alpha)
        if isinstance(child, nn.Linear):
            # Replace with a LoRA-adapted linear layer
            setattr(module, name, LoraLinear(child, r=r, alpha=alpha))


class LoraLinear(nn.Module):
    """
    LoRA adapter applied to a linear layer.
    It freezes the original weight and learns a low-rank update.
    """
    def __init__(self, linear_layer: nn.Linear, r=4, alpha=1.0):
        super().__init__()
        self.linear = linear_layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.r = r
        self.alpha = alpha

        # Create low-rank matrices A and B
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        # Initialize lora_B with zeros so initially only original weight is used
        nn.init.zeros_(self.lora_B.weight)
        # Freeze original parameters
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Original output plus scaled low-rank update
        return self.linear(x) + self.alpha * self.lora_B(self.lora_A(x))


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding box from (center_x, center_y, width, height) format to (x1, y1, x2, y2) format.
    
    Args:
        x: tensor or array of shape (N, 4) where N is the number of boxes
        
    Returns:
        Converted boxes in (x1, y1, x2, y2) format
    """
    x_c, y_c, w, h = x.unbind(-1) if isinstance(x, torch.Tensor) else np.split(x, 4, axis=-1)
    
    if isinstance(x, torch.Tensor):
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    else:
        b = [x_c - 0.5 * w, y_c - 0.5 * h,
             x_c + 0.5 * w, y_c + 0.5 * h]
        return np.concatenate(b, axis=-1)


def evaluate_coco(model, dataloader, device, output_file=None):
    """
    Evaluate the model on COCO dataset.
    
    Args:
        model: The detector model
        dataloader: DataLoader for the COCO dataset
        device: Device to run evaluation on
        output_file: Path to save COCO detection results
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            
            # Forward pass to get predictions
            outputs = model(images)
            
            # Process predictions for COCO format
            pred_logits = outputs["pred_logits"]  # [batch_size, num_queries, num_classes]
            pred_boxes = outputs["pred_boxes"]    # [batch_size, num_queries, 4]
            
            # Convert scores and apply confidence threshold
            scores = torch.sigmoid(pred_logits)
            
            # For each image in the batch
            for i, target in enumerate(targets):
                img_id = target.get('image_id', i)
                
                # Process predictions for this image
                img_scores = scores[i]  # [num_queries, num_classes]
                img_boxes = pred_boxes[i]  # [num_queries, 4]
                
                # Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)
                img_boxes_xyxy = box_cxcywh_to_xyxy(img_boxes)
                
                # Select predictions with score > threshold for each class
                for cls_idx in range(img_scores.shape[1]):
                    if cls_idx == 0:  # Skip background class
                        continue
                        
                    cls_scores = img_scores[:, cls_idx]
                    keep = cls_scores > 0.05  # confidence threshold
                    
                    if not keep.any():
                        continue
                        
                    cls_scores = cls_scores[keep]
                    cls_boxes = img_boxes_xyxy[keep]
                    
                    # Create COCO detections
                    for score, box in zip(cls_scores.cpu().numpy(), cls_boxes.cpu().numpy()):
                        # Convert box to COCO format [x, y, width, height]
                        x1, y1, x2, y2 = box
                        coco_box = [float(x1), float(y1), float(x2-x1), float(y2-y1)]
                        
                        results.append({
                            'image_id': int(img_id),
                            'category_id': int(cls_idx),
                            'bbox': coco_box,
                            'score': float(score)
                        })
    
    # Save results if output file is provided
    if output_file is not None:
        with open(output_file, 'w') as f:
            json.dump(results, f)
    
    return results


def compute_coco_metrics(results, annotation_file):
    """
    Compute COCO metrics given detection results and ground truth annotations.
    
    Args:
        results: List of detection results in COCO format
        annotation_file: Path to COCO ground truth annotations
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    # Initialize COCO ground truth
    coco_gt = COCO(annotation_file)
    
    # Create COCO detection object
    coco_dt = coco_gt.loadRes(results)
    
    # Run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract metrics
    metrics = {
        'AP': coco_eval.stats[0],  # AP @ IoU=0.50:0.95
        'AP50': coco_eval.stats[1],  # AP @ IoU=0.50
        'AP75': coco_eval.stats[2],  # AP @ IoU=0.75
        'APs': coco_eval.stats[3],   # AP for small objects
        'APm': coco_eval.stats[4],   # AP for medium objects
        'APl': coco_eval.stats[5],   # AP for large objects
    }
    
    return metrics