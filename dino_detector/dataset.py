# dataset.py
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from PIL import Image
import os
import json

class COCODataset(Dataset):
    def __init__(self, images_dir, annotation_file, transform=None):
        """
        Args:
            images_dir: path to images directory.
            annotation_file: path to COCO annotations JSON.
            transform: optional transforms to apply.
        """
        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)
        self.images_dir = images_dir
        # Store the annotation file path for evaluation
        self.coco_path = annotation_file
        
        # Build an index for image id -> file name and annotations
        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        self.image_ids = list(self.images.keys())
        self.transform = transform
        
        # Get category information for class labels
        self.categories = {cat['id']: idx for idx, cat in enumerate(self.coco['categories'])}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Get an image and its annotations.
        
        Args:
            idx: Index of the dataset item
            
        Returns:
            tuple: (image, target) where image is the processed image tensor
                  and target contains the annotations for the image
        """
        # Get image information
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        # Get original image dimensions
        width, height = image.size if hasattr(image, 'size') else (
            img_info.get('width', 0), img_info.get('height', 0))
        
        # Apply transforms first to get target image size
        if self.transform is not None:
            image = self.transform(image)
        
        # Get target image dimensions (after transforms)
        img_h, img_w = image.shape[-2:]  # Assuming image is tensor [C, H, W]
            
        # Process annotations for current image
        anns = self.annotations.get(img_id, [])
        
        # Extract bounding boxes and labels in the format expected by the matcher
        boxes = []
        labels = []
        for ann in anns:
            if 'bbox' in ann and ann.get('iscrowd', 0) == 0:
                # COCO bbox is [x, y, width, height] in absolute coordinates
                x, y, w, h = ann['bbox']
                
                # Skip invalid boxes
                if w <= 0 or h <= 0:
                    continue
                
                # Convert to [center_x, center_y, width, height] and normalize
                cx = (x + w / 2) / width
                cy = (y + h / 2) / height
                nw = w / width
                nh = h / height
                    
                # Skip boxes that are too small or outside the image
                if nw < 0.001 or nh < 0.001 or cx <= 0 or cy <= 0 or cx >= 1 or cy >= 1:
                    continue
                
                # Add to the list
                boxes.append([cx, cy, nw, nh])
                
                # Get category and convert to zero-indexed label
                category_id = ann['category_id']
                label = self.categories.get(category_id, 0)  # Default to 0 if not found
                labels.append(label)
        
        # Format target dictionary with expected keys for the matcher
        target = {
            'image_id': img_id,
            'orig_size': torch.as_tensor([height, width]),
            'size': torch.as_tensor([img_h, img_w]),
            'filename': img_info['file_name'],
            'boxes': torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'area': torch.as_tensor([ann.get('area', 0) for ann in anns if 'bbox' in ann and ann.get('iscrowd', 0) == 0]),
            'iscrowd': torch.as_tensor([ann.get('iscrowd', 0) for ann in anns if 'bbox' in ann])
        }
            
        return image, target
        
        
class COCOTestDataset(Dataset):
    """
    COCO dataset specifically for test-dev evaluation.
    """
    def __init__(self, images_dir, annotation_file=None, transform=None):
        """
        Args:
            images_dir: path to images directory
            annotation_file: optional path to annotations (may not exist for test-dev)
            transform: optional transforms to apply
        """
        self.images_dir = images_dir
        self.transform = transform
        self.coco_path = annotation_file if annotation_file and os.path.exists(annotation_file) else None
        
        # If annotation file is provided, load it (useful for validation)
        if self.coco_path:
            with open(self.coco_path, 'r') as f:
                self.coco = json.load(f)
            self.images = {img['id']: img for img in self.coco['images']}
            self.image_ids = list(self.images.keys())
            
            # Get category information if available
            if 'categories' in self.coco:
                self.categories = {cat['id']: idx for idx, cat in enumerate(self.coco['categories'])}
            else:
                self.categories = {}
        else:
            # No annotations - just list all image files
            self.image_files = [f for f in os.listdir(images_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_ids = [int(os.path.splitext(f)[0]) for f in self.image_files]
            self.images = {img_id: {'file_name': f, 'id': img_id} 
                          for img_id, f in zip(self.image_ids, self.image_files)}
            self.categories = {}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Get an image and minimal information for evaluation.
        
        Args:
            idx: Index of the dataset item
            
        Returns:
            tuple: (image, target) where image is the processed image tensor
                  and target contains the image_id
        """
        # Get image information
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        # Get original image dimensions
        width, height = image.size
        
        # Apply transforms to get target image size
        if self.transform is not None:
            image = self.transform(image)
            
        # Get target image dimensions
        img_h, img_w = image.shape[-2:]  # Assuming image is tensor [C, H, W]
        
        # Create target information with properly formatted keys for the matcher
        # For test dataset, provide empty boxes and labels since we don't have annotations
        target = {
            'image_id': img_id,
            'orig_size': torch.as_tensor([height, width]),
            'size': torch.as_tensor([img_h, img_w]),
            'filename': img_info['file_name'],
            'boxes': torch.zeros((0, 4)),  # Empty tensor for boxes
            'labels': torch.zeros((0,), dtype=torch.int64),  # Empty tensor for labels
        }
            
        return image, target
def collate_fn(batch):
    """
    Custom collate function to handle variable sized annotations in the batch.
    
    Args:
        batch: List of (image, target) tuples from dataset
        
    Returns:
        tuple: (images, targets) where:
            - images is a tensor of shape [batch_size, C, H, W]
            - targets is a list of dictionaries
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Stack images into a single tensor
    images = torch.stack(images, dim=0)
    
    # Keep targets as a list of dictionaries
    # No need to stack or pad since we're keeping it as a list
    
    return images, targets
