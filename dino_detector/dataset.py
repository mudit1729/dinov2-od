# dataset.py
import torch
from torch.utils.data import Dataset
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
        
        # Process annotations for current image
        target = {
            'image_id': img_id,
            'annotations': self.annotations.get(img_id, []),
            'orig_size': torch.as_tensor([img_info.get('height', 0), img_info.get('width', 0)]),
            'filename': img_info['file_name']
        }
        
        if self.transform is not None:
            image = self.transform(image)
            
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
        
        # If annotation file is provided, load it (useful for validation)
        if annotation_file and os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                self.coco = json.load(f)
            self.images = {img['id']: img for img in self.coco['images']}
            self.image_ids = list(self.images.keys())
        else:
            # No annotations - just list all image files
            self.image_files = [f for f in os.listdir(images_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_ids = [int(os.path.splitext(f)[0]) for f in self.image_files]
            self.images = {img_id: {'file_name': f, 'id': img_id} 
                          for img_id, f in zip(self.image_ids, self.image_files)}

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
        
        # Get original image size
        width, height = image.size
        
        # Create minimal target information
        target = {
            'image_id': img_id,
            'orig_size': torch.as_tensor([height, width]),
            'filename': img_info['file_name']
        }
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, target