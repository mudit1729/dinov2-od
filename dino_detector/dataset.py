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
        
        # For simplicity, we return dummy targets.
        target = {}
        # Process annotations for current image
        target['annotations'] = self.annotations.get(img_id, [])
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, target