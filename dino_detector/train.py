# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import json
from models.detector import DINOv2ObjectDetector
from dataset import COCODataset, COCOTestDataset
from utils import evaluate_coco, compute_coco_metrics
from config import batch_size, num_epochs, learning_rate, weight_decay, num_workers
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Dummy loss function for demonstration
def dummy_loss(outputs):
    """
    A simplified loss function for demonstration purposes.
    
    In a real-world implementation, you would need to implement:
    1. Bipartite matching (e.g., Hungarian algorithm) to assign predictions to ground truth
    2. Classification loss (e.g., focal loss)
    3. Bounding box regression loss (e.g., L1 or GIoU loss)
    
    Args:
        outputs: dict with "pred_logits" and "pred_boxes"
        
    Returns:
        A scalar loss value
    """
    # Here we simply return the sum of logits and bbox predictions as a dummy loss.
    loss = outputs["pred_logits"].sum() + outputs["pred_boxes"].sum()
    return loss

def validate(model, val_dataloader, device, epoch, output_dir):
    """
    Validate the model on the validation set.
    
    Args:
        model: The model to evaluate
        val_dataloader: DataLoader for validation data
        device: Device to run evaluation on
        epoch: Current epoch number
        output_dir: Directory to save results
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    print(f"Running validation for epoch {epoch}...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate predictions on validation set
    results_file = os.path.join(output_dir, f"val_predictions_epoch_{epoch}.json")
    results = evaluate_coco(model, val_dataloader, device, results_file)
    
    # Compute metrics
    val_annotation_file = val_dataloader.dataset.coco_path if hasattr(val_dataloader.dataset, 'coco_path') else None
    
    if val_annotation_file and os.path.exists(val_annotation_file):
        metrics = compute_coco_metrics(results, val_annotation_file)
        
        # Save metrics to file
        metrics_file = os.path.join(output_dir, f"val_metrics_epoch_{epoch}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        print(f"Validation AP: {metrics['AP']:.4f}, AP50: {metrics['AP50']:.4f}, AP75: {metrics['AP75']:.4f}")
        return metrics
    else:
        print("No validation annotations available, skipping metrics computation")
        return None

def plot_metrics(metrics_history, output_dir):
    """
    Plot training and validation metrics.
    
    Args:
        metrics_history: Dictionary with metrics history
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_history['epochs'], metrics_history['train_loss'], 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()
    
    # Plot validation metrics if available
    if 'val_ap' in metrics_history and len(metrics_history['val_ap']) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(metrics_history['val_epochs'], metrics_history['val_ap'], 'r-', label='mAP')
        plt.plot(metrics_history['val_epochs'], metrics_history['val_ap50'], 'g-', label='AP50')
        plt.plot(metrics_history['val_epochs'], metrics_history['val_ap75'], 'b-', label='AP75')
        plt.xlabel('Epoch')
        plt.ylabel('AP')
        plt.title('Validation AP Metrics')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'validation_ap.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train DINOv2 Object Detector')
    parser.add_argument('--train_images', type=str, default="path/to/coco/train2017", 
                        help='Path to training images')
    parser.add_argument('--train_annotations', type=str, default="path/to/coco/annotations/instances_train2017.json", 
                        help='Path to training annotations')
    parser.add_argument('--val_images', type=str, default="path/to/coco/val2017", 
                        help='Path to validation images')
    parser.add_argument('--val_annotations', type=str, default="path/to/coco/annotations/instances_val2017.json", 
                        help='Path to validation annotations')
    parser.add_argument('--testdev_images', type=str, default="", 
                        help='Path to test-dev images (optional)')
    parser.add_argument('--output_dir', type=str, default="outputs", 
                        help='Directory to save outputs')
    parser.add_argument('--checkpoint', type=str, default="", 
                        help='Path to model checkpoint to resume training (optional)')
    parser.add_argument('--val_frequency', type=int, default=5, 
                        help='Validate every N epochs')
    parser.add_argument('--only_evaluate', action='store_true', 
                        help='Only run evaluation, no training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define transforms for input images (resize and convert to tensor)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize detector model
    print("Initializing DINOv2 Object Detector...")
    model = DINOv2ObjectDetector()
    model = model.to(device)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    metrics_history = {
        'epochs': [], 'train_loss': [],
        'val_epochs': [], 'val_ap': [], 'val_ap50': [], 'val_ap75': []
    }
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        
        # Load metrics history if available
        if 'metrics_history' in checkpoint:
            metrics_history = checkpoint['metrics_history']
            
        print(f"Resuming from epoch {start_epoch}")
    
    # If only evaluating, skip to validation
    if args.only_evaluate:
        print("Running evaluation only...")
        
        if args.testdev_images:
            # Test-dev evaluation
            print(f"Evaluating on test-dev set: {args.testdev_images}")
            test_dataset = COCOTestDataset(args.testdev_images, transform=transform)
            test_dataloader = DataLoader(
                test_dataset, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
            print(f"Test dataset loaded with {len(test_dataset)} images")
            
            # Generate predictions for test-dev
            test_results_file = os.path.join(args.output_dir, "testdev_predictions.json")
            model.eval()
            test_results = evaluate_coco(model, test_dataloader, device, test_results_file)
            print(f"Test-dev predictions saved to {test_results_file}")
        
        # Validation set evaluation
        if os.path.exists(args.val_images) and os.path.exists(args.val_annotations):
            val_dataset = COCODataset(args.val_images, args.val_annotations, transform=transform)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
            print(f"Validation dataset loaded with {len(val_dataset)} images")
            
            model.eval()
            _ = validate(model, val_dataloader, device, 0, args.output_dir)
        
        return
    
    # Initialize training dataset and dataloader
    print(f"Loading training dataset from {args.train_images}")
    train_dataset = COCODataset(args.train_images, args.train_annotations, transform=transform)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    print(f"Training dataset loaded with {len(train_dataset)} images")
    
    # Initialize validation dataset and dataloader if paths exist
    val_dataloader = None
    if os.path.exists(args.val_images) and os.path.exists(args.val_annotations):
        print(f"Loading validation dataset from {args.val_images}")
        val_dataset = COCODataset(args.val_images, args.val_annotations, transform=transform)
        val_dataset.coco_path = args.val_annotations  # Store annotation path for evaluation
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        print(f"Validation dataset loaded with {len(val_dataset)} images")

    # Collect parameters that require gradients (decoder and LORA parameters)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Load optimizer state if resuming training
    if args.checkpoint and os.path.exists(args.checkpoint) and not args.only_evaluate:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Starting training for {num_epochs} epochs")
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, targets in pbar:
            images = images.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = dummy_loss(outputs)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))
            
        # Print epoch statistics
        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
        
        # Update metrics history
        metrics_history['epochs'].append(epoch + 1)
        metrics_history['train_loss'].append(epoch_loss)
        
        # Validation phase
        if val_dataloader is not None and (epoch + 1) % args.val_frequency == 0:
            model.eval()
            metrics = validate(model, val_dataloader, device, epoch + 1, args.output_dir)
            
            if metrics:
                # Update validation metrics history
                metrics_history['val_epochs'].append(epoch + 1)
                metrics_history['val_ap'].append(metrics['AP'])
                metrics_history['val_ap50'].append(metrics['AP50'])
                metrics_history['val_ap75'].append(metrics['AP75'])
        
        # Plot metrics
        plot_metrics(metrics_history, args.output_dir)
        
        # Save checkpoint every 10 epochs or on the last epoch
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f"dino_detector_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'metrics_history': metrics_history
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Save the final model
    final_model_path = os.path.join(args.output_dir, "dino_detector_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")
    
    # Final evaluation on test-dev if provided
    if args.testdev_images:
        print(f"Evaluating final model on test-dev set: {args.testdev_images}")
        test_dataset = COCOTestDataset(args.testdev_images, transform=transform)
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # Generate predictions for test-dev
        model.eval()
        test_results_file = os.path.join(args.output_dir, "testdev_predictions_final.json")
        test_results = evaluate_coco(model, test_dataloader, device, test_results_file)
        print(f"Test-dev predictions saved to {test_results_file}")

if __name__ == "__main__":
    main()