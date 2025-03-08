# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import argparse
import json
import sys
import urllib.request
import zipfile
from tqdm import tqdm
from dino_detector.models.detector import DINOv2ObjectDetector
from dino_detector.dataset import COCODataset, COCOTestDataset, collate_fn
from dino_detector.utils import evaluate_coco, compute_coco_metrics
from dino_detector.matching import HungarianMatcher, build_matcher
from dino_detector.losses import SetCriterion, build_criterion
from dino_detector.config import (
    batch_size, num_epochs, learning_rate, weight_decay, num_workers,
    distributed_backend, find_unused_parameters, gradient_accumulation_steps,
    gradient_clip_val,
    set_cost_class, set_cost_bbox, set_cost_giou, focal_alpha, focal_gamma,
    loss_weights, num_classes,
    debug_mode, debug_dataset_size, debug_epochs, debug_learning_rate,
    use_deformable, n_points, dim_feedforward, dropout
)
from torchvision import transforms
import matplotlib.pyplot as plt

# COCO dataset URLs
COCO_URLS = {
    'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
    'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
    'test_images': 'http://images.cocodataset.org/zips/test2017.zip',
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
}

# Create losses and matcher
def create_debug_subset(dataset, num_samples):
    """
    Create a small subset of the dataset for debugging/overfitting.
    
    Args:
        dataset: The original dataset
        num_samples: Number of samples to include in the subset
        
    Returns:
        A subset of the original dataset
    """
    # Get a small subset of the dataset for overfitting
    from torch.utils.data import Subset
    import random
    
    # Make sure we don't try to get more samples than exist in the dataset
    num_samples = min(num_samples, len(dataset))
    
    # Choose random indices without replacement
    random.seed(42)  # For reproducibility
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Create and return a subset with those indices
    return Subset(dataset, indices)

def create_criterion(args):
    """
    Create and return the criterion and matcher for object detection.
    
    Args:
        args: Command line arguments
        
    Returns:
        criterion: Loss criterion with Hungarian matching and losses
    """
    # Create the matcher
    matcher = HungarianMatcher(
        cost_class=set_cost_class,
        cost_bbox=set_cost_bbox,
        cost_giou=set_cost_giou,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma
    )
    
    # Create the criterion
    criterion = SetCriterion(
        matcher=matcher,
        num_classes=args.num_classes,
        weight_dict=loss_weights,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma
    )
    
    return criterion

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

def download_file(url, destination, desc=None):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download
        destination: Path to save the file
        desc: Description for the progress bar
    """
    if not desc:
        desc = os.path.basename(destination)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(destination):
        print(f"{desc} already exists. Skipping download.")
        return
    
    # Download with progress bar
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=desc) as progress:
        def report_progress(count, block_size, total_size):
            if total_size > 0:
                progress.total = total_size
                progress.update(count * block_size - progress.n)
        
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)

def extract_archive(archive_path, extract_dir, desc=None, debug_mode=False, max_samples=None):
    """
    Extract an archive file. In debug mode, limits the number of image files extracted.
    
    Args:
        archive_path: Path to the archive file
        extract_dir: Directory to extract to
        desc: Description for the progress bar
        debug_mode: Whether to extract only a limited number of files (for debug mode)
        max_samples: Maximum number of image files to extract in debug mode
    """
    if not desc:
        desc = f"Extracting {os.path.basename(archive_path)}"
    
    if debug_mode and max_samples is not None:
        desc = f"{desc} (debug mode, max {max_samples} samples)"
    
    print(f"{desc}...")
    
    # Create directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract based on file extension
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            # Get the list of files
            file_list = zip_ref.namelist()
            
            # In debug mode, limit the number of image files
            if debug_mode and max_samples is not None and 'images' in archive_path:
                # Always extract directories
                dirs = [f for f in file_list if f.endswith('/')]
                
                # Extract only a limited number of image files
                image_files = [f for f in file_list if f.endswith('.jpg') and not f.endswith('/')]
                if len(image_files) > max_samples:
                    image_files = image_files[:max_samples]
                
                # Extract all non-image files (like annotations, metadata)
                other_files = [f for f in file_list if not f.endswith('.jpg') and not f.endswith('/')]
                
                # Final list of files to extract
                files_to_extract = dirs + image_files + other_files
                print(f"Debug mode: Extracting {len(image_files)} images " +
                      f"and {len(other_files)} other files")
                
                total_files = len(files_to_extract)
            else:
                files_to_extract = file_list
                total_files = len(file_list)
            
            # Extract the files
            with tqdm(total=total_files, desc=desc) as pbar:
                for file in files_to_extract:
                    zip_ref.extract(file, extract_dir)
                    pbar.update(1)
    else:
        print(f"Unsupported archive format: {archive_path}")

def download_coco_dataset(args):
    """
    Download and extract COCO dataset based on command line arguments.
    In debug mode, only extract a limited number of samples.
    
    Args:
        args: Command line arguments
        
    Returns:
        dict: Updated argument namespace with dataset paths
    """
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    
    downloads_dir = os.path.join(data_dir, 'downloads')
    os.makedirs(downloads_dir, exist_ok=True)
    
    # Determine which parts of the dataset to download
    download_parts = []
    
    if args.download_train_data:
        if not args.train_images or not args.train_annotations:
            download_parts.extend(['train_images', 'annotations'])
    
    if args.download_val_data:
        if not args.val_images or not args.val_annotations:
            download_parts.extend(['val_images', 'annotations'])
    
    if args.download_test_data:
        if not args.testdev_images:
            download_parts.append('test_images')
    
    # Remove duplicates
    download_parts = list(set(download_parts))
    
    if not download_parts:
        return args
    
    # Check if in debug mode
    is_debug_mode = getattr(args, 'debug', False)
    max_samples = getattr(args, 'debug_samples', debug_dataset_size) if is_debug_mode else None
    
    if is_debug_mode:
        print(f"Debug mode active: Will extract max {max_samples} samples for image archives")
    
    # Download and extract the specified parts
    for part in download_parts:
        # Download
        download_file(
            COCO_URLS[part],
            os.path.join(downloads_dir, f"{part}.zip"),
            f"Downloading {part}"
        )
        
        # Extract
        extract_archive(
            os.path.join(downloads_dir, f"{part}.zip"),
            data_dir,
            f"Extracting {part}",
            debug_mode=is_debug_mode,
            max_samples=max_samples
        )
    
    # Update paths in args if not provided
    if args.download_train_data and (not args.train_images or not args.train_annotations):
        args.train_images = os.path.join(data_dir, 'train2017')
        args.train_annotations = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
        print(f"Set training paths to:\n- Images: {args.train_images}\n- Annotations: {args.train_annotations}")
    
    if args.download_val_data and (not args.val_images or not args.val_annotations):
        args.val_images = os.path.join(data_dir, 'val2017')
        args.val_annotations = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
        print(f"Set validation paths to:\n- Images: {args.val_images}\n- Annotations: {args.val_annotations}")
    
    if args.download_test_data and not args.testdev_images:
        args.testdev_images = os.path.join(data_dir, 'test2017')
        print(f"Set test-dev path to: {args.testdev_images}")
    
    return args

def setup_distributed(rank, world_size):
    """
    Initialize the distributed environment.
    
    Args:
        rank: Unique identifier of the process
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(backend=distributed_backend, rank=rank, world_size=world_size)
    
    # Set cuda device for this process
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()

def main_worker(rank, world_size, args):
    """
    Main worker function for distributed training.
    
    Args:
        rank: Unique identifier of the process
        world_size: Total number of processes
        args: Command line arguments
    """
    # Initialize distributed environment
    if args.distributed:
        setup_distributed(rank, world_size)
        print(f"Initialized process group: rank {rank}/{world_size}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download COCO dataset if requested (download first, validate after)
    if rank == 0 and (args.download_train_data or args.download_val_data or args.download_test_data):
        args = download_coco_dataset(args)
        
    if args.distributed:
        # Wait for rank 0 to download data
        dist.barrier()
        
    # Validate that we have at least training data for training or validation data for evaluation
    # Only after potential download has completed
    if not args.only_evaluate and (not args.train_images or not args.train_annotations):
        print(f"Process {rank}: Error: Training images and annotations are required for training.")
        print(f"Process {rank}:        Use --download_train_data to download COCO training data")
        print(f"Process {rank}:        or provide --train_images and --train_annotations paths.")
        return
        
    if args.only_evaluate and not (args.val_images and args.val_annotations) and not args.testdev_images:
        print(f"Process {rank}: Error: Validation or test-dev images are required for evaluation.")
        print(f"Process {rank}:        Use --download_val_data to download COCO validation data")
        print(f"Process {rank}:        or provide --val_images and --val_annotations paths.")
        print(f"Process {rank}:        Alternatively, use --download_test_data to download test-dev data")
        print(f"Process {rank}:        or provide --testdev_images path.")
        return
    
    # Define transforms for input images (resize and convert to tensor)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Initialize device
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"Process {rank}: Using device: {device}")
    
    # Initialize detector model
    print(f"Process {rank}: Initializing DINOv2 Object Detector...")
    model = DINOv2ObjectDetector(num_classes=args.num_classes)
    model = model.to(device)
    
    # Create the criterion for loss computation
    criterion = create_criterion(args)
    criterion = criterion.to(device)
    
    # Wrap the model with DDP if using distributed training
    if args.distributed:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused_parameters)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    metrics_history = {
        'epochs': [], 'train_loss': [],
        'val_epochs': [], 'val_ap': [], 'val_ap50': [], 'val_ap75': []
    }
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Process {rank}: Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if args.distributed:
            # Load model state dict for DDP model (need to handle 'module.' prefix)
            state_dict = checkpoint['model_state_dict']
            if not any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        else:
            # Remove 'module.' prefix if present for non-DDP model
            state_dict = checkpoint['model_state_dict']
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            
        start_epoch = checkpoint.get('epoch', 0) + 1
        
        # Load metrics history if available (only on rank 0)
        if rank == 0 and 'metrics_history' in checkpoint:
            metrics_history = checkpoint['metrics_history']
            
        print(f"Process {rank}: Resuming from epoch {start_epoch}")
    
    # If only evaluating, skip to validation
    if args.only_evaluate:
        print(f"Process {rank}: Running evaluation only...")
        
        if args.testdev_images:
            # Test-dev evaluation
            print(f"Process {rank}: Evaluating on test-dev set: {args.testdev_images}")
            test_dataset = COCOTestDataset(args.testdev_images, transform=transform)
            
            # Use DistributedSampler for distributed evaluation
            if args.distributed:
                test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
                test_dataloader = DataLoader(
                    test_dataset, 
                    batch_size=args.batch_size,
                    sampler=test_sampler,
                    num_workers=num_workers,
                    collate_fn=collate_fn
                )
            else:
                test_dataloader = DataLoader(
                    test_dataset, 
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=collate_fn
                )
            
            print(f"Process {rank}: Test dataset loaded with {len(test_dataset)} images")
            
            # Generate predictions for test-dev
            test_results_file = os.path.join(args.output_dir, f"testdev_predictions_rank{rank}.json")
            model.eval()
            test_results = evaluate_coco(model, test_dataloader, device, test_results_file)
            print(f"Process {rank}: Test-dev predictions saved to {test_results_file}")
        
        # Validation set evaluation
        if os.path.exists(args.val_images) and os.path.exists(args.val_annotations):
            val_dataset = COCODataset(args.val_images, args.val_annotations, transform=transform)
            
            # Use DistributedSampler for distributed evaluation
            if args.distributed:
                val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    sampler=val_sampler,
                    num_workers=num_workers,
                    collate_fn=collate_fn
                )
            else:
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=collate_fn
                )
                
            print(f"Process {rank}: Validation dataset loaded with {len(val_dataset)} images")
            
            model.eval()
            _ = validate(model, val_dataloader, device, 0, args.output_dir)
        
        # Clean up distributed environment
        if args.distributed:
            cleanup_distributed()
            
        return
    
    # Initialize training dataset and dataloader
    print(f"Process {rank}: Loading training dataset from {args.train_images}")
    train_dataset = COCODataset(args.train_images, args.train_annotations, transform=transform)
    
    # Create a small subset for debug/overfit mode if requested
    if args.debug:
        original_size = len(train_dataset)
        train_dataset = create_debug_subset(train_dataset, args.debug_samples)
        print(f"Process {rank}: DEBUG MODE - Using {len(train_dataset)} training samples out of {original_size}")
    
    # Use DistributedSampler for distributed training
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            sampler=train_sampler, 
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
    print(f"Process {rank}: Training dataset loaded with {len(train_dataset)} images")
    
    # Initialize validation dataset and dataloader if paths exist
    val_dataloader = None
    if os.path.exists(args.val_images) and os.path.exists(args.val_annotations):
        print(f"Process {rank}: Loading validation dataset from {args.val_images}")
        val_dataset = COCODataset(args.val_images, args.val_annotations, transform=transform)
        val_dataset.coco_path = args.val_annotations  # Store annotation path for evaluation
        
        # In debug mode, use the same small subset for validation
        # This helps with overfitting verification
        if args.debug:
            original_val_size = len(val_dataset)
            val_dataset = create_debug_subset(val_dataset, args.debug_samples * 2)  # Increase validation samples
            print(f"Process {rank}: DEBUG MODE - Using {len(val_dataset)} validation samples out of {original_val_size}")
        
        # Use DistributedSampler for distributed evaluation
        if args.distributed:
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                sampler=val_sampler,
                num_workers=num_workers,
                collate_fn=collate_fn
            )
        else:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn
            )
            
        print(f"Process {rank}: Validation dataset loaded with {len(val_dataset)} images")

    # Collect parameters that require gradients (decoder and LORA parameters)
    # Use higher learning rate in debug mode
    lr = args.debug_lr if args.debug else learning_rate
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr,
        weight_decay=weight_decay
    )
    
    if args.debug:
        print(f"Process {rank}: DEBUG MODE - Using learning rate: {lr}")
    
    # Load optimizer state if resuming training
    if args.checkpoint and os.path.exists(args.checkpoint) and not args.only_evaluate:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Use more epochs in debug mode for overfitting
    epochs_to_train = debug_epochs if args.debug else num_epochs
    print(f"Process {rank}: Starting training for {epochs_to_train} epochs")
    
    # Set validation frequency more frequent in debug mode for better monitoring
    val_freq = max(1, args.val_frequency // 5) if args.debug else args.val_frequency
    if args.debug and rank == 0:
        print(f"Process {rank}: DEBUG MODE - Validating every {val_freq} epochs")
        
    for epoch in range(start_epoch, epochs_to_train):
        # Set epoch for distributed sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)
            
        # Training phase
        model.train()
        running_loss = 0.0
        
        if rank == 0:  # Only rank 0 shows progress bar
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            iterator = pbar
        else:
            iterator = train_dataloader
        
        for batch_idx, (images, targets) in enumerate(iterator):
            images = images.to(device)
            
            # Process targets to ensure they are on the correct device
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # Zero gradients at the beginning of the batch
            if batch_idx % gradient_accumulation_steps == 0:
                optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss using criterion with Hungarian matching
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())
            
            # Backward pass and optimize with gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Apply gradient clipping
            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                
            # Step optimizer every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Update statistics
            running_loss += loss.item()
            if rank == 0:
                # Display individual loss components
                loss_str = f"total: {loss.item():.3f} "
                loss_str += " ".join(f"{k}: {v.item():.3f}" for k, v in loss_dict.items())
                pbar.set_postfix_str(loss_str)
                
        # Calculate average loss across all processes
        if args.distributed:
            world_size = dist.get_world_size()
            loss_tensor = torch.tensor(running_loss, device=device) / len(train_dataloader)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = loss_tensor.item() / world_size
        else:
            epoch_loss = running_loss / len(train_dataloader)
            
        # Print epoch statistics (only rank 0)
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
            
            # Update metrics history
            metrics_history['epochs'].append(epoch + 1)
            metrics_history['train_loss'].append(epoch_loss)
        
        # Validation phase
        if val_dataloader is not None and (epoch + 1) % val_freq == 0:
            model.eval()
            
            # Only rank 0 runs validation and reports metrics
            if rank == 0:
                metrics = validate(model, val_dataloader, device, epoch + 1, args.output_dir)
                
                if metrics:
                    # Update validation metrics history
                    metrics_history['val_epochs'].append(epoch + 1)
                    metrics_history['val_ap'].append(metrics['AP'])
                    metrics_history['val_ap50'].append(metrics['AP50'])
                    metrics_history['val_ap75'].append(metrics['AP75'])
                    
                # Plot metrics
                plot_metrics(metrics_history, args.output_dir)
            
            # Wait for validation to finish before continuing training
            if args.distributed:
                dist.barrier()
        
        # Save checkpoint every 10 epochs or on the last epoch (only rank 0)
        if rank == 0 and ((epoch + 1) % 10 == 0 or epoch == num_epochs - 1):
            checkpoint_path = os.path.join(args.output_dir, f"dino_detector_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'metrics_history': metrics_history
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Save the final model (only rank 0)
    if rank == 0:
        final_model_path = os.path.join(args.output_dir, "dino_detector_final.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"Training complete. Final model saved to {final_model_path}")
    
    # Final evaluation on test-dev if provided
    if rank == 0 and args.testdev_images:
        print(f"Evaluating final model on test-dev set: {args.testdev_images}")
        test_dataset = COCOTestDataset(args.testdev_images, transform=transform)
        
        # Use non-distributed dataloader for final evaluation on rank 0
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        # Generate predictions for test-dev
        model.eval()
        test_results_file = os.path.join(args.output_dir, "testdev_predictions_final.json")
        test_results = evaluate_coco(model, test_dataloader, device, test_results_file)
        print(f"Test-dev predictions saved to {test_results_file}")
    
    # Clean up distributed environment
    if args.distributed:
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description='Train DINOv2 Object Detector')
    
    # Dataset paths
    parser.add_argument('--train_images', type=str, default="", 
                        help='Path to training images')
    parser.add_argument('--train_annotations', type=str, default="", 
                        help='Path to training annotations')
    parser.add_argument('--val_images', type=str, default="", 
                        help='Path to validation images')
    parser.add_argument('--val_annotations', type=str, default="", 
                        help='Path to validation annotations')
    parser.add_argument('--testdev_images', type=str, default="", 
                        help='Path to test-dev images (optional)')
    
    # Dataset download options
    parser.add_argument('--data_dir', type=str, default="coco_data",
                        help='Directory to store downloaded dataset')
    parser.add_argument('--download_train_data', action='store_true',
                        help='Download COCO training data')
    parser.add_argument('--download_val_data', action='store_true',
                        help='Download COCO validation data')
    parser.add_argument('--download_test_data', action='store_true',
                        help='Download COCO test-dev data')
    
    # Training options
    parser.add_argument('--output_dir', type=str, default="outputs", 
                        help='Directory to save outputs')
    parser.add_argument('--checkpoint', type=str, default="", 
                        help='Path to model checkpoint to resume training (optional)')
    parser.add_argument('--val_frequency', type=int, default=5, 
                        help='Validate every N epochs')
    parser.add_argument('--only_evaluate', action='store_true', 
                        help='Only run evaluation, no training')
    
    # Distributed training options
    parser.add_argument('--distributed', action='store_true',
                       help='Use distributed data parallel for multi-GPU training')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(),
                       help='Number of GPUs to use for distributed training')
    parser.add_argument('--dist_url', default='env://', type=str,
                       help='URL used to set up distributed training')
                       
    # Debug/Overfit mode options
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug/overfitting mode on a small subset of data')
    parser.add_argument('--debug_samples', type=int, default=debug_dataset_size,
                       help='Number of samples to use in debug/overfit mode')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                       help='Batch size for training')
    parser.add_argument('--debug_lr', type=float, default=debug_learning_rate,
                       help='Learning rate to use in debug/overfit mode')
                       
    # Model architecture options
    parser.add_argument('--use_deformable', type=bool, default=use_deformable,
                       help='Use deformable attention in the decoder')
    parser.add_argument('--n_points', type=int, default=n_points,
                       help='Number of sampling points in deformable attention')
                       
    # Loss and matcher options
    parser.add_argument('--set_cost_class', type=float, default=set_cost_class,
                       help='Class cost coefficient for Hungarian matching')
    parser.add_argument('--set_cost_bbox', type=float, default=set_cost_bbox,
                       help='L1 box cost coefficient for Hungarian matching')
    parser.add_argument('--set_cost_giou', type=float, default=set_cost_giou,
                       help='GIoU cost coefficient for Hungarian matching')
    parser.add_argument('--focal_alpha', type=float, default=focal_alpha,
                       help='Alpha parameter for focal loss')
    parser.add_argument('--focal_gamma', type=float, default=focal_gamma,
                       help='Gamma parameter for focal loss')
    parser.add_argument('--device', type=str, default="cuda",
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--num_classes', type=int, default=num_classes,
                       help='Number of classes for detection')
    
    args = parser.parse_args()
    
    # For non-distributed mode, run normally
    if not args.distributed:
        # Download dataset first if requested (for non-distributed mode)
        if args.download_train_data or args.download_val_data or args.download_test_data:
            args = download_coco_dataset(args)
        
        # Then validate data paths after potential download
        if not args.only_evaluate and (not args.train_images or not args.train_annotations):
            print("Error: Training images and annotations are required for training.")
            print("       Use --download_train_data to download COCO training data")
            print("       or provide --train_images and --train_annotations paths.")
            return
            
        if args.only_evaluate and not (args.val_images and args.val_annotations) and not args.testdev_images:
            print("Error: Validation or test-dev images are required for evaluation.")
            print("       Use --download_val_data to download COCO validation data")
            print("       or provide --val_images and --val_annotations paths.")
            print("       Alternatively, use --download_test_data to download test-dev data")
            print("       or provide --testdev_images path.")
            return
        
        # Run main worker function with rank 0
        main_worker(0, 1, args)
    else:
        # For distributed mode, spawn multiple processes
        # Make sure world_size is set correctly
        if args.world_size > torch.cuda.device_count():
            print(f"Warning: Requested {args.world_size} GPUs but only {torch.cuda.device_count()} are available.")
            args.world_size = torch.cuda.device_count()
        
        if args.world_size < 1:
            print("Error: No GPUs available for distributed training.")
            return
            
        print(f"Starting distributed training with {args.world_size} GPUs")
        
        # Spawn processes for each GPU
        mp.spawn(
            main_worker,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )

if __name__ == "__main__":
    main()