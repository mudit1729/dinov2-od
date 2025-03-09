# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import os
import argparse
import json
import sys
import urllib.request
import zipfile
from tqdm import tqdm
import gc
import random
from dino_detector.validate import print_tensors_by_size, clear_memory, memory_stats
from dino_detector.models.detector import DINOv2ObjectDetector
from dino_detector.dataset import COCODataset, COCOTestDataset, collate_fn
from dino_detector.utils import (
    evaluate_coco, compute_coco_metrics, setup_logger, 
    setup_tensorboard, log_metrics, log_images
)
from dino_detector.matching import HungarianMatcher, build_matcher
from dino_detector.losses import SetCriterion, build_criterion
from dino_detector.config import (
    batch_size, num_epochs, learning_rate, weight_decay, num_workers,
    distributed_backend, find_unused_parameters, gradient_accumulation_steps,
    gradient_clip_val,
    set_cost_class, set_cost_bbox, set_cost_giou, focal_alpha, focal_gamma,
    loss_weights, num_classes,
    debug_mode, debug_dataset_size, debug_epochs, debug_learning_rate,
    use_deformable, n_points, dim_feedforward, dropout,
    dino_model_name
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

# Create dataset subset functions
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
    # Subset and random are now imported at the top of the file
    
    # Preserve COCO annotation path if it exists
    coco_path = None
    if hasattr(dataset, 'coco_path'):
        coco_path = dataset.coco_path
    
    # Make sure we don't try to get more samples than exist in the dataset
    num_samples = min(num_samples, len(dataset))
    
    # Choose random indices without replacement
    random.seed(42)  # For reproducibility
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Create a subset with those indices
    subset = Subset(dataset, indices)
    
    # Add coco_path attribute to the subset if it exists in the original dataset
    if coco_path:
        subset.coco_path = coco_path
        print(f"Preserved COCO annotation path in debug subset: {coco_path}")
        
    return subset

def load_coco_mini_indices(indices_file):
    """
    Load saved COCO mini dataset indices from a JSON file.
    
    Args:
        indices_file: Path to the JSON file with saved indices
        
    Returns:
        list: List of dataset indices 
        dict: Additional metadata from the file
    """
    import json
    
    try:
        with open(indices_file, 'r') as f:
            data = json.load(f)
            
        indices = data.get('indices', [])
        if not indices:
            print(f"Warning: No indices found in {indices_file}")
            return [], data
            
        print(f"Loaded {len(indices)} indices from {indices_file}")
        return indices, data
    except Exception as e:
        print(f"Error loading indices file: {e}")
        return [], {}

def create_coco_mini(dataset, mini_size="1k", random_seed=42, save_indices=True, output_dir="outputs", indices_file=None):
    """
    Create a COCO mini dataset with a standardized size.
    
    Args:
        dataset: The original COCO dataset
        mini_size: Size of the mini dataset: "1k", "5k", or "10k"
        random_seed: Seed for reproducibility
        save_indices: Whether to save the indices for reproducibility
        output_dir: Directory to save indices file
        indices_file: Optional path to a file with predefined indices
        
    Returns:
        A subset of the original dataset
    """
    # Subset, random, json, and os are now imported at the top of the file
    
    # Preserve COCO annotation path if it exists
    coco_path = None
    if hasattr(dataset, 'coco_path'):
        coco_path = dataset.coco_path
    
    # If indices file is provided, load indices from it
    if indices_file:
        loaded_indices, metadata = load_coco_mini_indices(indices_file)
        if loaded_indices:
            # Verify indices are valid (within dataset size)
            valid_indices = [idx for idx in loaded_indices if idx < len(dataset)]
            
            if len(valid_indices) != len(loaded_indices):
                print(f"Warning: {len(loaded_indices) - len(valid_indices)} indices were out of range and skipped")
                
            if not valid_indices:
                print(f"Error: No valid indices found in {indices_file}. Falling back to random selection.")
            else:
                mini_size = metadata.get('mini_size', str(len(valid_indices)))
                random_seed = metadata.get('random_seed', random_seed)
                print(f"Using {len(valid_indices)} predefined indices from {indices_file}")
                indices = valid_indices
                subset = Subset(dataset, indices)
                
                # Add coco_path attribute to the subset if it exists in the original dataset
                if coco_path:
                    subset.coco_path = coco_path
                    
                return subset
    
    # If no indices file or it wasn't valid, generate new indices
    # Set the number of samples based on the specified size
    if mini_size == "1k":
        num_samples = 1000
    elif mini_size == "5k":
        num_samples = 5000
    elif mini_size == "10k":
        num_samples = 10000
    else:
        try:
            # Try parsing as a direct number
            num_samples = int(mini_size)
        except ValueError:
            print(f"Invalid mini_size: {mini_size}. Using 1000 samples.")
            num_samples = 1000
    
    # Make sure we don't try to get more samples than exist in the dataset
    num_samples = min(num_samples, len(dataset))
    print(f"Creating COCO mini dataset with {num_samples} samples")
    
    # Choose random indices without replacement
    random.seed(random_seed)  # For reproducibility
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Save the indices for reproducibility if requested
    if save_indices:
        os.makedirs(output_dir, exist_ok=True)
        indices_file = os.path.join(output_dir, f"coco_mini_{mini_size}_seed{random_seed}_indices.json")
        
        # Get image IDs if dataset has this information
        image_ids = []
        if hasattr(dataset, 'ids'):
            # For COCO dataset, get the actual image IDs
            image_ids = [dataset.ids[i] for i in indices]
        
        # Save both indices and image IDs
        with open(indices_file, 'w') as f:
            json.dump({
                'dataset_size': len(dataset),
                'mini_size': mini_size,
                'num_samples': num_samples,
                'random_seed': random_seed,
                'indices': indices,
                'image_ids': image_ids
            }, f, indent=2)
        print(f"Saved COCO mini dataset indices to: {indices_file}")
    
    # Create subset with those indices
    subset = Subset(dataset, indices)
    
    # Add coco_path attribute to the subset if it exists in the original dataset
    if coco_path:
        subset.coco_path = coco_path
        print(f"Preserved COCO annotation path: {coco_path}")
        
    return subset

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
    
    # Set up logger
    log_dir = args.log_dir if args.log_dir else args.output_dir
    if rank == 0:  # Only main process sets up logging
        logger = setup_logger(log_dir)
        
        # Set up TensorBoard
        # Create default experiment name based on configuration
        default_experiment_name = f"dinov2_detector_bs{args.batch_size}"
        if args.debug:
            default_experiment_name += f"_debug{args.debug_samples}"
        elif args.use_coco_mini:
            default_experiment_name += f"_cocomini{args.coco_mini_size}"
        else:
            default_experiment_name += "_full"
            
        if args.lightweight:
            default_experiment_name += "_lightweight"
            
        writer = setup_tensorboard(
            log_dir, 
            experiment_name=args.experiment_name if args.experiment_name else default_experiment_name
        )
    else:
        logger = None
        writer = None
    
    # Download COCO dataset if requested (download first, validate after)
    if rank == 0 and (args.download_train_data or args.download_val_data or args.download_test_data):
        args = download_coco_dataset(args)
        if logger:
            logger.info(f"Downloaded COCO dataset to {args.data_dir}")
        
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
    
    # Log device information
    if rank == 0 and logger:
        logger.info(f"Using device: {device}")
        if device.type == 'cuda':
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(device)}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
    
    # Initialize detector model with potentially custom configuration
    print(f"Process {rank}: Initializing DINOv2 Object Detector...")
    
    # If lightweight flag is specified, override configs to minimize parameters
    if args.lightweight:
        print(f"Process {rank}: Using lightweight configuration for fewer parameters")
        # Use smallest DINOv2 model if not explicitly specified
        if args.dino_model == dino_model_name:  # If user didn't override
            args.dino_model = "facebook/dinov2-small"  # Use small as the default for lightweight
        
        # Create custom model arguments for lightweight version
        # Determine appropriate hidden_dim based on model variant
        if "small" in args.dino_model:
            base_hidden_dim = 384
            target_hidden_dim = 256  # Slightly reduce for small
        elif "base" in args.dino_model:
            base_hidden_dim = 768
            target_hidden_dim = 384  # Half for base
        elif "large" in args.dino_model:
            base_hidden_dim = 1024
            target_hidden_dim = 512  # Half for large
        elif "giant" in args.dino_model:
            base_hidden_dim = 1536
            target_hidden_dim = 768  # Half for giant
        else:
            base_hidden_dim = 768
            target_hidden_dim = 384  # Default
            
        custom_args = {
            "num_classes": args.num_classes,
            "dino_model_name": args.dino_model,
            "hidden_dim": target_hidden_dim,  
            "num_queries": 25,  # Reduced queries
            "num_decoder_layers": 2,  # Only 2 decoder layers
            "dim_feedforward": target_hidden_dim * 2,  # 2x hidden_dim
            "lora_r": 1,  # Minimum LoRA rank
            "nheads": 4  # Reduce number of attention heads
        }
        model = DINOv2ObjectDetector(**custom_args)
        
        if rank == 0:
            print(f"Lightweight model settings:")
            for k, v in custom_args.items():
                print(f"  {k}: {v}")
    else:
        # Just use the model with specified dinov2 variant
        model = DINOv2ObjectDetector(
            num_classes=args.num_classes,
            dino_model_name=args.dino_model
        )
    
    model = model.to(device)
    
    # Log model information
    if rank == 0 and logger:
        logger.info(f"Model initialized with {args.num_classes} classes")
        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        # Log the model info to TensorBoard
        if writer:
            # Skip adding model graph due to conversion warnings with deformable attention
            # This avoids the TracerWarnings from tensor-to-Python conversions
            logger.info(f"Skipping model graph visualization in TensorBoard due to tensor conversion in deformable attention")
    
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
    
    if args.checkpoint and os.path.exists(args.checkpoint) and not args.skip_checkpoint_load:
        print(f"Process {rank}: Loading checkpoint from {args.checkpoint}")
        
        # Handle the case when user is using a lightweight model with a regular model checkpoint
        if args.lightweight and not args.skip_checkpoint_load:
            print(f"Process {rank}: Note: You're loading a checkpoint into a lightweight model.")
            print(f"Process {rank}: Some parameters may not be compatible and will be initialized randomly.")
            print(f"Process {rank}: Use --skip_checkpoint_load to start with a fresh model if you experience issues.")
            
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Handle model architecture differences between checkpoint and current model
        current_model_is_lightweight = args.lightweight
        
        if args.distributed:
            # Load model state dict for DDP model (need to handle 'module.' prefix)
            state_dict = checkpoint['model_state_dict']
            if not any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
        else:
            # Remove 'module.' prefix if present for non-DDP model
            state_dict = checkpoint['model_state_dict']
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Get the current model state dict to compare shapes
        model_state_dict = model.state_dict()
        
        # Filter out parameters with mismatched shapes
        compatible_state_dict = {}
        incompatible_keys = []
        
        for k, v in state_dict.items():
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    compatible_state_dict[k] = v
                else:
                    incompatible_keys.append(k)
        
        # Report on compatibility
        if incompatible_keys and rank == 0:
            if current_model_is_lightweight:
                print(f"Process {rank}: You're loading a standard model checkpoint into a lightweight model configuration.")
                print(f"Process {rank}: {len(incompatible_keys)} parameters skipped due to shape mismatches.")
                print(f"Process {rank}: This is normal when loading a checkpoint into a different model configuration.")
                print(f"Process {rank}: Compatible parameters: {len(compatible_state_dict)}/{len(model_state_dict)}")
            else:
                print(f"Process {rank}: WARNING: {len(incompatible_keys)} parameters skipped due to shape mismatches.")
                print(f"Process {rank}: This may indicate an architecture mismatch between the checkpoint and current model.")
                print(f"Process {rank}: First incompatible key: {incompatible_keys[0]}")
        
        # Load the compatible parameters
        model.load_state_dict(compatible_state_dict, strict=False)
        print(f"Process {rank}: Loaded {len(compatible_state_dict)} compatible parameters from checkpoint")
            
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
            
            # Create a mini test set if requested
            if args.test_mini:
                original_size = len(test_dataset)
                from torch.utils.data import Subset
                import random
                
                # Create small subset for quick evaluation
                test_mini_size = min(args.test_mini_size, len(test_dataset))
                random.seed(args.coco_mini_seed)  # Use same seed as COCO mini for consistency
                test_indices = random.sample(range(len(test_dataset)), test_mini_size)
                
                # Save indices for reproducibility (only from rank 0)
                if rank == 0:
                    # os and json are already imported at the top of the file
                    # No need to import them here
                    os.makedirs(args.output_dir, exist_ok=True)
                    indices_file = os.path.join(args.output_dir, f"test_mini_{test_mini_size}_seed{args.coco_mini_seed}_indices.json")
                    with open(indices_file, 'w') as f:
                        json.dump({
                            'dataset_size': len(test_dataset),
                            'mini_size': test_mini_size,
                            'random_seed': args.coco_mini_seed,
                            'indices': test_indices
                        }, f, indent=2)
                    print(f"Process {rank}: Saved test mini dataset indices to: {indices_file}")
                
                # Create and use subset
                test_dataset = Subset(test_dataset, test_indices)
                print(f"Process {rank}: TEST MINI MODE - Using {len(test_dataset)} test samples out of {original_size}")
            
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
            
            # Memory tracking before evaluation
            if args.memory_monitor and torch.cuda.is_available() and device.type == 'cuda':
                print(f"Before Evaluation - CUDA Memory: {torch.cuda.memory_allocated(device)/1024**2:.2f}MB (allocated), "
                      f"{torch.cuda.max_memory_allocated(device)/1024**2:.2f}MB (max allocated)")
                torch.cuda.empty_cache()  # Try to free some memory
                print(f"After Empty Cache - CUDA Memory: {torch.cuda.memory_allocated(device)/1024**2:.2f}MB (allocated)")
            elif torch.cuda.is_available() and device.type == 'cuda':
                # Always do basic cleanup even without monitoring
                torch.cuda.empty_cache()
            
            # Generate predictions for test-dev
            test_results_file = os.path.join(args.output_dir, f"testdev_predictions_rank{rank}.json")
            model.eval()
            test_results = evaluate_coco(model, test_dataloader, device, test_results_file)
            print(f"Process {rank}: Test-dev predictions saved to {test_results_file}")
        
        # Validation set evaluation
        if os.path.exists(args.val_images) and os.path.exists(args.val_annotations):
            val_dataset = COCODataset(args.val_images, args.val_annotations, transform=transform)
            
            # Explicitly store the annotation path
            val_dataset.coco_path = args.val_annotations
            
            # Create a COCO mini dataset for validation if requested
            original_val_size = len(val_dataset)
            if args.use_coco_mini:
                # Only save indices from rank 0 process to avoid file conflicts
                save_indices = (rank == 0)
                indices_file = args.coco_mini_indices_file if args.coco_mini_indices_file else None
                val_dataset = create_coco_mini(
                    val_dataset, 
                    args.coco_mini_size, 
                    args.coco_mini_seed,
                    save_indices=save_indices,
                    output_dir=args.output_dir,
                    indices_file=indices_file
                )
                # Double-check the coco_path is preserved
                if not hasattr(val_dataset, 'coco_path'):
                    val_dataset.coco_path = args.val_annotations
                    print(f"Process {rank}: Added COCO annotation path to subset")
                    
                print(f"Process {rank}: COCO MINI MODE - Using {len(val_dataset)} validation samples out of {original_val_size}")
            # In debug mode, use a small subset for validation
            elif args.debug:
                val_dataset = create_debug_subset(val_dataset, args.debug_samples * 2)  # Increase validation samples
                # Double-check the coco_path is preserved
                if not hasattr(val_dataset, 'coco_path'):
                    val_dataset.coco_path = args.val_annotations
                    print(f"Process {rank}: Added COCO annotation path to debug subset")
                    
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
    # Create a COCO mini dataset if requested (but not in debug mode)
    elif args.use_coco_mini:
        original_size = len(train_dataset)
        # Only save indices from rank 0 process to avoid file conflicts
        save_indices = (rank == 0)
        indices_file = args.coco_mini_indices_file if args.coco_mini_indices_file else None
        train_dataset = create_coco_mini(
            train_dataset, 
            args.coco_mini_size, 
            args.coco_mini_seed,
            save_indices=save_indices,
            output_dir=args.output_dir,
            indices_file=indices_file
        )
        print(f"Process {rank}: COCO MINI MODE - Using {len(train_dataset)} training samples out of {original_size}")
    
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
        # Create a COCO mini dataset for validation if requested
        elif args.use_coco_mini:
            original_val_size = len(val_dataset)
            # Use twice the samples for validation to get better metrics
            val_samples = args.coco_mini_size
            if args.coco_mini_size == "1k":
                val_samples = "2k"  # 2000 samples
            elif args.coco_mini_size == "5k":
                val_samples = "5k"  # Keep same size
            elif args.coco_mini_size == "10k":
                val_samples = "5k"  # Half the size for validation
                
            # Only save indices from rank 0 process to avoid file conflicts
            save_indices = (rank == 0)
            indices_file = args.coco_mini_indices_file if args.coco_mini_indices_file else None
            val_dataset = create_coco_mini(
                val_dataset, 
                val_samples, 
                args.coco_mini_seed,
                save_indices=save_indices,
                output_dir=args.output_dir,
                indices_file=indices_file
            )
            print(f"Process {rank}: COCO MINI MODE - Using {len(val_dataset)} validation samples out of {original_val_size}")
        
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
    if args.checkpoint and os.path.exists(args.checkpoint) and not args.only_evaluate and not args.skip_checkpoint_load:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            try:
                # Try to load optimizer state
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Process {rank}: Loaded optimizer state from checkpoint")
            except ValueError as e:
                # Handle case where optimizer parameters don't match
                if args.lightweight:
                    print(f"Process {rank}: Cannot load optimizer state due to model architecture differences.")
                    print(f"Process {rank}: This is normal when using a lightweight model with a standard checkpoint.")
                    print(f"Process {rank}: Training will continue with a freshly initialized optimizer.")
                else:
                    print(f"Process {rank}: WARNING: Failed to load optimizer state: {e}")
                    print(f"Process {rank}: This may indicate an architecture mismatch between the checkpoint and current model.")

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
        
        # Log training start
        if rank == 0 and logger:
            logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        
        # Memory tracking at the start of epoch
        if args.memory_monitor and torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            memory_info = f"Epoch {epoch+1} Start - CUDA Memory: {torch.cuda.memory_allocated(device)/1024**2:.2f}MB (allocated), " \
                          f"{torch.cuda.max_memory_allocated(device)/1024**2:.2f}MB (max allocated)"
            print(memory_info)
            if rank == 0 and logger:
                logger.info(memory_info)
        
        if rank == 0:  # Only rank 0 shows progress bar
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            iterator = pbar
        else:
            iterator = train_dataloader
        
        # Initialize batch metrics for TensorBoard
        global_step = epoch * len(train_dataloader)
        
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
            
            # Debug memory before loss calculation
            if args.memory_monitor and batch_idx % 5 == 0 and torch.cuda.is_available() and device.type == 'cuda':
                memory_info = f"Before loss - {memory_stats(device)}"
                print(memory_info)
                if rank == 0 and logger and batch_idx % 20 == 0:
                    logger.debug(memory_info)
            
            # Compute loss using criterion with Hungarian matching
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())
            
            # Debug memory after loss calculation
            if args.memory_monitor and batch_idx % 5 == 0 and torch.cuda.is_available() and device.type == 'cuda':
                memory_info = f"After loss - {memory_stats(device)}"
                print(memory_info)
                if rank == 0 and logger and batch_idx % 20 == 0:
                    logger.debug(memory_info)
            
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
                
                # Clear memory every few iterations
                if args.memory_monitor and batch_idx % 10 == 0 and torch.cuda.is_available() and device.type == 'cuda':
                    # Try to clean up memory
                    clear_memory(model)
                    memory_info = f"After memory cleanup - {memory_stats(device)}"
                    print(memory_info)
                    if rank == 0 and logger and batch_idx % 20 == 0:
                        logger.debug(memory_info)
                    
                    # If memory usage is excessive, print tensor sizes
                    if torch.cuda.memory_allocated(device) > 4 * 1024 * 1024 * 1024:  # 4GB
                        print_tensors_by_size()
                elif batch_idx % 20 == 0 and torch.cuda.is_available() and device.type == 'cuda':
                    # Always do basic cleanup even without monitoring
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Update statistics
            running_loss += loss.item()
            
            # Log to TensorBoard
            if rank == 0 and writer and batch_idx % args.log_frequency == 0:
                # Log losses
                log_metrics(writer, {f"loss/{k}": v.item() for k, v in loss_dict.items()}, global_step, prefix="train/")
                log_metrics(writer, {"loss/total": loss.item() * gradient_accumulation_steps}, global_step, prefix="train/")
                
                # Log learning rate
                log_metrics(writer, {"lr": optimizer.param_groups[0]['lr']}, global_step)
                
                # Log memory usage if available
                if torch.cuda.is_available() and device.type == 'cuda':
                    log_metrics(writer, {
                        "memory/allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
                        "memory/reserved_mb": torch.cuda.memory_reserved(device) / 1024**2
                    }, global_step)
                
                # Log sample images with bounding boxes periodically
                if args.log_images and batch_idx % args.log_images_frequency == 0:
                    log_images(writer, images, targets, outputs, global_step, tag="train/images")
            
            # Track memory after each batch
            if args.memory_monitor and batch_idx % 5 == 0 and torch.cuda.is_available() and device.type == 'cuda':
                memory_info = f"Batch {batch_idx} - CUDA Memory: {torch.cuda.memory_allocated(device)/1024**2:.2f}MB (allocated), " \
                             f"{torch.cuda.max_memory_allocated(device)/1024**2:.2f}MB (max allocated)"
                print(memory_info)
                if rank == 0 and logger and batch_idx % 20 == 0:
                    logger.debug(memory_info)
            
            # Update progress bar
            if rank == 0:
                # Display individual loss components
                loss_str = f"total: {loss.item():.3f} "
                loss_str += " ".join(f"{k}: {v.item():.3f}" for k, v in loss_dict.items())
                pbar.set_postfix_str(loss_str)
            
            # Increment global step
            global_step += 1
                
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
            
            # Memory tracking at the end of epoch
            if args.memory_monitor and torch.cuda.is_available() and device.type == 'cuda':
                memory_info = f"Epoch {epoch+1} End - CUDA Memory: {torch.cuda.memory_allocated(device)/1024**2:.2f}MB (allocated), " \
                              f"{torch.cuda.max_memory_allocated(device)/1024**2:.2f}MB (max allocated)"
                print(memory_info)
                if logger:
                    logger.info(memory_info)
                    
                peak_memory_info = f"Epoch {epoch+1} Peak CUDA Memory: {torch.cuda.max_memory_allocated(device)/1024**2:.2f}MB"
                print(peak_memory_info)
                if logger:
                    logger.info(peak_memory_info)
            
            # Update metrics history
            metrics_history['epochs'].append(epoch + 1)
            metrics_history['train_loss'].append(epoch_loss)
            
            # Log epoch metrics to TensorBoard
            if writer:
                log_metrics(writer, {
                    "epoch/train_loss": epoch_loss,
                }, epoch + 1)  # Use epoch number as step for epoch-level metrics
                
                if torch.cuda.is_available() and device.type == 'cuda':
                    log_metrics(writer, {
                        "epoch/peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2
                    }, epoch + 1)
        
        # Validation phase
        if val_dataloader is not None and (epoch + 1) % val_freq == 0:
            model.eval()
            
            # Only rank 0 runs validation and reports metrics
            if rank == 0:
                if logger:
                    logger.info(f"Running validation for epoch {epoch+1}")
                
                # Memory tracking before validation
                if args.memory_monitor and torch.cuda.is_available() and device.type == 'cuda':
                    memory_info = f"Before Validation - CUDA Memory: {torch.cuda.memory_allocated(device)/1024**2:.2f}MB (allocated), " \
                                 f"{torch.cuda.max_memory_allocated(device)/1024**2:.2f}MB (max allocated)"
                    print(memory_info)
                    if logger:
                        logger.info(memory_info)
                        
                    torch.cuda.empty_cache()  # Try to free some memory
                    
                    after_clear_info = f"After Empty Cache - CUDA Memory: {torch.cuda.memory_allocated(device)/1024**2:.2f}MB (allocated)"
                    print(after_clear_info)
                    if logger:
                        logger.info(after_clear_info)
                elif torch.cuda.is_available() and device.type == 'cuda':
                    # Always do basic cleanup even without monitoring
                    torch.cuda.empty_cache()
                
                metrics = validate(model, val_dataloader, device, epoch + 1, args.output_dir)
                
                if metrics:
                    # Update validation metrics history
                    metrics_history['val_epochs'].append(epoch + 1)
                    metrics_history['val_ap'].append(metrics['AP'])
                    metrics_history['val_ap50'].append(metrics['AP50'])
                    metrics_history['val_ap75'].append(metrics['AP75'])
                    
                    # Log validation metrics
                    if logger:
                        logger.info(f"Epoch {epoch+1} Validation: "
                                   f"AP={metrics['AP']:.4f}, AP50={metrics['AP50']:.4f}, AP75={metrics['AP75']:.4f}")
                    
                    # Log validation metrics to TensorBoard
                    if writer:
                        log_metrics(writer, {
                            "val/AP": metrics['AP'],
                            "val/AP50": metrics['AP50'],
                            "val/AP75": metrics['AP75'],
                            "val/APs": metrics['APs'],
                            "val/APm": metrics['APm'],
                            "val/APl": metrics['APl']
                        }, epoch + 1)  # Use epoch number as step for epoch-level metrics
                
                # Plot metrics
                plot_metrics(metrics_history, args.output_dir)
                
                # Memory tracking after validation
                if args.memory_monitor and torch.cuda.is_available() and device.type == 'cuda':
                    memory_info = f"After Validation - CUDA Memory: {torch.cuda.memory_allocated(device)/1024**2:.2f}MB (allocated), " \
                                 f"{torch.cuda.max_memory_allocated(device)/1024**2:.2f}MB (max allocated)"
                    print(memory_info)
                    if logger:
                        logger.info(memory_info)
            
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
        if logger:
            logger.info(f"Training complete. Final model saved to {final_model_path}")
    
    # Final evaluation on test-dev if provided
    if rank == 0 and args.testdev_images:
        print(f"Evaluating final model on test-dev set: {args.testdev_images}")
        if logger:
            logger.info(f"Evaluating final model on test-dev set: {args.testdev_images}")
            
        test_dataset = COCOTestDataset(args.testdev_images, transform=transform)
        
        # Create a mini test set if requested
        if args.test_mini:
            original_size = len(test_dataset)
            # Subset is already imported at the top in main evaluation function
            # random is already used elsewhere, so it should also be imported at the top
            
            # Create small subset for quick evaluation
            test_mini_size = min(args.test_mini_size, len(test_dataset))
            random.seed(args.coco_mini_seed)  # Use same seed as COCO mini for consistency
            test_indices = random.sample(range(len(test_dataset)), test_mini_size)
            
            # Create and use subset
            test_dataset = Subset(test_dataset, test_indices)
            print(f"TEST MINI MODE - Using {len(test_dataset)} test samples out of {original_size}")
            if logger:
                logger.info(f"TEST MINI MODE - Using {len(test_dataset)} test samples out of {original_size}")
        
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
        if logger:
            logger.info(f"Test-dev predictions saved to {test_results_file}")
    
    # Close TensorBoard writer
    if rank == 0 and writer:
        writer.close()
        print(f"TensorBoard logs closed.")
        if logger:
            logger.info(f"TensorBoard logs closed.")
            for handler in logger.handlers:
                handler.close()
    
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
                       
    # COCO mini dataset options
    parser.add_argument('--use_coco_mini', action='store_true',
                       help='Use a smaller COCO mini dataset for faster training')
    parser.add_argument('--coco_mini_size', type=str, default="1k",
                       help='Size of COCO mini dataset: "1k", "5k", "10k", or custom number')
    parser.add_argument('--coco_mini_seed', type=int, default=42,
                       help='Random seed for COCO mini dataset creation')
    parser.add_argument('--coco_mini_indices_file', type=str, default="",
                       help='Path to a JSON file with saved mini dataset indices to use (overrides size and seed)')
    parser.add_argument('--test_mini', action='store_true',
                       help='Use only 30 test images for quick evaluation')
    parser.add_argument('--test_mini_size', type=int, default=30,
                       help='Number of test images to use when --test_mini is enabled')
                       
    # Model architecture options
    parser.add_argument('--use_deformable', type=bool, default=use_deformable,
                       help='Use deformable attention in the decoder')
    parser.add_argument('--n_points', type=int, default=n_points,
                       help='Number of sampling points in deformable attention')
    parser.add_argument('--dino_model', type=str, default=dino_model_name,
                       help='DINOv2 model variant (facebook/dinov2-tiny, facebook/dinov2-small, facebook/dinov2-base, facebook/dinov2-large)')
    parser.add_argument('--lightweight', action='store_true',
                       help='Use lightweight configuration with fewer parameters')
    parser.add_argument('--skip_checkpoint_load', action='store_true',
                       help='Skip loading checkpoint even if --checkpoint is specified (useful for lightweight models)')
                       
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
    parser.add_argument('--memory_monitor', action='store_true',
                       help='Enable detailed memory usage monitoring')
    
    # Logging options
    parser.add_argument('--log_dir', type=str, default=None,
                       help='Directory to save logs (defaults to output_dir)')
    parser.add_argument('--log_frequency', type=int, default=10,
                       help='Log metrics every N batches')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for the experiment (used in tensorboard logs)')
    parser.add_argument('--log_images', action='store_true',
                       help='Log images to tensorboard')
    parser.add_argument('--log_images_frequency', type=int, default=100,
                       help='Log images every N batches')
    
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