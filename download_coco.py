#!/usr/bin/env python
# download_coco.py
"""
Downloads the COCO dataset and extracts it to a specified directory.
After downloading, runs the training script with the appropriate paths.
"""

import os
import argparse
import subprocess
import sys
import shutil
from tqdm import tqdm
import urllib.request
import zipfile
import tarfile

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

def extract_archive(archive_path, extract_dir, desc=None):
    """
    Extract an archive file.
    
    Args:
        archive_path: Path to the archive file
        extract_dir: Directory to extract to
        desc: Description for the progress bar
    """
    if not desc:
        desc = f"Extracting {os.path.basename(archive_path)}"
    
    print(f"{desc}...")
    
    # Create directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract based on file extension
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            total_files = len(zip_ref.namelist())
            with tqdm(total=total_files, desc=desc) as pbar:
                for file in zip_ref.namelist():
                    zip_ref.extract(file, extract_dir)
                    pbar.update(1)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            total_files = len(tar_ref.getmembers())
            with tqdm(total=total_files, desc=desc) as pbar:
                for member in tar_ref.getmembers():
                    tar_ref.extract(member, extract_dir)
                    pbar.update(1)
    else:
        print(f"Unsupported archive format: {archive_path}")

def download_coco(args):
    """
    Download the COCO dataset.
    
    Args:
        args: Command line arguments
    
    Returns:
        dict: Paths to the dataset directories and files
    """
    # Create base directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    # URLs for COCO dataset
    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'test_images': 'http://images.cocodataset.org/zips/test2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    
    # Paths for downloaded files
    downloads = {
        'train_images': os.path.join(args.data_dir, 'downloads', 'train2017.zip'),
        'val_images': os.path.join(args.data_dir, 'downloads', 'val2017.zip'),
        'test_images': os.path.join(args.data_dir, 'downloads', 'test2017.zip'),
        'annotations': os.path.join(args.data_dir, 'downloads', 'annotations_trainval2017.zip')
    }
    
    # Download files based on command line arguments
    download_types = []
    if args.download_train:
        download_types.extend(['train_images', 'annotations'])
    if args.download_val:
        download_types.extend(['val_images', 'annotations'])
    if args.download_test:
        download_types.append('test_images')
    
    # Remove duplicates
    download_types = list(set(download_types))
    
    # Download files
    for download_type in download_types:
        download_file(urls[download_type], downloads[download_type], f"Downloading {download_type}")
    
    # Extract files
    for download_type in download_types:
        extract_archive(downloads[download_type], args.data_dir, f"Extracting {download_type}")
    
    # Return paths to dataset directories and files
    return {
        'train_images': os.path.join(args.data_dir, 'train2017'),
        'val_images': os.path.join(args.data_dir, 'val2017'),
        'test_images': os.path.join(args.data_dir, 'test2017'),
        'annotations_dir': os.path.join(args.data_dir, 'annotations'),
        'train_annotations': os.path.join(args.data_dir, 'annotations', 'instances_train2017.json'),
        'val_annotations': os.path.join(args.data_dir, 'annotations', 'instances_val2017.json')
    }

def run_training(args, dataset_paths):
    """
    Run the training script with the downloaded dataset.
    
    Args:
        args: Command line arguments
        dataset_paths: Paths to the dataset directories and files
    """
    if not args.train:
        return
    
    # Construct command for training
    cmd = [
        'python', '-m', 'dino_detector.train',
        '--output_dir', args.output_dir
    ]
    
    # Add dataset paths
    if args.download_train:
        cmd.extend([
            '--train_images', dataset_paths['train_images'],
            '--train_annotations', dataset_paths['train_annotations']
        ])
    
    if args.download_val:
        cmd.extend([
            '--val_images', dataset_paths['val_images'],
            '--val_annotations', dataset_paths['val_annotations']
        ])
    
    if args.download_test:
        cmd.extend([
            '--testdev_images', dataset_paths['test_images']
        ])
    
    # Add additional training arguments
    if args.checkpoint:
        cmd.extend(['--checkpoint', args.checkpoint])
    
    if args.only_evaluate:
        cmd.append('--only_evaluate')
    
    # Print command
    print("Running training command:")
    print(' '.join(cmd))
    
    # Run command
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description='Download COCO dataset and run training')
    parser.add_argument('--data_dir', type=str, default='coco_data',
                        help='Directory to store the COCO dataset')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save training outputs')
    parser.add_argument('--download_train', action='store_true',
                        help='Download training data')
    parser.add_argument('--download_val', action='store_true',
                        help='Download validation data')
    parser.add_argument('--download_test', action='store_true',
                        help='Download test data')
    parser.add_argument('--train', action='store_true',
                        help='Run training after downloading')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to checkpoint for resuming training')
    parser.add_argument('--only_evaluate', action='store_true',
                        help='Only run evaluation, no training')
    
    args = parser.parse_args()
    
    # Default to downloading training data if nothing specified
    if not (args.download_train or args.download_val or args.download_test):
        args.download_train = True
    
    # Download dataset
    dataset_paths = download_coco(args)
    
    # Run training
    run_training(args, dataset_paths)

if __name__ == "__main__":
    main()