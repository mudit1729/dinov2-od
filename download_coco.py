#!/usr/bin/env python
# download_coco.py
"""
Downloads the COCO dataset and extracts it to a specified directory.
"""

import os
import argparse
from tqdm import tqdm
import urllib.request
import zipfile
import tarfile

# COCO dataset URLs
COCO_URLS = {
    'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
    'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
    'test_images': 'http://images.cocodataset.org/zips/test2017.zip',
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
}

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

def download_coco(data_dir, download_train=False, download_val=False, download_test=False):
    """
    Download the COCO dataset.
    
    Args:
        data_dir: Directory to store the dataset
        download_train: Whether to download training data
        download_val: Whether to download validation data
        download_test: Whether to download test data
    
    Returns:
        dict: Paths to the dataset directories and files
    """
    # Create base directory
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'downloads'), exist_ok=True)
    
    # Paths for downloaded files
    downloads = {
        'train_images': os.path.join(data_dir, 'downloads', 'train2017.zip'),
        'val_images': os.path.join(data_dir, 'downloads', 'val2017.zip'),
        'test_images': os.path.join(data_dir, 'downloads', 'test2017.zip'),
        'annotations': os.path.join(data_dir, 'downloads', 'annotations_trainval2017.zip')
    }
    
    # Download files based on command line arguments
    download_types = []
    if download_train:
        download_types.extend(['train_images', 'annotations'])
    if download_val:
        download_types.extend(['val_images', 'annotations'])
    if download_test:
        download_types.append('test_images')
    
    # Remove duplicates
    download_types = list(set(download_types))
    
    if not download_types:
        print("No data specified for download. Use --download_train, --download_val, or --download_test.")
        return None
    
    # Download files
    for download_type in download_types:
        download_file(COCO_URLS[download_type], downloads[download_type], f"Downloading {download_type}")
    
    # Extract files
    for download_type in download_types:
        extract_archive(downloads[download_type], data_dir, f"Extracting {download_type}")
    
    # Return paths to dataset directories and files
    return {
        'train_images': os.path.join(data_dir, 'train2017'),
        'val_images': os.path.join(data_dir, 'val2017'),
        'test_images': os.path.join(data_dir, 'test2017'),
        'annotations_dir': os.path.join(data_dir, 'annotations'),
        'train_annotations': os.path.join(data_dir, 'annotations', 'instances_train2017.json'),
        'val_annotations': os.path.join(data_dir, 'annotations', 'instances_val2017.json')
    }

def main():
    parser = argparse.ArgumentParser(description='Download COCO dataset')
    parser.add_argument('--data_dir', type=str, default='coco_data',
                        help='Directory to store the COCO dataset')
    parser.add_argument('--download_train', action='store_true',
                        help='Download training data')
    parser.add_argument('--download_val', action='store_true',
                        help='Download validation data')
    parser.add_argument('--download_test', action='store_true',
                        help='Download test data')
    
    args = parser.parse_args()
    
    # Default to downloading training data if nothing specified
    if not (args.download_train or args.download_val or args.download_test):
        args.download_train = True
    
    # Download dataset
    dataset_paths = download_coco(
        args.data_dir, 
        args.download_train, 
        args.download_val, 
        args.download_test
    )
    
    if dataset_paths:
        print("\nCOCO dataset downloaded successfully!")
        print("\nDataset paths:")
        for name, path in dataset_paths.items():
            if os.path.exists(path):
                print(f"- {name}: {path}")
        
        print("\nTo use this dataset for training:")
        if args.download_train:
            print(f"python -m dino_detector.train --train_images {dataset_paths['train_images']} --train_annotations {dataset_paths['train_annotations']}")
        
        if args.download_val:
            print(f"  --val_images {dataset_paths['val_images']} --val_annotations {dataset_paths['val_annotations']}")
        
        if args.download_test:
            print(f"  --testdev_images {dataset_paths['test_images']}")

if __name__ == "__main__":
    main()