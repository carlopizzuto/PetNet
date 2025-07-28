#!/usr/bin/env python3
"""
Script to download and organize the Stanford Dogs dataset for PetNet training.
This script downloads the Stanford Dogs dataset and organizes it into train/val splits.
"""

import os
import sys
import shutil
import tarfile
import urllib.request
from pathlib import Path
import random
from collections import defaultdict

def download_file(url, filename):
    """Download a file with progress bar"""
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            sys.stdout.write(f'\rDownloading {filename}: {percent:.1f}%')
            sys.stdout.flush()
    
    urllib.request.urlretrieve(url, filename, progress_hook)
    print()  # New line after progress

def extract_tar(tar_path, extract_to):
    """Extract tar file"""
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(extract_to)

def organize_dataset(images_dir, lists_dir, output_dir, train_ratio=0.8):
    """
    Organize Stanford Dogs dataset into train/val splits.
    
    Args:
        images_dir: Path to extracted Images directory
        lists_dir: Path to extracted lists directory  
        output_dir: Where to create organized dataset
        train_ratio: Fraction of data to use for training
    """
    
    # Read train/test file lists (we'll use these as guidance but create our own split)
    train_list_file = os.path.join(lists_dir, 'train_list.mat')
    test_list_file = os.path.join(lists_dir, 'test_list.mat')
    
    # Since .mat files need scipy, we'll organize based on directory structure instead
    print("Organizing dataset based on directory structure...")
    
    # Get all breed directories
    breed_dirs = [d for d in os.listdir(images_dir) 
                  if os.path.isdir(os.path.join(images_dir, d)) and d.startswith('n')]
    
    print(f"Found {len(breed_dirs)} dog breeds")
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    total_images = 0
    breed_mapping = {}
    
    for breed_code in breed_dirs:
        breed_path = os.path.join(images_dir, breed_code)
        
        # Extract readable breed name (remove n########- prefix)
        breed_name = breed_code.split('-', 1)[1] if '-' in breed_code else breed_code
        breed_name = breed_name.replace('_', ' ').title()
        breed_mapping[breed_code] = breed_name
        
        # Get all images in this breed directory
        images = [f for f in os.listdir(breed_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            print(f"Warning: No images found for breed {breed_name}")
            continue
            
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Create breed directories in train/val
        train_breed_dir = os.path.join(train_dir, breed_name)
        val_breed_dir = os.path.join(val_dir, breed_name)
        os.makedirs(train_breed_dir, exist_ok=True)
        os.makedirs(val_breed_dir, exist_ok=True)
        
        # Copy images
        for img in train_images:
            src = os.path.join(breed_path, img)
            dst = os.path.join(train_breed_dir, img)
            shutil.copy2(src, dst)
            
        for img in val_images:
            src = os.path.join(breed_path, img)
            dst = os.path.join(val_breed_dir, img)
            shutil.copy2(src, dst)
        
        total_images += len(images)
        print(f"Processed {breed_name}: {len(train_images)} train, {len(val_images)} val")
    
    print(f"\nDataset organization complete!")
    print(f"Total images: {total_images}")
    print(f"Total breeds: {len(breed_dirs)}")
    print(f"Train directory: {train_dir}")
    print(f"Validation directory: {val_dir}")
    
    # Save breed mapping
    import json
    with open(os.path.join(output_dir, 'breed_mapping.json'), 'w') as f:
        json.dump(breed_mapping, f, indent=2)

def main():
    # URLs for Stanford Dogs dataset
    IMAGES_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    LISTS_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"
    
    # Paths
    data_dir = "data/dogs"
    temp_dir = "temp_download"
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        print("Downloading Stanford Dogs Dataset...")
        print("This may take a while (dataset is ~750MB)")
        
        # Download files
        images_tar = os.path.join(temp_dir, "images.tar")
        lists_tar = os.path.join(temp_dir, "lists.tar")
        
        if not os.path.exists(images_tar):
            download_file(IMAGES_URL, images_tar)
        else:
            print(f"Images archive already exists: {images_tar}")
            
        if not os.path.exists(lists_tar):
            download_file(LISTS_URL, lists_tar)
        else:
            print(f"Lists archive already exists: {lists_tar}")
        
        # Extract files
        extract_tar(images_tar, temp_dir)
        extract_tar(lists_tar, temp_dir)
        
        # Organize dataset
        images_dir = os.path.join(temp_dir, "Images")
        lists_dir = os.path.join(temp_dir, "lists")
        
        organize_dataset(images_dir, lists_dir, data_dir)
        
        print("\n✅ Stanford Dogs dataset setup complete!")
        print(f"Dataset organized in: {data_dir}")
        print("\nYou can now train your model using:")
        print("python src/train.py --train_dir data/dogs/train --val_dir data/dogs/val --output_dir models/dog_classifier")
        
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_dir):
            print(f"\nCleaning up temporary files...")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Set random seed for reproducible splits
    random.seed(42)
    main() 