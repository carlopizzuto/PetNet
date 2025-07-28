import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Tuple, Optional
import json


class PetDataset(Dataset):
    """
    Generic dataset class for pet classification.
    Supports folder-based organization where each subfolder is a class.
    """
    
    def __init__(self, data_dir: str, transform=None, class_to_idx: Optional[Dict] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Build class to index mapping
        if class_to_idx is None:
            self.class_to_idx = self._build_class_to_idx()
        else:
            self.class_to_idx = class_to_idx
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Load all images and labels
        self._load_data()
    
    def _build_class_to_idx(self) -> Dict[str, int]:
        """Build mapping from class names to indices"""
        classes = sorted([d for d in os.listdir(self.data_dir) 
                         if os.path.isdir(os.path.join(self.data_dir, d))])
        return {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    def _load_data(self):
        """Load all image paths and corresponding labels"""
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_num_classes(self):
        return len(self.class_to_idx)
    
    def save_class_mapping(self, path: str):
        """Save class to index mapping for later use"""
        with open(path, 'w') as f:
            json.dump(self.class_to_idx, f, indent=2)
    
    @classmethod
    def load_class_mapping(cls, path: str) -> Dict[str, int]:
        """Load class to index mapping"""
        with open(path, 'r') as f:
            return json.load(f)


def get_transforms(image_size: int = 224, is_training: bool = True):
    """
    Get image transformations for training or validation.
    ViT expects 224x224 images by default.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(train_dir: str, val_dir: str, batch_size: int = 32, 
                      num_workers: int = 4, image_size: int = 224,
                      pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create training and validation dataloaders.
    Returns train_loader, val_loader, and class_to_idx mapping.
    """
    # Create datasets
    train_dataset = PetDataset(
        data_dir=train_dir,
        transform=get_transforms(image_size, is_training=True)
    )
    
    val_dataset = PetDataset(
        data_dir=val_dir,
        transform=get_transforms(image_size, is_training=False),
        class_to_idx=train_dataset.class_to_idx  # Use same mapping as training
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, train_dataset.class_to_idx