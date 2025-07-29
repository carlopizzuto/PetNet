import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
from typing import Dict, Tuple
# Hugging Face helpers for better scheduling
from transformers import get_cosine_schedule_with_warmup

from model import PetClassifier
from data import create_dataloaders
from utils import calculate_accuracy, save_training_plot, get_device

# ------------------------
# Early stopping helper
# ------------------------


class EarlyStopping:
    """Simple early stopping on validation accuracy"""

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.counter = 0
        self.best_val_acc = 0.0

    def step(self, val_acc: float) -> bool:
        """Return True if training should stop"""
        if val_acc > self.best_val_acc + 1e-3:  # significant improvement
            self.best_val_acc = val_acc
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device, scheduler=None) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # Scheduler is stepped every batch (needed for warm-up + cosine)
        if scheduler is not None:
            scheduler.step()
        
        # Statistics
        total_loss += loss.item()
        total_correct += calculate_accuracy(outputs, labels, return_count=True)
        total_samples += labels.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.0 * total_correct / total_samples:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = 100.0 * total_correct / total_samples
    
    return avg_loss, avg_acc


def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, 
                  device: torch.device) -> Tuple[float, float]:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            total_correct += calculate_accuracy(outputs, labels, return_count=True)
            total_samples += labels.size(0)
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.0 * total_correct / total_samples:.2f}%'
            })
    
    avg_loss = total_loss / len(val_loader)
    avg_acc = 100.0 * total_correct / total_samples
    
    return avg_loss, avg_acc


def train_model(args):
    """Main training function"""
    # Set device with MPS support
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading data...")
    # Respect Colab warning: cap workers at 2 unless user overrides
    effective_workers = min(args.num_workers, 2)
    train_loader, val_loader, class_to_idx = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=effective_workers,
        pin_memory=device.type == 'cuda'
    )
    
    num_classes = len(class_to_idx)
    print(f"Found {num_classes} classes: {list(class_to_idx.keys())}")
    
    # Create model
    model = PetClassifier(num_classes=num_classes)
    model.to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Cosine schedule WITH linear warm-up (better for ViT as per HF article)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Training loop
    best_val_acc = 0.0
    early_stopper = EarlyStopping(patience=args.early_stopping_patience)
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Save metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Log current LR (scheduler may not expose get_last_lr before first step)
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.output_dir, 'best_model.pt')
            model.save_model(best_model_path)
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")

        # Early stopping check
        if early_stopper.step(val_acc):
            print(f"Early stopping triggered (no improvement for {args.early_stopping_patience} epochs)")
            break
    
    # Save final model and training plot
    final_model_path = os.path.join(args.output_dir, 'final_model.pt')
    model.save_model(final_model_path)
    
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    save_training_plot(train_losses, val_losses, train_accs, val_accs, plot_path)
    
    # Save class mapping
    class_mapping_path = os.path.join(args.output_dir, 'class_mapping.json')
    with open(class_mapping_path, 'w') as f:
        import json
        json.dump(class_to_idx, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Pet Classifier')
    
    # Data arguments
    parser.add_argument('--train_dir', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, required=True,
                       help='Path to validation data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save models and outputs')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--step_size', type=int, default=10,
                       help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Gamma for learning rate scheduler')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Proportion of total steps used for linear LR warm-up')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loader workers (default 2 for Colab)')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='Epochs to wait for improvement before early stopping')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/mps/cpu). If not specified, auto-detects best available.')

    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train_model(args)


if __name__ == '__main__':
    main()