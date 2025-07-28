import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def get_device(device_name: str = None) -> torch.device:
    """
    Get the best available device for training/inference.
    
    Args:
        device_name: Specific device name ('cuda', 'mps', 'cpu') or None for auto-detection
        
    Returns:
        torch.device object
    """
    if device_name:
        return torch.device(device_name)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor, return_count: bool = False) -> Union[float, int]:
    """
    Calculate accuracy from model outputs and true labels.
    
    Args:
        outputs: Model predictions (logits)
        labels: True labels
        return_count: If True, return count of correct predictions, else percentage
    
    Returns:
        Accuracy as percentage or count of correct predictions
    """
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    
    if return_count:
        return correct
    else:
        return 100.0 * correct / labels.size(0)


def save_training_plot(train_losses: List[float], val_losses: List[float], 
                      train_accs: List[float], val_accs: List[float], save_path: str):
    """
    Save training history plots (loss and accuracy).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                  device: torch.device, class_names: List[str]):
    """
    Comprehensive model evaluation with metrics and confusion matrix.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate overall accuracy
    accuracy = 100.0 * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    
    # Generate classification report
    report = classification_report(all_labels, all_predictions, 
                                 target_names=class_names, output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return accuracy, report, cm


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str):
    """
    Plot and save confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def predict_single_image(model: torch.nn.Module, image_path: str, 
                        transform, class_names: List[str], device: torch.device):
    """
    Predict class for a single image.
    """
    from PIL import Image
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item()
    
    return predicted_class, confidence_score


def print_model_summary(model: torch.nn.Module, input_size: tuple = (3, 224, 224)):
    """
    Print model summary including parameter count.
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Input size: {input_size}")
    print("=" * 50)


def save_evaluation_report(accuracy: float, classification_report: dict, 
                          save_path: str, model_name: str):
    """
    Save evaluation results to a text file.
    """
    with open(save_path, 'w') as f:
        f.write(f"EVALUATION REPORT - {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")
        
        f.write("Per-Class Metrics:\n")
        f.write("-" * 30 + "\n")
        
        for class_name, metrics in classification_report.items():
            if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {metrics['precision']:.3f}\n")
                f.write(f"  Recall: {metrics['recall']:.3f}\n")
                f.write(f"  F1-Score: {metrics['f1-score']:.3f}\n")
                f.write(f"  Support: {metrics['support']}\n\n")
        
        f.write("Summary Metrics:\n")
        f.write("-" * 20 + "\n")
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in classification_report:
                metrics = classification_report[avg_type]
                f.write(f"{avg_type}:\n")
                f.write(f"  Precision: {metrics['precision']:.3f}\n")
                f.write(f"  Recall: {metrics['recall']:.3f}\n")
                f.write(f"  F1-Score: {metrics['f1-score']:.3f}\n\n")