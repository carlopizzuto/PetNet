import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from typing import Dict, Any


class PetClassifier(nn.Module):
    """
    Base Vision Transformer classifier for pet classification tasks.
    Can be configured for different numbers of classes (dog breeds, cat breeds, pet types).
    """
    
    def __init__(self, num_classes: int, model_name: str = "google/vit-base-patch16-224"):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pre-trained ViT
        self.vit = ViTModel.from_pretrained(model_name)
        
        # Custom classification head
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, pixel_values):
        # Get ViT outputs
        outputs = self.vit(pixel_values=pixel_values)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0]  # Shape: (batch_size, hidden_size)
        
        # Apply dropout and classification
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits
    
    def save_model(self, path: str):
        """Save model state dict and config"""
        torch.save({
            'state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'model_name': self.model_name
        }, path)
    
    @classmethod
    def load_model(cls, path: str):
        """Load model from saved state dict"""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(
            num_classes=checkpoint['num_classes'],
            model_name=checkpoint['model_name']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    def freeze_backbone(self):
        """Freeze ViT backbone for fine-tuning only the classifier head"""
        for param in self.vit.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze ViT backbone for full fine-tuning"""
        for param in self.vit.parameters():
            param.requires_grad = True