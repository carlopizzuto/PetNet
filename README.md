# PetNet - Pet Breed Classification

Fine-tuning Google's Vision Transformer (ViT) to classify pet breeds and species for a pet community app.

## Project Goal

Create specialized models for accurate pet identification:
- **Dog Breed Classifier** - Identifies specific dog breeds
- **Cat Breed Classifier** - Identifies specific cat breeds  
- **Fish Breed Classifier** - Identifies specific fish species
- **Pet Type Classifier** - Identifies pet type (dog, cat, fish, turtle, etc.)

## Roadmap

### Phase 1: Dog Breed Classification ✨ *Current*
- Fine-tune ViT on Stanford Dog Dataset (120 breeds)
- Establish training pipeline and evaluation metrics

### Phase 2: Cat Breed Classification
- Fine-tune ViT on cat breeds dataset
- Compare performance with dog classifier

### Phase 3: Pet Type Classification  
- Train classifier to distinguish between pet types
- Use for initial routing to specialized models

### Phase 4: Additional Species
- Fish breed classifier
- Other pet types as needed

## Current Status

🚧 **Phase 1 in progress** - Setting up dog breed classification

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies (will create requirements.txt as we go)
pip install torch torchvision transformers datasets
```

## Quick Start

Coming soon - Phase 1 implementation in progress.

## Final Architecture

```
Input Image
    ↓
Pet Type Classifier → Dog/Cat/Fish/etc.
    ↓
Specialized Breed Classifier → Specific Breed
```

Each model will be independently trained and optimized for its specific task.
