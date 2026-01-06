# Reconocimiento de Caracteres - Directory Structure

Organized structure for character recognition and mask refinement neural networks.

## 📁 Directory Layout

```
Reconocimiento de Caracteres/
│
├── 📚 models/                               # Trained model weights (production only)
│   ├── segmentation/                       # Main character segmentation models
│   │   └── checkpoint_epoch_70.pth        # ⭐ Production model (used by NeuralPaint)
│   │
│   └── refinement/                         # Mask refinement models (anti-aliasing)
│       └── best_model.pth                 # ⭐ Best refinement model (used in pipeline)
│
├── 🔧 scripts/                              # Executable scripts
│   ├── training/                           # Model training
│   │   ├── train_segmentation.py          # Train main segmentation U-Net
│   │   ├── train_refinement.py            # Train mask refinement network
│   │   └── hpo.py                         # Hyperparameter optimization
│   │
│   ├── data_generation/                    # Dataset creation
│   │   ├── generate_dataset.py            # Generate synthetic character dataset
│   │   ├── generate_refinement_dataset.py # Generate refinement training pairs
│   │   ├── preprocess_binary.py           # Preprocess dataset to binary format
│   │   └── recognition_font_assets.py     # Font asset management
│   │
│   └── inference/                          # Testing and prediction
│       └── testing.py                     # Run inference on test images
│
├── 💾 datasets/                             # Training and validation data
│   ├── synthetic/                          # Main synthetic character dataset
│   │   ├── images/                        # Input images (rendered characters)
│   │   ├── masks/                         # Binary segmentation masks
│   │   ├── masks_ignore/                  # Ignore masks (partial characters)
│   │   ├── binary/                        # Preprocessed binary cache (.pt files)
│   │   └── metadata.jsonl                 # Sample metadata
│   │
│   ├── refinement/                         # Refinement network training data
│   │   ├── binary/                        # Input: binary masks (0/255)
│   │   ├── smooth/                        # Target: anti-aliased masks
│   │   └── metadata.jsonl                 # Refinement sample metadata
│   │
│   └── splits/                             # Train/validation splits
│       ├── hpo_train_files.txt            # HPO training file list
│       └── hpo_val_files.txt              # HPO validation file list
│
├── 📊 outputs/                              # Training results and visualizations
│   ├── segmentation/                       # Segmentation training outputs
│   │   ├── confusion_matrix_final.png
│   │   ├── loss_curve.png
│   │   └── val_iou_curve.png
│   │
│   ├── refinement/                         # Refinement training outputs
│   │   └── training_curve.png
│   │
│   └── test_predictions/                   # Inference results
│       └── (generated test outputs)
│
├── 🖼️ test_images/                          # Test images for inference
│   └── (various screenshot test images)
│
├── 📁 assets/                               # Project resources
│   └── fonts/                             # Font files for dataset generation
│       ├── downloads/                     # Downloaded fonts
│       └── extracted/                     # Extracted/processed fonts
│
├── 📄 README.md                             # Project documentation
├── 📄 STRUCTURE.md                          # This file
└── 📄 requirements.txt                      # Python dependencies
```

## 🎯 Key Files

### Production Models (Used by NeuralPaint)
- **`models/segmentation/checkpoint_epoch_70.pth`** - Main character segmentation model (1.8 MB)
- **`models/refinement/best_model.pth`** - Neural anti-aliasing refinement model (272 KB)

### Training Scripts
- **`scripts/training/train_segmentation.py`** - Train the main U-Net segmentation model
- **`scripts/training/train_refinement.py`** - Train the mask refinement network

### Dataset Generation
- **`scripts/data_generation/generate_dataset.py`** - Create synthetic character training data
- **`scripts/data_generation/generate_refinement_dataset.py`** - Create refinement training pairs

### Inference
- **`scripts/inference/testing.py`** - Run predictions on test images

## 🚀 Quick Start

### Generate Training Data
```bash
# Generate synthetic character dataset
python "scripts/data_generation/generate_dataset.py" --samples 1000

# Generate refinement training pairs
python "scripts/data_generation/generate_refinement_dataset.py" --fraction 0.1
```

### Train Models
```bash
# Train segmentation model
python "scripts/training/train_segmentation.py" --epochs 70 --batch-size 32

# Train refinement model
python "scripts/training/train_refinement.py" --epochs 20 --batch-size 32
```

### Run Inference
```bash
# Test on images
python "scripts/inference/testing.py" --model "models/segmentation/checkpoint_epoch_70.pth"
```

## 📝 Notes

- All paths in scripts use relative paths from the script location
- Model checkpoints automatically save to `outputs/` directories
- Dataset preprocessing creates binary `.pt` cache files for faster loading
- The refinement network adds smooth anti-aliasing to binary masks from the main network

## 🔄 Migration from Old Structure

The old structure had files scattered in the root directory. The reorganization:
- ✅ Groups related functionality together
- ✅ Separates models, scripts, datasets, and outputs
- ✅ Makes paths more predictable and maintainable
- ✅ Easier to navigate and understand the project

### Old → New Path Mappings
| Old Path | New Path |
|----------|----------|
| `final_net.py` | `scripts/training/train_segmentation.py` |
| `train_refinement_net.py` | `scripts/training/train_refinement.py` |
| `testing.py` | `scripts/inference/testing.py` |
| `generate_synthetic_dataset.py` | `scripts/data_generation/generate_dataset.py` |
| `visualizaciones/checkpoint_epoch_70.pth` | `models/segmentation/checkpoint_epoch_70.pth` |
| `visualizaciones/fine_tuning_model/best_refinement_model.pth` | `models/refinement/best_model.pth` |
| `datasets/` (root level) | `datasets/synthetic/` |
| `datasets/fine_tuning/` | `datasets/refinement/` |
