# PanelFormer Project

A deep learning framework for garment panel reconstruction and stitching prediction.

## Project Structure

```
panelformer_project/
├── data/
│   ├── raw/              # Original dataset
│   ├── processed/        # Preprocessed data (train/val/test splits)
│   └── utils/            # Utilities for data processing
│       ├── parse_svg.py  # Convert SVG patterns to edge format
│       └── segmentation_utils.py  # Read segmentation.txt & match vertices
├── models/
│   ├── panel_transformer.py  # Transformer + ResNet50 encoder + decoders
│   ├── stitch_predictor.py   # MLP-based stitch predictor
│   └── __init__.py
├── train/
│   ├── train_panel_transformer.py
│   ├── train_stitch_predictor.py
│   └── losses.py         # Edge, placement, loop, and stitch losses
├── augmentations/
│   ├── panel_masking.py
│   ├── garment_mixing.py
│   └── utils.py
├── evaluation/
│   ├── evaluate_panels.py
│   ├── evaluate_stitches.py
│   └── metrics.py
├── configs/
│   └── panelformer.yaml  # Config for hyperparams, paths, etc.
├── outputs/
│   ├── logs/
│   ├── checkpoints/
│   └── visualizations/
├── inference/
│   └── predict_from_image.py  # Run full model on a real garment image
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

[Add usage instructions here]

# Evaluate panel prediction
python evaluation/evaluate_panels.py --checkpoint outputs/checkpoints/panel_best_model.pt --data_dir data/test_garments

# Evaluate stitch prediction
python evaluation/evaluate_stitches.py --checkpoint outputs/checkpoints/stitch_predictor_best_model.pt --data_dir data/test_garments


# Predict garment pattern from image
python inference/predict_garment_pattern.py --model_path outputs/checkpoints/panel_best_model.pt --image_path data/test_garments/1.jpg


python -m train.train_techpack_style --data_root DATASET --output_dir outputs/garment_pattern_estimation --batch_size 8 --epochs 100

python -m inference.predict_garment_pattern --model_path outputs/garment_pattern_estimation/best_model.pth --image_path assets/your_image.jpg

# This is how we will split train/val dataset when we have enough dataset
```python
    train_dataset = StitchDataset(
        data_dir=data_dir,
        transform=None  # Use default transforms
    )
    
    # Split dataset into train/val
    from torch.utils.data import random_split
    val_size = int(len(train_dataset) * 0.2)  # 20% for validation
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config_manager.get('training.batch_size', 32),
        shuffle=True,
        num_workers=config_manager.get('training.num_workers', 4),
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config_manager.get('training.batch_size', 32),
        shuffle=False,
        num_workers=config_manager.get('training.num_workers', 4),
        pin_memory=True
    )
    ```