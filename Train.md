# Training the Sewformer Model

This document provides detailed instructions on how to train the Sewformer model, configure checkpoints, and resume training from saved checkpoints.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Dataset Preparation](#dataset-preparation)
- [Training Configuration](#training-configuration)
- [Running Training](#running-training)
- [Checkpoint Configuration](#checkpoint-configuration)
- [Resuming Training](#resuming-training)
- [Monitoring Training](#monitoring-training)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before starting the training process, ensure you have:

1. Installed all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or using UV (recommended for faster dependency resolution):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh  # Install UV on Unix/Linux
   uv sync  # Install dependencies
   ```

2. GPU with CUDA support (recommended)
   - The training script will automatically detect available GPUs
   - If no GPU is available, it will fall back to CPU (significantly slower)

3. PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
   ```

## Dataset Preparation

The Sewformer model requires a specific dataset structure:

```
Factory/sewformer_dataset/
├── garment_type1/
│   ├── renders/        # Garment images (.png files)
│   └── static/         # Ground truth data (specification.json)
├── garment_type2/
│   ├── renders/
│   └── static/
...
```

### Creating a Test Dataset

If you don't have a real dataset, you can create a dummy dataset for testing:

```bash
python create_dummy_dataset.py
```

This script creates a minimal dataset with sample garments in the correct structure.

## Training Configuration

The training process is controlled by several configuration files:

### 1. System Configuration (`former/system.json`)

This file defines paths to datasets and output directories:

```json
{
  "output": "outputs",
  "datasets_path": "Factory/sewformer_dataset",
  "sim_root": "outputs",
  "wandb_username": ""
}
```

- `output`: Directory where training outputs and checkpoints will be saved
- `datasets_path`: Path to the dataset directory
- `sim_root`: Directory for simulation outputs
- `wandb_username`: Username for Weights & Biases logging (optional)

### 2. Training Configuration (`former/configs/train.yaml`)

This file contains all training hyperparameters and model configuration:

```yaml
experiment:
  project_name: Pattern-Recovery
  run_name: Training
  run_id: 
  local_dir: outputs/checkpoints
  is_training: True
  save_checkpoint_steps: 100  # Save a checkpoint every 100 steps
  save_checkpoint_epochs: 1   # Save a checkpoint every epoch

# Model configuration
NN:
  pre-trained:  # Path to pre-trained model for resuming training
  # ... other model parameters

# Training parameters
trainer:
  batch_size: 64
  devices: [0]  # GPU devices to use
  epochs: 40
  lr: 0.0002
  # ... other training parameters
```

Key parameters to consider:
- `experiment.local_dir`: Directory for saving checkpoints
- `experiment.save_checkpoint_steps`: Save a checkpoint every N steps
- `experiment.save_checkpoint_epochs`: Save a checkpoint every N epochs
- `NN.pre-trained`: Path to a pre-trained model for resuming training
- `trainer.epochs`: Number of training epochs
- `trainer.batch_size`: Batch size for training
- `trainer.devices`: GPU devices to use for training

## Running Training

To start training, use the provided training script:

```bash
./train.sh
```

This script:
1. Sets up the Python path correctly
2. Creates necessary directories
3. Ensures configuration files are in the right locations
4. Creates a minimal dataset if none exists
5. Runs the training with the correct parameters

### Manual Training Command

If you prefer to run the training command manually:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/Factory/packages
torchrun --standalone --nnodes=1 --nproc_per_node=1 former/train.py -c former/configs/train.yaml
```

## Checkpoint Configuration

The Sewformer model saves checkpoints in several ways:

### 1. Regular Checkpoints

By default, the model saves a checkpoint at the end of each epoch. You can configure this with:

```yaml
experiment:
  save_checkpoint_epochs: 1  # Save checkpoint every epoch
```

### 2. Step-Based Checkpoints

You can save checkpoints at specific training steps:

```yaml
experiment:
  save_checkpoint_steps: 100  # Save checkpoint every 100 steps
```

Set to 0 to disable step-based checkpoints.

### 3. Best Model Checkpoints

The model automatically saves a special checkpoint when:
- The validation loss improves
- There's a structural update to the loss function

These checkpoints are tagged with the 'best' alias.

### Checkpoint Location

Checkpoints are saved in the directory specified by:
- `system.json` -> `output` parameter
- `train.yaml` -> `experiment.local_dir` parameter

The default location is: `outputs/checkpoints/`

### Checkpoint Contents

Each checkpoint contains:
- Model state dictionary
- Optimizer state
- Scheduler state (if used)
- Current epoch
- Current step (for step-based checkpoints)

## Resuming Training

To resume training from a checkpoint:

### 1. Specify the Checkpoint in Configuration

Edit `former/configs/train.yaml` and set the pre-trained model path:

```yaml
NN:
  pre-trained: outputs/checkpoints/checkpoint_37.pth  # Path to your checkpoint
```

### 2. Run Training

Run the training script as usual:

```bash
./train.sh
```

The training will:
1. Load the model weights from the checkpoint
2. Restore optimizer and scheduler states
3. Continue training from the saved epoch

### 3. Resuming from Best Checkpoint

To resume from the best checkpoint:

```yaml
NN:
  pre-trained: outputs/checkpoints/checkpoint_best.pth
```

## Monitoring Training

### Console Output

The training script outputs:
- Loss values for each epoch
- Validation loss
- Learning rate
- Checkpoint saving information

### Weights & Biases Integration

The model supports logging to Weights & Biases:

1. Set your W&B username in `system.json`:
   ```json
   {
     "wandb_username": "your_username"
   }
   ```

2. Log in to W&B:
   ```bash
   wandb login
   ```

3. Run training as usual

## Troubleshooting

### Common Issues

1. **FileNotFoundError for system.json**
   - Ensure you're running the script from the project root directory
   - Check that `former/system.json` exists

2. **No module named 'customconfig'**
   - Make sure PYTHONPATH includes the Factory/packages directory
   - Use the provided `train.sh` script which sets up paths correctly

3. **Dataset directory not found**
   - Ensure the dataset path in `system.json` is correct
   - Run `python create_dummy_dataset.py` to create a test dataset

4. **CUDA out of memory**
   - Reduce batch size in `former/configs/train.yaml`
   - Use a smaller model or fewer layers

5. **Distributed training issues**
   - For single-GPU training, use `--nproc_per_node=1`
   - For multi-GPU training, set `trainer.devices` to include all GPUs

### Getting Help

If you encounter issues not covered here, check:
- The original Sewformer repository documentation
- PyTorch distributed training documentation
- Open an issue in the project repository
