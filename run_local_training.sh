#!/bin/bash
# Script to run a local training session for PanelFormer

# Set environment variables
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/Panelformer/hybrid

# Create output directories
mkdir -p outputs/checkpoints
mkdir -p outputs/logs


# Run the local training script
cd /home/ubuntu/Panelformer/hybrid
python train/train_local.py --config configs/train.yaml

echo "Local training completed. Check outputs/ for results."
