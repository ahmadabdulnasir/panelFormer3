#!/bin/bash
# Script to run a local training session for PanelFormer

# Set environment variables
export PYTHONPATH=$PYTHONPATH:/Users/ahmadabdulnasirshuaib/wsp/labs/panelFormer3

# Create output directories
mkdir -p outputs/checkpoints
mkdir -p outputs/logs
mkdir -p outputs/artifacts


# Run the local training script
cd /Users/ahmadabdulnasirshuaib/wsp/labs/panelFormer3
python train/train_local.py --config configs/train.yaml

echo "Local training completed. Check outputs/ for results."
