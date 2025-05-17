#!/bin/bash
# Script to run a test training session for PanelFormer

# Set environment variables
export PYTHONPATH=$PYTHONPATH:/Users/ahmadabdulnasirshuaib/wsp/labs/panelFormer3

# Create output directories
mkdir -p outputs/test_run/checkpoints
mkdir -p outputs/test_run/predictions
mkdir -p outputs/test_run/visualizations

# Run the training script with the test configuration
cd /Users/ahmadabdulnasirshuaib/wsp/labs/panelFormer3
python core/train.py --config configs/test.yaml

echo "Test training completed. Check outputs/test_run for results."
