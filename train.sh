#!/bin/bash
# Script to train the Sewformer model

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project root directory
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include necessary directories
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/Factory/packages

# Create assets directory if it doesn't exist
mkdir -p "$(pwd)/former/assets/data_configs"

# Copy configuration files if they don't exist in the target location
if [ ! -f "$(pwd)/former/assets/data_configs/panel_classes_condenced.json" ]; then
  echo "Copying panel_classes_condenced.json to assets directory"
  cp "$(pwd)/former/assets/data_configs/panel_classes_condenced.json" "$(pwd)/assets/data_configs/" 2>/dev/null || echo "Warning: Could not copy panel_classes_condenced.json"
fi

if [ ! -f "$(pwd)/former/assets/data_configs/param_filter.json" ]; then
  echo "Copying param_filter.json to assets directory"
  cp "$(pwd)/former/assets/data_configs/param_filter.json" "$(pwd)/assets/data_configs/" 2>/dev/null || echo "Warning: Could not copy param_filter.json"
fi

# Run the training script
torchrun --standalone --nnodes=1 --nproc_per_node=1 former/train.py -c former/configs/train.yaml
