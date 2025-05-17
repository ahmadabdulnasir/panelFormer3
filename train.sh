#!/bin/bash
# Script to train the Sewformer model

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project root directory
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include necessary directories
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/Factory/packages

# Create all necessary directories
mkdir -p "$(pwd)/former/assets/data_configs"
mkdir -p "$(pwd)/data_configs"
mkdir -p "$(pwd)/outputs/checkpoints"
mkdir -p "$(pwd)/outputs/predictions"

# Copy configuration files if they don't exist in the target location
if [ ! -f "$(pwd)/former/assets/data_configs/panel_classes_condenced.json" ]; then
  echo "Copying panel_classes_condenced.json to assets directory"
  cp "$(pwd)/former/assets/data_configs/panel_classes_condenced.json" "$(pwd)/assets/data_configs/" 2>/dev/null || echo "Warning: Could not copy panel_classes_condenced.json"
fi

if [ ! -f "$(pwd)/former/assets/data_configs/param_filter.json" ]; then
  echo "Copying param_filter.json to assets directory"
  cp "$(pwd)/former/assets/data_configs/param_filter.json" "$(pwd)/assets/data_configs/" 2>/dev/null || echo "Warning: Could not copy param_filter.json"
fi

# Check if the dataset directory exists, if not create a minimal structure for testing
DATASET_DIR="$(pwd)/Factory/sewformer_dataset"
if [ ! -d "$DATASET_DIR" ]; then
  echo "Dataset directory not found. Creating a minimal dataset structure for testing."
  mkdir -p "$DATASET_DIR/sample_garment/renders"
  mkdir -p "$DATASET_DIR/sample_garment/static"
  
  # Create a dummy specification file
  echo '{
  "name": "sample_garment",
  "panels": [
    {
      "name": "panel1",
      "vertices": [[0, 0], [1, 0], [1, 1], [0, 1]],
      "edges": [[0, 1], [1, 2], [2, 3], [3, 0]]
    }
  ],
  "stitches": []
}' > "$DATASET_DIR/sample_garment/static/specification.json"
  
  # Create a dummy image file
  convert -size 1024x1024 xc:white "$DATASET_DIR/sample_garment/renders/sample.png" 2>/dev/null || \
  echo "Warning: Could not create sample image. Please install ImageMagick or manually create a sample image."
fi

# Run the training script
torchrun --standalone --nnodes=1 --nproc_per_node=1 former/train.py -c former/configs/train.yaml
