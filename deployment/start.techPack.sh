#!/bin/bash

# Path to your conda installation - adjust if needed
CONDA_PATH="/home/ubuntu/anaconda3/etc/profile.d/conda.sh"

# Name of your conda environment
CONDA_ENV="CatVTON-Flux"

# API service directory
SERVICE_DIR="/home/ubuntu/CatVTON-Flux/src"

# Socket path within service directory
SOCKET_PATH="$SERVICE_DIR/unet_api_automask.sock"

# Calculate optimal number of workers based on CPU cores
# Using number of CPU cores minus 1 for optimal performance, minimum of 1
WORKERS=$(( $(nproc) - 1 ))
if [ "$WORKERS" -lt 1 ]; then
    WORKERS=1
fi

# Source conda
source "$CONDA_PATH"

# Activate the environment
conda activate "$CONDA_ENV"

# Remove existing socket if it exists
if [ -e "$SOCKET_PATH" ]; then
    rm "$SOCKET_PATH"
fi

# Navigate to service directory
cd "$SERVICE_DIR"

# Start the FastAPI service using the conda environment's uvicorn
# python /home/ubuntu/virtualtryon/src/app_api.py
# Added --timeout 300 for longer running ML tasks
python -m uvicorn unet_api_automask:app \
    --uds "$SOCKET_PATH"
    # --workers $WORKERS \
    # --timeout-keep-alive 300 \
    # --log-level info \
    # --proxy-headers \
    # --limit-concurrency 1000