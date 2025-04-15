#!/bin/bash
# Script to train the Sewformer model

# Set PYTHONPATH to include necessary directories
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/Factory/packages

# Run the training script
torchrun --standalone --nnodes=1 --nproc_per_node=1 former/train.py -c former/configs/train.yaml
