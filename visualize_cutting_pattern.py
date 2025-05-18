#!/usr/bin/env python
import os
import sys
import yaml
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from pathlib import Path
import glob
import shutil

# Import local modules
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

from core.local_experiment import ExperimentWrapper
from core import data

def load_model():
    """Load the pre-trained tech pack model"""
    # Load config
    config_path = os.path.join(root_path, 'configs/inference.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add is_training flag to config if not present
    if 'experiment' in config and 'is_training' not in config['experiment']:
        config['experiment']['is_training'] = False
    
    # Initialize experiment
    shape_experiment = ExperimentWrapper(config)
    
    # Custom function to load dataset
    def custom_load_dataset(data_root, eval_config={}, unseen=False, batch_size=5):
        # Get data configuration from experiment
        split, _, data_config = shape_experiment.data_info()
        
        # Update configuration with evaluation config
        data_config.update(eval_config)
        
        # Fix paths to be absolute
        if 'panel_classification' in data_config:
            data_config['panel_classification'] = os.path.join(root_path, 'configs/data_configs/panel_classes_condenced.json')
        
        if 'filter_by_params' in data_config:
            data_config['filter_by_params'] = os.path.join(root_path, 'configs/data_configs/param_filter.json')
        
        # Get the dataset class
        data_class = getattr(data, data_config['class'])
        
        # Create dataset with correct parameters
        dataset = data_class(data_root, data_root, data_config, 
                           gt_caching=eval_config.get('gt_caching', False),
                           feature_caching=eval_config.get('feature_caching', False))
        
        # Create data wrapper
        datawrapper = data.GarmentDatasetWrapper(dataset, known_split=None, batch_size=batch_size)
        return dataset, datawrapper
    
    # Custom function to load model with pre-trained weights
    def custom_load_model(data_config):
        import models
        import torch
        from torch import nn
        
        # Build the model
        model, criterion = models.build_former(shape_experiment.in_config)
        
        # Check CUDA availability
        print(f"CUDA available: {torch.cuda.is_available()}")
        device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            
        # Move model to device first (before DataParallel wrapping)
        model.to(device)
        criterion.to(device)
        
        # Load pre-trained weights
        model_path = os.path.join(root_path, config["model_to_use"])
        if os.path.exists(model_path):
            print(f"Loading Trained weights from {model_path}")
            try:
                # First try with weights_only=False to handle PyTorch 2.6 security changes
                try:
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                    print("Loaded checkpoint with weights_only=False")
                except Exception as e1:
                    print(f"Error loading with weights_only=False: {e1}")
                    # Try with safe_globals context manager
                    from torch.serialization import safe_globals
                    print("Trying with safe_globals for CosineAnnealingLR...")
                    with safe_globals(['torch.optim.lr_scheduler.CosineAnnealingLR']):
                        checkpoint = torch.load(model_path, map_location=device)
                        print("Loaded checkpoint with safe_globals")
                
                # Load state dict directly (without DataParallel wrapping) with strict=False
                if 'model_state_dict' in checkpoint:
                    # Use strict=False to ignore mismatched keys
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print("Trained weights loaded successfully (with strict=False to handle architecture differences)")
                else:
                    print("No model_state_dict found in checkpoint")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Initializing model with random weights")
        else:
            print(f"Warning: Trained model not found at {model_path}")
            print("Using model with random initialization")
        
        # Wrap model with DataParallel AFTER loading weights
        if torch.cuda.is_available():
            model = nn.DataParallel(model, device_ids=[0])
        
        return model, criterion, device
    
    # Load dataset and model
    shape_dataset, _ = custom_load_dataset(
        [],  # Empty data root for inference only
        {'feature_caching': False, 'gt_caching': False},
        unseen=True, 
        batch_size=1
    )
    
    # Load model
    model, _, device = custom_load_model(shape_dataset.config)
    model.eval()
    
    print(f"Model loaded successfully to {device}")
    return model, shape_dataset, device

def load_source_appearance(img_path):
    """Process input image for the model"""
    ref_img = Image.open(img_path).convert('RGB')
    h, w = ref_img.size
    min_size, max_size = min(h, w), max(h, w)
    
    # Pad image to make it square
    pad_ref_img = T.Pad(padding=(int((max_size - h) / 2), int((max_size - w) / 2)), fill=255)(ref_img)
    
    # Resize, convert to tensor, and normalize with ImageNet mean and std
    img_tensor = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])(pad_ref_img)
    
    return img_tensor.unsqueeze(0), ref_img

def find_svg_files(directory):
    """Find all SVG files in the directory and its subdirectories"""
    svg_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".svg"):
                svg_files.append(os.path.join(root, file))
    return svg_files

def visualize_cutting_pattern(img_path, output_path=None):
    """Visualize input image with complete cutting pattern"""
    # Load model
    model, shape_dataset, device = load_model()
    
    # Load and process image
    img_tensor, original_img = load_source_appearance(img_path)
    
    # Create output directory
    if output_path is None:
        output_path = os.path.join(root_path, "outputs/visualizations")
    os.makedirs(output_path, exist_ok=True)
    
    # Generate unique ID for this prediction
    import uuid
    prediction_id = str(uuid.uuid4())
    
    # Create output directory for this prediction
    prediction_dir = os.path.join(output_path, prediction_id)
    os.makedirs(prediction_dir, exist_ok=True)
    
    # Run inference
    with torch.no_grad():
        output = model(img_tensor.to(device), return_stitches=True)
    
    try:
        # Save prediction
        panel_order, panel_idx, prediction_img = shape_dataset.save_prediction_single(
            output,
            dataname=prediction_id,
            save_to=prediction_dir,
            return_stitches=True
        )
        
        # Check if prediction_img exists
        if not os.path.exists(prediction_img):
            print(f"Warning: Prediction image not found at {prediction_img}")
            # Try to find the pattern image in the prediction directory
            pattern_files = []
            for root, _, files in os.walk(prediction_dir):
                for file in files:
                    if file.endswith("_pattern.png"):
                        pattern_files.append(os.path.join(root, file))
            
            if pattern_files:
                prediction_img = pattern_files[0]
                print(f"Found alternative pattern image at {prediction_img}")
            else:
                raise FileNotFoundError(f"No pattern image found in {prediction_dir}")
        
        # Find SVG files (these contain the cutting pattern details)
        svg_files = find_svg_files(prediction_dir)
        
        # Create a figure with the input image and cutting pattern
        fig = plt.figure(figsize=(15, 10))
        
        # Create a 2x2 grid for the layout
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])
        
        # Add the original image in the top-left
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original_img)
        ax1.set_title("Input Garment Image")
        ax1.axis("off")
        
        # Add the pattern visualization in the top-right
        ax2 = fig.add_subplot(gs[0, 1])
        pattern_img = Image.open(prediction_img)
        ax2.imshow(pattern_img)
        ax2.set_title("Panel Visualization")
        ax2.axis("off")
        
        # Add the complete cutting pattern in the bottom row (spanning both columns)
        ax3 = fig.add_subplot(gs[1, :])
        
        # If we have SVG files, convert the first one to PNG and display it
        if svg_files:
            from cairosvg import svg2png
            
            # Use the first SVG file (usually the main pattern)
            svg_file = svg_files[0]
            png_file = os.path.join(prediction_dir, "cutting_pattern.png")
            
            # Convert SVG to PNG
            svg2png(url=svg_file, write_to=png_file, scale=2.0)
            
            # Display the PNG
            cutting_pattern = Image.open(png_file)
            ax3.imshow(cutting_pattern)
            ax3.set_title("Complete Cutting Pattern")
            ax3.axis("off")
        else:
            ax3.text(0.5, 0.5, "No cutting pattern available", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax3.transAxes, fontsize=14)
        
        plt.tight_layout()
        
        # Save the complete visualization
        vis_path = os.path.join(prediction_dir, "complete_pattern_visualization.png")
        plt.savefig(vis_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        
        print(f"Visualization saved to {vis_path}")
        return vis_path, prediction_dir
            
    except Exception as e:
        print(f"Error during visualization: {e}")
        # Create a simple visualization with just the input image
        plt.figure(figsize=(8, 8))
        plt.imshow(original_img)
        plt.title("Input Image (Error in pattern generation)")
        plt.axis("off")
        
        # Save visualization
        vis_path = os.path.join(prediction_dir, "input_only.png")
        plt.savefig(vis_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        
        print(f"Error visualization saved to {vis_path}")
        return vis_path, prediction_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize complete cutting pattern from an input image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    vis_path, _ = visualize_cutting_pattern(args.image, args.output)
    print(f"Visualization complete. Results saved to {vis_path}")
