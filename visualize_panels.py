#!/usr/bin/env python
"""
Visualization script for PanelFormer model outputs
Shows input image and panels side by side
"""
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from pathlib import Path
import uuid
import shutil

# Add parent directory to path
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

# Import model related modules
from core.local_experiment import ExperimentWrapper
from core import data
import yaml


def load_model(config_path):
    """Load the pre-trained PanelFormer model"""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add is_training flag to config if not present
    if 'experiment' in config and 'is_training' not in config['experiment']:
        config['experiment']['is_training'] = False
    
    # Initialize experiment
    shape_experiment = ExperimentWrapper(config)
    
    # Load dataset
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
        
        # Set up device
        device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Move model to device
        model = nn.DataParallel(model, device_ids=[0] if torch.cuda.is_available() else None)
        criterion.to(device)
        
        # Load pre-trained weights
        model_path = os.path.join(root_path, config["model_to_use"])
        if os.path.exists(model_path):
            print(f"Loading trained weights from {model_path}")
            try:
                # First try with weights_only=False to handle the optimizer state
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Trained weights loaded successfully")
            except Exception as e:
                print(f"Error loading model with weights_only=False: {e}")
                try:
                    # Try with safe_globals context manager
                    from torch.serialization import safe_globals
                    with safe_globals(['torch.optim.lr_scheduler.CosineAnnealingLR']):
                        checkpoint = torch.load(model_path, map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        print("Trained weights loaded successfully using safe_globals")
                except Exception as e2:
                    print(f"Error loading model with safe_globals: {e2}")
                    print("Initializing model with random weights")
        else:
            print(f"Warning: Trained model not found at {model_path}")
            print("Using model with random initialization")
        
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
    
    # Resize and convert to tensor
    img_tensor = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor()
    ])(pad_ref_img)
    
    return img_tensor.unsqueeze(0), ref_img


def visualize_panels(input_img_path, output_path=None, config_path=None):
    """
    Visualize the input image and the panels side by side
    
    Args:
        input_img_path: Path to the input image
        output_path: Path to save the visualization (optional)
        config_path: Path to the model configuration file (optional)
    """
    # Set default config path if not provided
    if config_path is None:
        config_path = os.path.join(root_path, 'configs/inference.yaml')
    
    # Load model
    model, shape_dataset, device = load_model(config_path)
    
    # Process image
    img_tensor, original_img = load_source_appearance(input_img_path)
    
    # Generate unique ID for this prediction
    prediction_id = str(uuid.uuid4())
    
    # Create output directory for this prediction
    prediction_dir = os.path.join(root_path, "outputs/predictions", prediction_id)
    os.makedirs(prediction_dir, exist_ok=True)
    
    # Run inference
    with torch.no_grad():
        output = model(img_tensor.to(device), return_stitches=True)
    
    # Save prediction and get panel information
    panel_order, panel_idx, prediction_img = shape_dataset.save_prediction_single(
        output,
        dataname=prediction_id,
        save_to=prediction_dir,
        return_stitches=True
    )
    
    print(f"Panel order: {panel_order}")
    print(f"Panel indices: {panel_idx}")
    
    # Find individual panel images using the panel_order information
    panel_images = []
    panel_names = []
    
    # Look for panel images in the prediction directory
    for panel_name in panel_order:
        if panel_name is None:
            continue
            
        # Try different patterns to find the panel image
        panel_patterns = [
            f"**/panel_{panel_name}.*",  # panel_front.png
            f"**/{panel_name}_panel.*",  # front_panel.png
            f"**/{panel_name}.*"         # front.png
        ]
        
        found = False
        for pattern in panel_patterns:
            matches = list(Path(prediction_dir).glob(pattern))
            for match in matches:
                if match.is_file() and match.suffix.lstrip('.') in ['png', 'jpg', 'jpeg', 'svg']:
                    panel_images.append(match)
                    panel_names.append(panel_name)
                    found = True
                    break
            if found:
                break
    
    # If no panel images found using panel_order, fall back to searching by pattern
    if not panel_images:
        print("No panel images found using panel_order, falling back to pattern search")
        # Look for files with 'panel' in the name but not 'pattern'
        for file_path in Path(prediction_dir).glob("**/*"):
            if file_path.is_file() and "panel" in file_path.name.lower() and "pattern" not in file_path.name.lower():
                file_type = file_path.suffix.lstrip('.')
                if file_type in ['png', 'jpg', 'jpeg', 'svg']:
                    panel_images.append(file_path)
                    # Try to extract panel name from filename
                    name_parts = file_path.stem.split('_')
                    if len(name_parts) > 1:
                        panel_names.append(name_parts[1])
                    else:
                        panel_names.append(file_path.stem)
        
        # If still no panel images found, look for any image
        if not panel_images:
            for file_path in Path(prediction_dir).glob("**/*"):
                if file_path.is_file():
                    file_type = file_path.suffix.lstrip('.')
                    if file_type in ['png', 'jpg', 'jpeg', 'svg']:
                        panel_images.append(file_path)
                        panel_names.append(file_path.stem)
    
    # Create visualization
    # Determine the number of panels to display (up to 4)
    num_panels = min(len(panel_images), 4) if panel_images else 0
    
    if num_panels == 0:
        # Simple 1x2 layout if no panels found
        plt.figure(figsize=(16, 8))
        
        # Plot input image
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title("Input Image")
        plt.axis('off')
        
        # Empty panel area
        plt.subplot(1, 2, 2)
        plt.text(0.5, 0.5, "No panel images found", ha='center', va='center')
        plt.title("Prediction Failed")
        plt.axis('off')
    
    elif num_panels == 1:
        # Simple 1x2 layout for one panel
        plt.figure(figsize=(16, 8))
        
        # Plot input image
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title("Input Image")
        plt.axis('off')
        
        # Plot single panel
        plt.subplot(1, 2, 2)
        panel_img = Image.open(panel_images[0])
        plt.imshow(panel_img)
        plt.title(f"Panel: {panel_images[0].name}")
        plt.axis('off')
    
    else:
        # Create a layout with multiple panels
        # Input image on left, panels on right in a grid
        plt.figure(figsize=(16, 12))
        
        # Plot input image (larger)
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title("Input Image")
        plt.axis('off')
        
        # Create a subplot for the panels
        panel_grid = plt.subplot(1, 2, 2)
        panel_grid.set_title("Individual Panels")
        panel_grid.axis('off')
        
        # Determine grid layout based on number of panels
        if num_panels == 2:
            rows, cols = 1, 2
        else:  # 3 or 4 panels
            rows, cols = 2, 2
        
        # Create a grid of panels
        for i in range(num_panels):
            plt.subplot(rows, cols*2, cols*2 + i + 1)
            panel_img = Image.open(panel_images[i])
            plt.imshow(panel_img)
            
            # Use panel name if available, otherwise use filename
            if i < len(panel_names):
                panel_title = f"Panel: {panel_names[i]}"
            else:
                panel_title = f"Panel {i+1}: {panel_images[i].name}"
                
            plt.title(panel_title)
            plt.axis('off')
    
    plt.tight_layout()
    
    plt.tight_layout()
    
    # Save or show the visualization
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    # Return paths to all generated files
    return {
        "input_image": input_img_path,
        "prediction_dir": prediction_dir,
        "panel_images": panel_images
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PanelFormer predictions")
    parser.add_argument("--input", "-i", required=True, help="Path to input image")
    parser.add_argument("--output", "-o", help="Path to save visualization")
    parser.add_argument("--config", "-c", help="Path to model configuration file")
    
    args = parser.parse_args()
    
    visualize_panels(args.input, args.output, args.config)
