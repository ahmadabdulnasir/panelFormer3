#!/usr/bin/env python3
"""
Script to create a dummy dataset for testing the Sewformer model training.
This creates a minimal dataset structure with sample garment data.
"""

import os
import json
import numpy as np
from PIL import Image
import shutil

def create_dummy_dataset(base_dir, num_samples=5):
    """Create a dummy dataset with the specified number of samples."""
    dataset_dir = os.path.join(base_dir, "Factory", "sewformer_dataset")
    
    # Create the main dataset directory
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create sample garments
    for i in range(num_samples):
        garment_name = f"sample_garment_{i}"
        garment_dir = os.path.join(dataset_dir, garment_name)
        renders_dir = os.path.join(garment_dir, "renders")
        static_dir = os.path.join(garment_dir, "static")
        
        # Create directories
        os.makedirs(renders_dir, exist_ok=True)
        os.makedirs(static_dir, exist_ok=True)
        
        # Create a dummy image (1024x1024 white image)
        img = Image.new('RGB', (1024, 1024), color='white')
        img.save(os.path.join(renders_dir, f"sample_{i}.png"))
        
        # Create a dummy specification file with random panels
        num_panels = np.random.randint(2, 5)
        panels = []
        
        for p in range(num_panels):
            # Create a simple quadrilateral panel with random vertices
            vertices = []
            for _ in range(4):
                vertices.append([float(np.random.rand()), float(np.random.rand())])
            
            panels.append({
                "name": f"panel{p}",
                "vertices": vertices,
                "edges": [[0, 1], [1, 2], [2, 3], [3, 0]]
            })
        
        # Create some random stitches between panels
        stitches = []
        if num_panels > 1:
            for s in range(np.random.randint(1, num_panels)):
                panel1 = np.random.randint(0, num_panels)
                panel2 = np.random.randint(0, num_panels)
                while panel2 == panel1:
                    panel2 = np.random.randint(0, num_panels)
                
                stitches.append({
                    "from_panel": f"panel{panel1}",
                    "from_edge": np.random.randint(0, 4),
                    "to_panel": f"panel{panel2}",
                    "to_edge": np.random.randint(0, 4)
                })
        
        # Write the specification file
        spec = {
            "name": garment_name,
            "panels": panels,
            "stitches": stitches
        }
        
        with open(os.path.join(static_dir, "specification.json"), 'w') as f:
            json.dump(spec, f, indent=2)
    
    print(f"Created dummy dataset with {num_samples} samples at {dataset_dir}")

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the dummy dataset
    create_dummy_dataset(script_dir, num_samples=5)
