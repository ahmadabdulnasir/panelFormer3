import os
import sys
import json
import yaml
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Dict, Any

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add Sewformer to path
root_path = os.path.dirname(os.path.abspath(__file__))
sewformer_path = os.path.join(root_path, "Sewformer")
sys.path.append(sewformer_path)

# Add SewFactory packages to path
pkg_path = os.path.join(root_path, "SewFactory", "packages")
sys.path.append(pkg_path)

# Import Sewformer modules
import customconfig
import data
import models
from experiment import ExperimentWrappper

# Create FastAPI app
app = FastAPI(
    title="Sewformer API",
    description="API for generating sewing patterns from garment images",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory
OUTPUT_DIR = os.path.join(root_path, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create static files directory for serving results
STATIC_DIR = os.path.join(root_path, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global variables for model and device
model = None
device = None
shape_dataset = None

def load_model():
    """Load the pre-trained Sewformer model"""
    global model, device, shape_dataset
    
    # Load system info
    system_info = customconfig.Properties(os.path.join(sewformer_path, 'system.json'))
    
    # Load config
    config_path = os.path.join(sewformer_path, 'configs/test.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add is_training flag to config if not present
    if 'experiment' in config and 'is_training' not in config['experiment']:
        config['experiment']['is_training'] = False
    
    # Initialize experiment
    wandb_username = system_info.properties['wandb_username'] if system_info.has('wandb_username') else ''
    shape_experiment = ExperimentWrappper(config, wandb_username)
    
    # Override the load_detr_dataset method to fix the dataset loading issue
    def custom_load_dataset(data_root, eval_config={}, unseen=False, batch_size=5):
        # Get data configuration from experiment
        split, _, data_config = shape_experiment.data_info()
        
        # Update configuration with evaluation config
        data_config.update(eval_config)
        
        # Fix paths to be absolute
        if 'panel_classification' in data_config:
            data_config['panel_classification'] = os.path.join(sewformer_path, 'assets/data_configs/panel_classes_condenced.json')
        
        if 'filter_by_params' in data_config:
            data_config['filter_by_params'] = os.path.join(sewformer_path, 'assets/data_configs/param_filter.json')
        
        # Get the dataset class
        import data
        data_class = getattr(data, data_config['class'])
        
        # Create dataset with correct parameters (including sim_root)
        dataset = data_class(data_root, data_root, data_config, 
                           gt_caching=eval_config.get('gt_caching', False),
                           feature_caching=eval_config.get('feature_caching', False))
        
        # Create data wrapper
        datawrapper = data.RealisticDatasetDetrWrapper(dataset, known_split=None, batch_size=batch_size)
        return dataset, datawrapper
    
    # Load dataset and model using custom loader
    shape_dataset, _ = custom_load_dataset(
        [],  # Empty data root for inference only
        {'feature_caching': False, 'gt_caching': False},
        unseen=True, 
        batch_size=1
    )
    
    # Load model
    model, _, device = shape_experiment.load_detr_model(shape_dataset.config, others=False)
    model.eval()
    
    print(f"Model loaded successfully to {device}")
    return model

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
    
    return img_tensor.unsqueeze(0)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    if model is None:
        model = load_model()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Sewformer API. Use /docs to see the API documentation."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Process an image and return the sewing pattern
    """
    global model, device, shape_dataset
    
    if model is None:
        model = load_model()
    
    # Check if file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file to a temporary file
    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Process image
        img_tensor = load_source_appearance(temp_file_path)
        
        # Generate unique ID for this prediction
        import uuid
        prediction_id = str(uuid.uuid4())
        
        # Create output directory for this prediction
        prediction_dir = os.path.join(OUTPUT_DIR, prediction_id)
        os.makedirs(prediction_dir, exist_ok=True)
        
        # Copy input image to output directory
        input_img_path = os.path.join(prediction_dir, "input.jpg")
        shutil.copy(temp_file_path, input_img_path)
        
        # Run inference
        with torch.no_grad():
            output = model(img_tensor.to(device), return_stitches=True)
        
        # Save prediction
        _, _, prediction_img = shape_dataset.save_prediction_single(
            output,
            dataname=prediction_id,
            save_to=prediction_dir,
            return_stitches=True
        )
        
        # Copy results to static directory for serving
        static_prediction_dir = os.path.join(STATIC_DIR, prediction_id)
        os.makedirs(static_prediction_dir, exist_ok=True)
        
        # Copy all files from prediction_dir to static_prediction_dir
        for file_path in Path(prediction_dir).glob("*"):
            shutil.copy(file_path, static_prediction_dir)
        
        # Get all result files
        result_files = {}
        for file_path in Path(static_prediction_dir).glob("*"):
            file_type = file_path.suffix.lstrip('.')
            if file_type in ['png', 'jpg', 'jpeg', 'svg']:
                result_files[file_path.name] = f"/static/{prediction_id}/{file_path.name}"
        
        # Return results
        return {
            "prediction_id": prediction_id,
            "input_image": f"/static/{prediction_id}/input.jpg",
            "results": result_files,
            "message": "Prediction completed successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
