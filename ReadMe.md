# TechPack: Garment Sewing Pattern Generator API by Fashable.AI

A FastAPI-based web service that generates sewing patterns from garment images using the former model.

## Overview

TechPack is a production-ready implementation of the former model, which can generate sewing patterns from a single garment image. The system is deployed as a REST API service that allows users to upload garment images and receive detailed sewing pattern information.

## Features

- **Single Image Processing**: Generate complete sewing patterns from a single garment image
- **REST API Interface**: Easy integration with web and mobile applications
- **GPU Acceleration**: Utilizes GPU for faster inference (with CPU fallback)
- **Visualization**: Provides visual outputs of the generated sewing patterns
- **Production-Ready**: Includes deployment configurations for production environments

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended, but not required)
- 8GB+ RAM
- 200GB+ disk space

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/techPack.git
cd techPack
```

### 2. Set Up Python Environment

Using venv (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or Better use UV 
```bash
uv sync
```
Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Download Pre-trained Model

```bash
mkdir -p former/assets/ckpts
wget https://huggingface.co/liulj/sewformer/resolve/main/Detr2d-V6-final-dif-ce-focal-schd-agp_checkpoint_37.pth -O former/assets/ckpts/Detr2d-V6-final-dif-ce-focal-schd-agp_checkpoint_37.pth
```

### 4. GPU Support
If the system have GPU, install torch with cuda support (Make sure the environment is activated conda or venv)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
```


## Usage

### Starting the API Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. You can access the interactive API documentation at `http://localhost:8000/docs`.

### API Endpoints

- `GET /`: Root endpoint, returns a welcome message
- `POST /predict/`: Upload a garment image to generate a sewing pattern
- `GET /prediction/{prediction_id}`: Retrieve a previously generated prediction

### Example Usage

Using curl:

```bash
curl -X POST -F "file=@path/to/your/garment_image.jpg" http://localhost:8000/predict/
```

Using Python requests:

```python
import requests

url = "http://localhost:8000/predict/"
files = {"file": open("path/to/your/garment_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Deployment

### Systemd Service (Linux)

A systemd service configuration is provided in the `deployment` directory. To deploy as a service:

1. Copy the service file to systemd directory:
   ```bash
   sudo cp deployment/techPack.start.uvicorn.service /etc/systemd/system/
   ```

2. Create a startup script:
   ```bash
   echo '#!/bin/bash
   cd /path/to/techPack
   source .venv/bin/activate
   uvicorn main:app --host 0.0.0.0 --port 8000' > start.techPack.sh
   chmod +x start.techPack.sh
   ```

3. Enable and start the service:
   ```bash
   sudo systemctl enable techPack.start.uvicorn.service
   sudo systemctl start techPack.start.uvicorn.service
   ```

## Project Structure

```
techPack/
├── former/           # Core former model implementation
├── Factory/          # Simulation and data generation tools
├── deployment/          # Deployment configuration files
├── static/              # Static files for serving results
├── outputs/             # Generated output files
├── main.py              # FastAPI application entry point
├── requirements.txt     # Python dependencies
└── ReadMe.md            # This documentation
```

## Troubleshooting

### Common Issues

1. **GPU not detected**: The application will automatically fall back to CPU if no GPU is detected. To force CPU usage, set the environment variable `CUDA_VISIBLE_DEVICES=""`.

2. **Model loading errors**: Ensure the model checkpoint is correctly downloaded to `former/assets/ckpts/`.

3. **Memory errors**: If you encounter memory issues on GPU, try reducing the batch size in the configuration.