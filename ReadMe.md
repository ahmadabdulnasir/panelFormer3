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
wget "https://storage.googleapis.com/wandb-artifacts-prod/wandb_artifacts/616401790/1671104183/a6e7a724efedbb00e75cd56647c8ff37?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gorilla-files-url-signer-man%40wandb-production.iam.gserviceaccount.com%2F20250423%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250423T144911Z&X-Goog-Expires=3599&X-Goog-Signature=7563a5c4756b5647896b2ad5681412d296a5c3b00501792bb7eb4403fc08489510bdd3a2d0a4e0e79be4783239fd8ad445dc2308594cf847b73e39a4a69e9e31ddc1f017b98c61e34c37f03029d9e5f562a19c2bff5819c5d947b60e5d15c9f513863599f458bfbcc0786a5f8b589432e8c63d34fd359ad8da6730c0ec9661fcfee218832ddcd4ac5eb907181af10656b8ccb54123545ec062906d9089b0bbe236b60ac51cfe33e2f3e50b12f6daf33d770b197bb7faf936c38be065261619b2f98fb1b371f954bae73756927ca29e6ddfd81e55201b519ccdb2bb652705b7c1f41bd8ef3f839129fec66c7fef4b8ae99297c550baadbbd460bb65abdb39e1e7&X-Goog-SignedHeaders=host&X-User=ahmadabdulnasir&response-content-disposition=attachment%3B+filename%3D%22checkpoint_119.pth%22" -O former/assets/ckpts/Detr2d-V6-final-dif-ce-focal-schd-agp_checkpoint_37.pth
```

### 4. GPU Support
If the system have GPU, install torch with cuda support (Make sure the environment is activated conda or venv)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
```


/Users/ahmadabdulnasirshuaib/wsp/labs/techPack/Factory/
├── assets/
├── data_generator/
├── meta_infos/
├── packages/
├── sewformer_dataset/  <-- Your new dataset folder
│   ├── renders/        <-- Put garment images here
│   └── static/         <-- Put ground truth data here
├── .gitignore
└── ReadMe.md

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