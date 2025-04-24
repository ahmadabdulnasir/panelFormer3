# TechPack: Garment Sewing Pattern Generator - Codebase Explanation

This document provides a detailed explanation of the TechPack codebase, which is a FastAPI-based web service that generates sewing patterns from garment images using the Sewformer model.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Core Components](#core-components)
   - [FastAPI Server (main.py)](#fastapi-server-mainpy)
   - [Former Module](#former-module)
   - [Factory Module](#factory-module)
4. [Model Architecture](#model-architecture)
5. [Data Processing Pipeline](#data-processing-pipeline)
6. [Inference Process](#inference-process)
7. [Deployment](#deployment)

## Project Overview

TechPack is a production-ready implementation of the Sewformer model, which can generate sewing patterns from a single garment image. The system is deployed as a REST API service that allows users to upload garment images and receive detailed sewing pattern information.

The application uses a transformer-based architecture (DETR - Detection Transformer) adapted for garment pattern generation. It processes a single image of a garment and outputs the necessary sewing pattern components, including panel shapes, their relative positions, and stitching information.

## Directory Structure

The project is organized into the following main directories:

```
techPack/
├── former/           # Core Sewformer model implementation
│   ├── assets/       # Model checkpoints and data configurations
│   ├── configs/      # Model configuration files
│   ├── data/         # Data processing and dataset classes
│   ├── metrics/      # Loss functions and evaluation metrics
│   ├── models/       # Neural network model definitions
│   └── ...           # Other model-related files
├── Factory/          # Simulation and data generation tools
│   ├── assets/       # Assets for simulation
│   ├── data_generator/ # Data generation scripts
│   ├── packages/     # Utility packages for pattern generation
│   └── ...           # Other factory-related files
├── deployment/       # Deployment configuration files
├── static/           # Static files for serving results
├── outputs/          # Generated output files
├── main.py           # FastAPI application entry point
├── requirements.txt  # Python dependencies
└── ReadMe.md         # Project documentation
```

## Core Components

### FastAPI Server (main.py)

The `main.py` file is the entry point of the application and sets up the FastAPI server with the following key components:

#### Key Functions:

- `load_model()`: Loads the pre-trained Sewformer model, handling GPU detection and fallback to CPU.
- `load_source_appearance(img_path)`: Processes input images for the model, including resizing and normalization.
- `predict(file)`: API endpoint that processes uploaded images and returns sewing pattern data.
- `get_prediction(prediction_id)`: API endpoint to retrieve previously generated predictions.

The server provides RESTful endpoints for uploading garment images and retrieving generated sewing patterns.

### Former Module

The `former/` directory contains the core model implementation, including:

#### Key Files:

- `experiment.py`: Manages experiment tracking, model loading, and configuration.
- `train.py`: Contains the training pipeline for the model.
- `inference.py`: Implements inference functionality for generating predictions.
- `trainer.py`: Implements the training loop and optimization process.

#### Key Classes:

- `ExperimentWrappper`: Manages experiment configurations, model loading, and integration with Weights & Biases (wandb).
- `TrainerDetr`: Handles the training process, including data loading, optimization, and evaluation.

### Factory Module

The `Factory/` directory contains tools for data generation and simulation:

#### Key Directories:

- `data_generator/`: Scripts for generating synthetic data for training.
- `packages/`: Utility packages for pattern generation and manipulation.
  - `pattern/`: Core pattern generation utilities.
  - `mayaqltools/`: Integration with Maya for garment simulation.

## Model Architecture

The Sewformer model is based on the DETR (Detection Transformer) architecture, adapted for garment pattern generation. The key components include:

### GarmentDETRv6 (models/garment_detr_2d.py)

This is the main model class that implements the garment pattern generation network:

- **Backbone**: Extracts features from input images.
- **Panel Transformer**: Processes image features to generate panel representations.
- **Edge Decoder**: Generates panel outlines and stitching information.

The model takes a garment image as input and outputs:
- Panel shapes (outlines)
- Panel positions (translations)
- Panel orientations (rotations)
- Stitching information (which edges should be connected)

### Transformer Architecture

The model uses a transformer architecture with:
- **Encoder**: Processes image features.
- **Decoder**: Generates panel and stitching queries.
- **MLP Heads**: Convert transformer outputs to panel shapes, positions, and stitching information.

## Data Processing Pipeline

### GarmentDetrDataset (data/dataset.py)

This class handles data loading and preprocessing:

- Loads garment images and corresponding ground truth sewing patterns.
- Applies transformations to prepare data for the model.
- Handles data normalization and standardization.

### Data Transforms (data/transforms.py)

Implements various data transformations:
- Image resizing and normalization
- Ground truth standardization
- Data augmentation techniques

### Panel Classes (data/panel_classes.py)

Manages the classification of garment panels:
- Maps panel types to class indices
- Provides utilities for panel classification

## Inference Process

The inference process, implemented in `inference.py` and used by `main.py`, follows these steps:

1. **Image Loading**: Load and preprocess the input garment image.
2. **Feature Extraction**: Extract features using the backbone network.
3. **Panel Generation**: Generate panel shapes, positions, and orientations.
4. **Stitching Prediction**: Predict which panel edges should be stitched together.
5. **Post-processing**: Convert model outputs to a usable sewing pattern format.
6. **Visualization**: Generate visualizations of the predicted sewing pattern.

## Deployment

The application includes deployment configurations for production environments:

- **Systemd Service**: Configuration for running as a Linux service.
- **Docker Support**: Docker configuration for containerized deployment.
- **API Documentation**: Automatic API documentation using FastAPI's built-in Swagger UI.

The application is designed to be deployed as a web service that can be accessed by client applications through its REST API.
