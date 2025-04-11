#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = 'Ahmad Abdulnasir Shuaib <me@ahmadabdulnasir.com.ng>'
__homepage__ = https://ahmadabdulnasir.com.ng
__copyright__ = 'Copyright (c) 2025, salafi'
__version__ = "0.01t"
"""

import os
import numpy as np
import tensorflow as tf
import cv2
import albumentations as A

class ClothingKeypointDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, 
                 directory, 
                 batch_size=16, 
                 image_size=(224, 224), 
                 keypoints_count=11,
                 augment=True):
        """
        Custom data generator for clothing keypoint detection
        
        Args:
            directory (str): Path to directory containing images and keypoint annotations
            batch_size (int): Number of samples per batch
            image_size (tuple): Resize dimensions for input images
            keypoints_count (int): Number of keypoints to detect
            augment (bool): Whether to apply data augmentation
        """
        self.directory = directory
        self.batch_size = batch_size
        self.image_size = image_size
        self.keypoints_count = keypoints_count
        self.augment = augment
        
        # Load image and keypoint paths
        self.image_paths = []
        self.keypoint_paths = []
        
        for filename in os.listdir(directory):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(directory, filename)
                keypoint_path = os.path.join(directory, filename.replace('.jpg', '.json').replace('.png', '.json'))
                
                if os.path.exists(keypoint_path):
                    self.image_paths.append(image_path)
                    self.keypoint_paths.append(keypoint_path)
        
        # Shuffle data
        indices = np.random.permutation(len(self.image_paths))
        self.image_paths = [self.image_paths[i] for i in indices]
        self.keypoint_paths = [self.keypoint_paths[i] for i in indices]
        
        # Augmentation pipeline
        if augment:
            self.augmentation = A.Compose([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.RandomSizedCrop(
                    min_max_height=(int(image_size[0]*0.8), image_size[0]), 
                    height=image_size[0], 
                    width=image_size[1], 
                    p=0.3
                )
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            self.augmentation = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1])
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        """
        Generate one batch of data
        
        Returns:
            X (numpy.array): Batch of preprocessed images
            y (numpy.array): Batch of keypoint coordinates
        """
        # Select batch of images and keypoint files
        batch_image_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_keypoint_paths = self.keypoint_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Initialize batch arrays
        X = np.zeros((len(batch_image_paths), *self.image_size, 3), dtype=np.float32)
        y = np.zeros((len(batch_image_paths), self.keypoints_count * 2), dtype=np.float32)
        
        # Process each image in batch
        for i, (img_path, kp_path) in enumerate(zip(batch_image_paths, batch_keypoint_paths)):
            # Read image
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load keypoints from JSON
            keypoints = self._load_keypoints(kp_path)
            
            # Apply augmentation
            augmented = self.augmentation(image=image, keypoints=keypoints)
            aug_image = augmented['image']
            aug_keypoints = augmented['keypoints']
            
            # Normalize and resize
            X[i] = aug_image / 255.0
            
            # Flatten keypoints and normalize to 0-1 range
            flat_keypoints = [coord for kp in aug_keypoints for coord in kp]
            y[i] = np.array(flat_keypoints) / np.array([aug_image.shape[1], aug_image.shape[0]] * self.keypoints_count)
        
        return X, y
    
    def _load_keypoints(self, keypoint_path):
        """
        Load keypoints from JSON file
        
        Args:
            keypoint_path (str): Path to JSON file with keypoint annotations
        
        Returns:
            list: List of (x, y) keypoint coordinates
        """
        import json
        
        with open(keypoint_path, 'r') as f:
            keypoint_data = json.load(f)
        
        # Extract keypoint coordinates
        keypoints = [
            (kp['x'], kp['y']) for kp in keypoint_data['keypoints']
        ]
        
        return keypoints
    
    def on_epoch_end(self):
        """
        Shuffle data at the end of each epoch
        """
        indices = np.random.permutation(len(self.image_paths))
        self.image_paths = [self.image_paths[i] for i in indices]
        self.keypoint_paths = [self.keypoint_paths[i] for i in indices]