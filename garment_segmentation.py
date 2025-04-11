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
import matplotlib.pyplot as plt

class GarmentSegmentationDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, 
                 image_dir, 
                 mask_dir, 
                 batch_size=8, 
                 img_size=(256, 256),
                 augment=True):
        """
        Data generator for garment segmentation
        
        Args:
            image_dir (str): Directory with input images
            mask_dir (str): Directory with corresponding segmentation masks
            batch_size (int): Number of images per batch
            img_size (tuple): Target image dimensions
            augment (bool): Apply data augmentation
        """
        self.image_paths = [
            os.path.join(image_dir, fname) 
            for fname in os.listdir(image_dir) 
            if fname.endswith(('.jpg', '.png', '.jpeg'))
        ]
        
        self.mask_paths = [
            os.path.join(mask_dir, fname) 
            for fname in os.listdir(mask_dir) 
            if fname.endswith(('.jpg', '.png', '.jpeg'))
        ]
        
        self.batch_size = batch_size
        self.img_size = img_size
        
        # Augmentation pipeline
        if augment:
            self.augmentation = A.Compose([
                A.RandomCrop(width=img_size[0], height=img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomFog(p=0.1),
                A.GaussNoise(p=0.1)
            ])
        else:
            self.augmentation = A.Compose([
                A.Resize(height=img_size[0], width=img_size[1])
            ])
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_image_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_mask_paths = self.mask_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_images = np.zeros((len(batch_image_paths), *self.img_size, 3), dtype=np.float32)
        batch_masks = np.zeros((len(batch_mask_paths), *self.img_size, 1), dtype=np.float32)
        
        for i, (img_path, mask_path) in enumerate(zip(batch_image_paths, batch_mask_paths)):
            # Read image and mask
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply augmentation
            augmented = self.augmentation(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            
            # Normalize and prepare
            batch_images[i] = aug_image / 255.0
            batch_masks[i] = aug_mask[..., np.newaxis] / 255.0
        
        return batch_images, batch_masks

def create_unet_model(input_size=(256, 256, 3), num_classes=1):
    """
    Create U-Net model for garment segmentation
    
    Args:
        input_size (tuple): Input image dimensions
        num_classes (int): Number of segmentation classes
    
    Returns:
        tf.keras.Model: Compiled U-Net model
    """
    inputs = tf.keras.layers.Input(input_size)
    
    # Encoder
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bridge
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    
    # Decoder
    up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = tf.keras.layers.concatenate([up5, conv3], axis=-1)
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(up5)
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)
    
    up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
    up6 = tf.keras.layers.concatenate([up6, conv2], axis=-1)
    conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)
    
    up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
    up7 = tf.keras.layers.concatenate([up7, conv1], axis=-1)
    conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)
    
    # Output
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid')(conv7)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_garment_segmentation_model(
    train_image_dir, 
    train_mask_dir, 
    val_image_dir, 
    val_mask_dir,
    epochs=50,
    batch_size=8
):
    """
    Train garment segmentation model
    
    Args:
        train_image_dir (str): Training images directory
        train_mask_dir (str): Training masks directory
        val_image_dir (str): Validation images directory
        val_mask_dir (str): Validation masks directory
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    
    Returns:
        tf.keras.Model: Trained segmentation model
    """
    # Create data generators
    train_generator = GarmentSegmentationDataGenerator(
        train_image_dir, 
        train_mask_dir, 
        batch_size=batch_size,
        augment=True
    )
    
    val_generator = GarmentSegmentationDataGenerator(
        val_image_dir, 
        val_mask_dir, 
        batch_size=batch_size,
        augment=False
    )
    
    # Create model
    model = create_unet_model()
    
    # Model callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/garment_segmentation_model.h5', 
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return model

def visualize_predictions(model, test_image_path):
    """
    Visualize model predictions on a test image
    
    Args:
        model (tf.keras.Model): Trained segmentation model
        test_image_path (str): Path to test image
    """
    # Read and preprocess image
    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (256, 256))
    input_image = resized_image / 255.0
    input_image = np.expand_dims(input_image, axis=0)
    
    # Predict segmentation mask
    prediction = model.predict(input_image)[0]
    predicted_mask = (prediction > 0.5).astype(np.uint8) * 255
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(resized_image)
    plt.subplot(1, 3, 2)
    plt.title('Predicted Mask')
    plt.imshow(predicted_mask.squeeze(), cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('Segmented Image')
    segmented_image = cv2.bitwise_and(resized_image, resized_image, mask=predicted_mask.squeeze())
    plt.imshow(segmented_image)
    plt.show()


if __name__ == "__main__":
    # Train the model
    model = train_garment_segmentation_model(
        'dataset/train/images',
        'dataset/train/masks',
        'dataset/val/images',
        'dataset/val/masks'
    )
    
    # Visualize predictions
    visualize_predictions(model, 'dataset/test/image.jpg')