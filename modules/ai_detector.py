"""
AI-generated image detection module.
Uses a combination of CNN model and statistical analysis to detect AI-generated images.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import cv2
from PIL import Image
import os
import io
import requests
from huggingface_hub import hf_hub_download
import torch
from pathlib import Path

# Define AI detection model class
class AIDetector:
    def __init__(self):
        self.model = None
        self.huggingface_model = None
        self._load_models()
    
    def _load_models(self):
        """Load or create the AI detection models"""
        # Create and configure a CNN model based on EfficientNet
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        
        # Add classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # In a real app, we would load pre-trained weights here
        # self.model.load_weights('path_to_weights.h5')
        
        # For HuggingFace model, we would download a pre-trained model
        # We're just showing the code structure for now
        try:
            # Use a pretrained model if available
            model_path = hf_hub_download(repo_id="DivamGupta/synthetic-image-detector", filename="model.pt")
            self.huggingface_model = torch.load(model_path, map_location=torch.device('cpu'))
        except:
            # If download fails, we'll rely on our TensorFlow model
            self.huggingface_model = None
    
    def analyze_image_statistics(self, img_array):
        """
        Analyze statistical patterns in the image that are common in AI-generated images
        Returns a confidence score based on these patterns
        """
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Noise pattern analysis
        noise = gray.astype(float) - cv2.GaussianBlur(gray.astype(float), (5, 5), 0)
        noise_std = np.std(noise)
        
        # Edge coherence analysis
        edges = cv2.Canny(gray, 100, 200)
        edge_coherence = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        # Texture uniformity - calculate GLCM features
        h, w = gray.shape
        glcm = np.zeros((256, 256), dtype=np.uint32)
        for i in range(h-1):
            for j in range(w-1):
                glcm[gray[i, j], gray[i, j+1]] += 1
        
        glcm = glcm / np.sum(glcm)
        texture_uniformity = np.sum(glcm**2)
        
        # Combine factors into a confidence score
        # These weights would be calibrated with proper training data
        ai_score = (
            0.4 * (1 - min(noise_std / 30.0, 1.0)) +  # Low noise often indicates AI
            0.3 * min(edge_coherence * 10, 1.0) +     # Unnaturally coherent edges
            0.3 * min(texture_uniformity * 1000, 1.0)  # Uniform textures
        )
        
        return ai_score * 100  # Return as percentage
    
    def check_dalle_patterns(self, img_array):
        """Check for patterns specific to DALL-E generations"""
        # DALL-E often creates specific artifacts in corners
        # This is a simplified version of what would be a more complex check
        h, w = img_array.shape[:2]
        
        # Check corners for patterns - just a placeholder implementation
        corners = [
            img_array[:10, :10],                # Top-left
            img_array[:10, w-10:],              # Top-right
            img_array[h-10:, :10],              # Bottom-left
            img_array[h-10:, w-10:]             # Bottom-right
        ]
        
        corner_variances = [np.var(corner) for corner in corners]
        avg_variance = np.mean(corner_variances)
        
        # DALL-E images often have similar variance patterns in corners
        # This threshold would be tuned with real DALL-E images
        dalle_score = max(0, min(100, (30 - avg_variance) * 3))
        
        return dalle_score
    
    def predict(self, img_array):
        """Predict if image is AI-generated and return confidence"""
        # Resize for model input
        img_resized = cv2.resize(img_array, (224, 224))
        img_normalized = img_resized / 255.0
        
        # Get statistical analysis score
        stat_score = self.analyze_image_statistics(img_array)
        
        # Get DALL-E specific score
        dalle_score = self.check_dalle_patterns(img_array)
        
        # Get model prediction if available
        model_score = 0
        if self.model:
            # Prepare input for TensorFlow model
            input_tensor = np.expand_dims(img_normalized, axis=0)
            # Get prediction
            model_score = float(self.model.predict(input_tensor)[0][0]) * 100
        
        # Combine scores (with appropriate weighting)
        # In a real implementation, these weights would be calibrated
        combined_score = 0.4 * stat_score + 0.4 * model_score + 0.2 * dalle_score
        
        # Determine if the image is AI-generated based on threshold
        is_ai = combined_score > 60
        
        return is_ai, combined_score

# Create a singleton instance
_detector = None

def detect_ai_image(image_array):
    """
    Detect if an image is AI-generated.
    Returns a tuple (is_ai, confidence)
    """
    global _detector
    
    if _detector is None:
        _detector = AIDetector()
    
    return _detector.predict(image_array)
