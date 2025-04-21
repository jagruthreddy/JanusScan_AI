"""
Device detection module.
Identifies smartphone models that captured an image based on metadata and image characteristics.
"""

import numpy as np
import cv2
from PIL import Image, ExifTags
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import re

# Common camera model strings for popular smartphone brands
IPHONE_PATTERNS = [
    r'iphone', r'apple', r'iphone (\d+)', r'iphone (\d+) pro'
]

SAMSUNG_PATTERNS = [
    r'samsung', r'sm-\w+', r'galaxy s(\d+)', r'galaxy note(\d+)', r'galaxy a(\d+)'
]

GOOGLE_PATTERNS = [
    r'pixel', r'google', r'pixel (\d+)', r'pixel (\d+) pro'
]

class DeviceDetector:
    def __init__(self):
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load or create the device classification model"""
        # Create and configure CNN model based on MobileNetV2
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        
        # Add classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(4, activation='softmax')(x)  # 4 classes: iPhone, Samsung, Google, Other
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # In a real app, we would load pre-trained weights here
        # self.model.load_weights('path_to_weights.h5')
    
    def detect_from_metadata(self, metadata):
        """
        Detect device from image metadata
        Returns (device_name, confidence)
        """
        if not metadata:
            return "Unknown", 0
        
        # Look for device information in various metadata fields
        make = metadata.get('Make', '')
        model = metadata.get('Model', '')
        software = metadata.get('Software', '')
        
        device_info = f"{make} {model} {software}".lower()
        
        # Check for iPhone
        for pattern in IPHONE_PATTERNS:
            if re.search(pattern, device_info):
                # Extract model number if available
                match = re.search(r'iphone (\d+)', device_info)
                if match:
                    return f"iPhone {match.group(1)}", 95
                return "iPhone", 90
        
        # Check for Samsung
        for pattern in SAMSUNG_PATTERNS:
            if re.search(pattern, device_info):
                # Extract model if available
                match = re.search(r'galaxy ([\w\d]+)', device_info)
                if match:
                    return f"Samsung Galaxy {match.group(1).upper()}", 95
                return "Samsung Galaxy", 90
        
        # Check for Google Pixel
        for pattern in GOOGLE_PATTERNS:
            if re.search(pattern, device_info):
                # Extract model number if available
                match = re.search(r'pixel (\d+)', device_info)
                if match:
                    return f"Google Pixel {match.group(1)}", 95
                return "Google Pixel", 90
        
        # Check if there's any make/model but not matched above
        if make or model:
            return f"{make} {model}".strip(), 85
        
        return "Unknown", 0
    
    def analyze_noise_pattern(self, img_array):
        """
        Analyze image noise patterns which can differ between phone models
        Returns a feature vector
        """
        # Convert to YUV color space which better separates luminance from chrominance
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
            y_channel = yuv[:,:,0]
        else:
            y_channel = img_array
        
        # Apply noise extraction filter (simplified for this example)
        noise = y_channel.astype(float) - cv2.GaussianBlur(y_channel.astype(float), (3, 3), 0)
        
        # Compute noise statistics
        noise_mean = np.mean(noise)
        noise_std = np.std(noise)
        noise_skewness = np.mean(((noise - noise_mean) / noise_std) ** 3)
        noise_kurtosis = np.mean(((noise - noise_mean) / noise_std) ** 4) - 3
        
        # Apply FFT to analyze frequency components of noise
        fft = np.fft.fft2(noise)
        fft_mag = np.abs(np.fft.fftshift(fft))
        
        # Extract features from frequency domain
        h, w = fft_mag.shape
        center_h, center_w = h // 2, w // 2
        
        # Analyze different regions of frequency spectrum
        low_freq = np.mean(fft_mag[center_h-10:center_h+10, center_w-10:center_w+10])
        mid_freq = np.mean(fft_mag[center_h-50:center_h+50, center_w-50:center_w+50]) - low_freq
        high_freq = np.mean(fft_mag) - mid_freq - low_freq
        
        return np.array([noise_mean, noise_std, noise_skewness, noise_kurtosis, 
                         low_freq, mid_freq, high_freq])
    
    def predict_from_image(self, img_array):
        """Predict device from image characteristics"""
        # Resize for model input
        img_resized = cv2.resize(img_array, (224, 224))
        img_normalized = img_resized / 255.0
        
        # Get noise pattern features
        noise_features = self.analyze_noise_pattern(img_array)
        
        # Prepare input for the model
        input_tensor = np.expand_dims(img_normalized, axis=0)
        
        # Get model prediction
        if self.model:
            preds = self.model.predict(input_tensor)[0]
            
            # Map prediction indices to devices
            devices = ["iPhone", "Samsung Galaxy", "Google Pixel", "Other"]
            pred_idx = np.argmax(preds)
            confidence = float(preds[pred_idx]) * 100
            
            if confidence > 40:
                return devices[pred_idx], confidence
        
        # Fallback to a heuristic approach using noise features
        # This is a simplified decision tree that would be trained with real data
        if noise_features[1] < 5 and noise_features[4] > 1000:
            return "iPhone", 60
        elif noise_features[1] > 8 and noise_features[6] > 500:
            return "Samsung Galaxy", 55
        elif 5 <= noise_features[1] <= 8 and noise_features[5] > 800:
            return "Google Pixel", 50
        
        return "Unknown", 30
    
    def detect(self, img_array, metadata):
        """
        Combined detection using both metadata and image characteristics
        Returns (device_name, confidence)
        """
        # First try metadata detection
        meta_device, meta_confidence = self.detect_from_metadata(metadata)
        
        # If high confidence from metadata, return that result
        if meta_confidence >= 85:
            return meta_device, meta_confidence
        
        # Otherwise, use image-based prediction
        img_device, img_confidence = self.predict_from_image(img_array)
        
        # If metadata gave some signal, combine the results
        if meta_confidence > 0:
            # If devices match, increase confidence
            if meta_device == img_device:
                combined_confidence = max(meta_confidence, img_confidence) + 5
                return meta_device, min(combined_confidence, 95)
            
            # If different devices but both with decent confidence, go with metadata
            if meta_confidence >= 50 and img_confidence >= 50:
                return meta_device, meta_confidence
            
            # Otherwise use highest confidence result
            if meta_confidence > img_confidence:
                return meta_device, meta_confidence
        
        return img_device, img_confidence

# Create a singleton instance
_detector = None

def detect_device(image_array, metadata):
    """
    Detect which device captured an image.
    Returns a tuple (device_name, confidence)
    """
    global _detector
    
    if _detector is None:
        _detector = DeviceDetector()
    
    return _detector.detect(image_array, metadata)
