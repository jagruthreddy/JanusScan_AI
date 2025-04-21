"""
Text detector module.
Detects AI-generated text in images, with specific focus on OpenAI text generation.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pytesseract
import re
import torch
from PIL import Image

class TextDetector:
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load or create the text detection model"""
        # Create and configure a CNN model for detecting AI-generated text
        # In a real implementation, this would load a pre-trained model
        try:
            # Simple ResNet-based model
            base_model = ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3)
            )
            
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            
            self.model = Model(inputs=base_model.input, outputs=predictions)
            
            # In a real app, we would load pre-trained weights here
            # self.model.load_weights('path_to_weights.h5')
        except:
            # If model creation fails, we'll rely on rule-based methods
            self.model = None
    
    def extract_text(self, img_array):
        """Extract text from image using OCR"""
        try:
            # Convert to grayscale if needed
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply preprocessing to improve OCR accuracy
            # Adaptive thresholding to deal with different lighting conditions
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
            
            # Convert back to PIL Image for Tesseract
            pil_img = Image.fromarray(denoised)
            
            # Extract text using Tesseract OCR
            # In a real implementation, you would need Tesseract installed
            # text = pytesseract.image_to_string(pil_img)
            
            # For this example, we'll simulate extracted text
            # This would normally come from pytesseract
            text = self._simulate_text_extraction(img_array)
            
            return text
        except Exception as e:
            return ""
    
    def _simulate_text_extraction(self, img_array):
        """
        Simulates text extraction for demo purposes
        In a real application, this would use pytesseract
        """
        # Use very basic image analysis to determine if text might be present
        # This is just a placeholder for demonstration
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Edge detection to find potential text regions
        edges = cv2.Canny(gray, 100, 200)
        text_likely = np.sum(edges) > (gray.shape[0] * gray.shape[1] * 0.05)
        
        if not text_likely:
            return ""
        
        # Return simulated text for demo purposes
        # In real app, this would be the actual OCR result
        return "Sample text extracted from image. This would be the actual text detected by OCR."
    
    def analyze_text_patterns(self, text):
        """
        Analyze text patterns to identify AI-generated text
        Returns (is_ai_text, confidence, features)
        """
        if not text or len(text) < 10:
            return False, 0, {}
        
        features = {}
        
        # Calculate basic text statistics
        words = text.split()
        features['word_count'] = len(words)
        features['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0
        features['unique_words'] = len(set(words)) / len(words) if words else 0
        
        # Check for repetition patterns common in AI text
        repetition_score = 0
        for i in range(len(words) - 2):
            if words[i] == words[i+2]:
                repetition_score += 1
        features['repetition_score'] = repetition_score / len(words) if words else 0
        
        # Check for unusual clarity and precision (common in DALL-E text generation)
        clarity_markers = ['exactly', 'precise', 'specific', 'detailed', 'perfect']
        features['clarity_score'] = sum(1 for word in words if word.lower() in clarity_markers) / len(words) if words else 0
        
        # Check for OpenAI's typical text patterns 
        # OpenAI models often generate very well-structured, grammatically correct text
        # with specific transition patterns
        transition_markers = ['however', 'therefore', 'thus', 'additionally', 'moreover', 'furthermore']
        features['transition_score'] = sum(1 for word in words if word.lower() in transition_markers) / len(words) if words else 0
        
        # Check for unnaturally consistent punctuation and capitalization
        punct_consistency = len(re.findall(r'[.,;:]', text)) / len(text) if text else 0
        features['punct_consistency'] = punct_consistency
        
        # Calculate AI text likelihood based on features
        ai_score = (
            0.2 * min(features['unique_words'] * 2, 1.0) +  # Lower lexical diversity often indicates AI
            0.3 * min(features['repetition_score'] * 5, 1.0) +  # Repetition patterns
            0.2 * min(features['clarity_score'] * 4, 1.0) +  # Unusual clarity markers
            0.2 * min(features['transition_score'] * 5, 1.0) +  # Transition word patterns
            0.1 * min(features['punct_consistency'] * 10, 1.0)  # Punctuation consistency
        )
        
        return ai_score > 0.6, ai_score * 100, features
    
    def detect_openai_text_patterns(self, text):
        """
        Specifically detect patterns common in OpenAI text generation
        Focus on DALL-E, GPT-generated text in images
        """
        if not text or len(text) < 10:
            return False, 0
        
        # OpenAI models often produce text with these characteristics
        characteristics = [
            # Very consistent capitalization
            abs(sum(1 for c in text if c.isupper()) / len(text) - 0.15) < 0.05,
            
            # Specific punctuation density
            0.05 < len(re.findall(r'[.,;:]', text)) / len(text) < 0.15,
            
            # Absence of typographical errors (harder to detect programmatically)
            # We'll use proxy metrics like consistent spacing
            len(re.findall(r' {2,}', text)) == 0,
            
            # Consistent quotation usage
            len(re.findall(r'"', text)) % 2 == 0,
            
            # Absence of unique human typing patterns like multiple exclamation marks
            len(re.findall(r'!{2,}', text)) == 0,
            
            # DALL-E specific: presence of descriptive phrases
            any(phrase in text.lower() for phrase in ['high quality', 'detailed', '4k', 'realistic', 'professional']),
            
            # DALL-E specific: style descriptors
            any(style in text.lower() for phrase in ['style of', 'inspired by', 'looking like', 'resembling']),
        ]
        
        # Calculate score based on how many characteristics match
        openai_score = sum(1 for char in characteristics if char) / len(characteristics)
        
        return openai_score > 0.6, openai_score * 100
    
    def analyze_text_regions(self, img_array):
        """
        Identify and analyze text regions in the image
        Focus on detecting synthetic/AI-generated text
        """
        try:
            # Convert to grayscale if needed
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply MSER (Maximally Stable Extremal Regions) to detect text regions
            # This is a common technique for text detection
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            
            if not regions:
                return False, 0
            
            # Create mask of text regions
            text_mask = np.zeros_like(gray)
            for region in regions:
                hull = cv2.convexHull(region.reshape(-1, 1, 2))
                cv2.drawContours(text_mask, [hull], 0, 255, -1)
            
            # Filter text regions based on size and aspect ratio
            contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            text_regions = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter based on size and aspect ratio
                if w > 10 and h > 10 and 0.1 < w/h < 10:
                    text_regions.append((x, y, w, h))
            
            if not text_regions:
                return False, 0
            
            # Check each region for unnatural characteristics 
            # AI-generated text often has unnaturally perfect alignment and spacing
            heights = [h for _, _, _, h in text_regions]
            mean_height = np.mean(heights)
            height_variance = np.std(heights) / mean_height if mean_height > 0 else 0
            
            # AI text often has very consistent heights
            ai_height_indicator = height_variance < 0.15
            
            # Check horizontal alignment
            lefts = [x for x, _, _, _ in text_regions]
            left_variance = np.std(lefts) / np.mean(lefts) if lefts else 0
            
            # AI text often has perfect alignment
            ai_alignment_indicator = left_variance < 0.1
            
            # Combine indicators
            ai_region_score = (ai_height_indicator * 0.6 + ai_alignment_indicator * 0.4) * 100
            
            return ai_region_score > 60, ai_region_score
        except Exception as e:
            return False, 0
    
    def detect(self, img_array):
        """
        Main method to detect AI-generated text in images
        Returns (has_ai_text, confidence)
        """
        # Extract text from image
        text = self.extract_text(img_array)
        
        # If no significant text is found, return negative result
        if not text or len(text) < 10:
            return False, 0
        
        # Analyze text patterns
        text_is_ai, text_score, _ = self.analyze_text_patterns(text)
        
        # Check for OpenAI-specific patterns
        openai_is_ai, openai_score = self.detect_openai_text_patterns(text)
        
        # Analyze text regions in the image
        region_is_ai, region_score = self.analyze_text_regions(img_array)
        
        # Combine scores with weights
        combined_score = (0.4 * text_score + 0.4 * openai_score + 0.2 * region_score)
        
        # Determine if the text is AI-generated based on threshold
        is_ai_text = combined_score > 60
        
        return is_ai_text, combined_score

# Create a singleton instance
_detector = None

def detect_ai_text(image_array):
    """
    Detect if text in an image is AI-generated, with focus on OpenAI text generation.
    Returns a tuple (has_ai_text, confidence)
    """
    global _detector
    
    if _detector is None:
        _detector = TextDetector()
    
    return _detector.detect(image_array)
