"""
Edit detector module.
Detects and visualizes edits made to images using advanced computer vision techniques.
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature, measure, segmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
import io

class EditDetector:
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load or create the edit detection model"""
        # In a real implementation, we'd load a pre-trained model
        # For this example, we'll rely on classical CV techniques
        pass
    
    def detect_noise_inconsistency(self, img_array):
        """
        Detects areas with inconsistent noise patterns that may indicate edits
        Returns a heatmap highlighting potential edited regions
        """
        # Convert to grayscale if color image
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Extract noise pattern
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(float) - blur.astype(float)
        
        # Compute local noise statistics
        noise_var = ndimage.generic_filter(noise**2, np.mean, size=11)
        
        # Normalize the variance map
        noise_var = (noise_var - np.min(noise_var)) / (np.max(noise_var) - np.min(noise_var) + 1e-5)
        
        # Apply threshold to identify regions with abnormal noise patterns
        _, noise_mask = cv2.threshold(noise_var, 0.7, 1, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        noise_mask = cv2.morphologyEx(noise_mask, cv2.MORPH_OPEN, kernel)
        noise_mask = cv2.morphologyEx(noise_mask, cv2.MORPH_CLOSE, kernel)
        
        return noise_mask
    
    def detect_compression_artifacts(self, img_array):
        """
        Detects inconsistent compression artifacts which may indicate edits
        Returns a mask of regions with unusual compression patterns
        """
        # Convert to YCrCb color space which is commonly used in JPEG compression
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
            y_channel = ycrcb[:,:,0]
        else:
            y_channel = img_array
        
        # Apply DCT (Discrete Cosine Transform) to 8x8 blocks
        h, w = y_channel.shape
        block_size = 8
        
        # Create an empty map to store block statistics
        dct_stats = np.zeros((h // block_size, w // block_size))
        
        # Process each 8x8 block
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = y_channel[i:i+block_size, j:j+block_size].astype(float)
                
                # Apply DCT
                dct_block = cv2.dct(block)
                
                # Compute statistics on DCT coefficients that reveal compression
                # Focus on high-frequency coefficients which are most affected by JPEG
                high_freq = dct_block[4:, 4:]
                dct_stats[i // block_size, j // block_size] = np.std(high_freq)
        
        # Normalize statistics
        dct_stats = (dct_stats - np.min(dct_stats)) / (np.max(dct_stats) - np.min(dct_stats) + 1e-5)
        
        # Resize back to original dimensions
        dct_map = cv2.resize(dct_stats, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Threshold to find inconsistent areas
        _, dct_mask = cv2.threshold(dct_map, 0.65, 1, cv2.THRESH_BINARY)
        
        # Clean up mask
        kernel = np.ones((7, 7), np.uint8)
        dct_mask = cv2.morphologyEx(dct_mask, cv2.MORPH_OPEN, kernel)
        dct_mask = cv2.morphologyEx(dct_mask, cv2.MORPH_CLOSE, kernel)
        
        return dct_mask
    
    def detect_edge_inconsistencies(self, img_array):
        """
        Detects inconsistent edge patterns that may indicate cloning or splicing
        Returns a mask highlighting regions with unusual edge transitions
        """
        # Convert to grayscale if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply edge density analysis in local regions
        edge_density = ndimage.generic_filter(edges.astype(float) / 255, np.mean, size=25)
        
        # Get edge consistency by analyzing gradient directions
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_dir = np.arctan2(gradient_y, gradient_x)
        
        # Calculate local gradient direction consistency
        dir_consistency = ndimage.generic_filter(
            gradient_dir, 
            lambda x: np.var(np.sin(x)) + np.var(np.cos(x)), 
            size=15
        )
        
        # Normalize consistency map
        dir_consistency = (dir_consistency - np.min(dir_consistency)) / (np.max(dir_consistency) - np.min(dir_consistency) + 1e-5)
        
        # Combine edge density with direction consistency
        edge_inconsistency = edge_density * dir_consistency
        
        # Threshold to get binary mask
        _, edge_mask = cv2.threshold(edge_inconsistency, 0.7, 1, cv2.THRESH_BINARY)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, kernel)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
        
        return edge_mask
    
    def detect_color_inconsistencies(self, img_array):
        """
        Detects inconsistent color patterns that may indicate color editing
        Returns a mask highlighting regions with unusual color transitions
        """
        # Need color image for this analysis
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            return np.zeros(img_array.shape, dtype=np.float32)
        
        # Convert to LAB color space which better represents perceptual color differences
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Split channels
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Compute local color statistics for each channel
        l_var = ndimage.generic_filter(l_channel, np.var, size=15)
        a_var = ndimage.generic_filter(a_channel, np.var, size=15)
        b_var = ndimage.generic_filter(b_channel, np.var, size=15)
        
        # Combine variance maps
        color_var = (l_var + a_var + b_var) / 3
        
        # Normalize
        color_var = (color_var - np.min(color_var)) / (np.max(color_var) - np.min(color_var) + 1e-5)
        
        # Threshold to get binary mask
        _, color_mask = cv2.threshold(color_var, 0.75, 1, cv2.THRESH_BINARY)
        
        # Clean up mask
        kernel = np.ones((7, 7), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        return color_mask
    
    def analyze_edited_regions(self, img_array, combined_mask):
        """
        Analyze the detected edited regions to provide descriptive information
        Returns a textual description of edits
        """
        # Find contours in the mask
        contours, _ = cv2.findContours(combined_mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        if not contours:
            return "No significant edits detected."
        
        # Analyze each edited region
        edit_descriptions = []
        
        for i, contour in enumerate(contours):
            # Get region bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Get region mask
            region_mask = np.zeros_like(combined_mask)
            cv2.drawContours(region_mask, [contour], 0, 1, -1)
            
            # Get the region from the original image
            region = img_array[y:y+h, x:x+w]
            
            # Basic position description
            if y < img_array.shape[0] * 0.33:
                v_pos = "top"
            elif y > img_array.shape[0] * 0.66:
                v_pos = "bottom"
            else:
                v_pos = "middle"
                
            if x < img_array.shape[1] * 0.33:
                h_pos = "left"
            elif x > img_array.shape[1] * 0.66:
                h_pos = "right"
            else:
                h_pos = "center"
            
            position = f"{v_pos} {h_pos}"
            
            # Determine area percentage
            area_percentage = (cv2.contourArea(contour) / 
                              (img_array.shape[0] * img_array.shape[1])) * 100
            
            # Try to determine edit type
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                region_lab = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)
                l_mean, a_mean, b_mean = cv2.mean(region_lab)[:3]
                
                # Check if it's likely a color adjustment
                color_adj = False
                if a_mean > 135 or a_mean < 110 or b_mean > 140 or b_mean < 100:
                    edit_type = "color adjustment"
                    color_adj = True
                elif w > img_array.shape[1] * 0.5 and h > img_array.shape[0] * 0.5:
                    edit_type = "global adjustment"
                elif region_mask.shape == combined_mask.shape:  # If we're checking the whole mask
                    noise_mask = self.detect_noise_inconsistency(img_array)
                    compression_mask = self.detect_compression_artifacts(img_array)
                    if np.sum(noise_mask * region_mask) > np.sum(compression_mask * region_mask):
                        edit_type = "object insertion or removal"
                    else:
                        edit_type = "splice or composite"
                else:
                    edit_type = "local edit"
            else:
                edit_type = "local adjustment"
            
            # Format description
            if area_percentage < 5:
                size_desc = "small"
            elif area_percentage < 20:
                size_desc = "medium-sized"
            else:
                size_desc = "large"
            
            edit_descriptions.append(
                f"• {size_desc.capitalize()} {edit_type} detected in the {position} region " +
                f"({area_percentage:.1f}% of image area)"
            )
        
        # Return combined description
        if len(edit_descriptions) == 1:
            return edit_descriptions[0]
        else:
            return "Multiple edits detected:\n" + "\n".join(edit_descriptions)
    
    def detect(self, img_array):
        """
        Main method to detect edits in an image
        Returns (is_edited, edit_map, edit_description)
        """
        # Run all detection methods
        noise_mask = self.detect_noise_inconsistency(img_array)
        compress_mask = self.detect_compression_artifacts(img_array)
        edge_mask = self.detect_edge_inconsistencies(img_array)
        color_mask = self.detect_color_inconsistencies(img_array)
        
        # Combine masks with different weights based on reliability
        combined_mask = (0.3 * noise_mask + 
                         0.25 * compress_mask + 
                         0.25 * edge_mask + 
                         0.2 * color_mask)
        
        # Normalize combined mask
        combined_mask = (combined_mask - np.min(combined_mask)) / (np.max(combined_mask) - np.min(combined_mask) + 1e-5)
        
        # Create a colored heatmap
        heatmap = np.zeros((combined_mask.shape[0], combined_mask.shape[1], 3), dtype=np.uint8)
        heatmap[:,:,0] = (1 - combined_mask) * 255  # Blue channel (inverted)
        heatmap[:,:,1] = np.zeros_like(combined_mask)  # Green channel
        heatmap[:,:,2] = combined_mask * 255  # Red channel
        
        # Overlay on grayscale version of original image for context
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()
        
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(gray_rgb, 0.7, heatmap, 0.3, 0)
        
        # Determine if the image is edited
        edit_coverage = np.sum(combined_mask > 0.5) / combined_mask.size
        is_edited = edit_coverage > 0.02  # Consider edited if more than 2% of pixels are flagged
        
        # Generate description of edits
        if is_edited:
            edit_description = self.analyze_edited_regions(img_array, combined_mask > 0.5)
        else:
            edit_description = "No significant edits detected."
        
        return is_edited, overlay, edit_description

# Create a singleton instance
_detector = None

def detect_edits(image_array):
    """
    Detect edits in an image.
    Returns a tuple (is_edited, edit_map_image, edit_description)
    """
    global _detector
    
    if _detector is None:
        _detector = EditDetector()
    
    return _detector.detect(image_array)

def visualize_edits(image_array, edit_map):
    """Create a visual representation of edits for display"""
    # This is a helper function to generate a visualization
    # The implementation is already included in the detect method above
    # This function is here for API completeness
    return edit_map
