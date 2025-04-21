"""
Metadata analyzer module.
Extracts, analyzes, and recovers metadata from images, particularly focusing on
images that have lost metadata through messaging platforms.
"""

import exifread
import os
import numpy as np
import cv2
from PIL import Image
import io
import re
import hashlib
import json
import base64
from datetime import datetime
import requests

class MetadataAnalyzer:
    def __init__(self):
        self.metadata_signatures = {
            'iphone': {
                'noise_profile': [0.012, 0.035, 0.045],
                'color_profile': 'Apple',
                'compression': 'progressive'
            },
            'samsung': {
                'noise_profile': [0.015, 0.042, 0.052],
                'color_profile': 'Samsung',
                'compression': 'baseline'
            },
            'google_pixel': {
                'noise_profile': [0.010, 0.038, 0.048],
                'color_profile': 'Google',
                'compression': 'progressive'
            },
            'dall_e': {
                'noise_profile': [0.005, 0.020, 0.025],
                'color_profile': 'AI',
                'compression': 'baseline'
            },
            'midjourney': {
                'noise_profile': [0.004, 0.018, 0.028],
                'color_profile': 'AI',
                'compression': 'baseline'
            },
            'stable_diffusion': {
                'noise_profile': [0.006, 0.022, 0.030],
                'color_profile': 'AI',
                'compression': 'baseline'
            }
        }
    
    def extract_exif(self, image_path):
        """Extract EXIF metadata from an image file"""
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                
                # Convert exifread tags to a more usable dictionary
                metadata = {}
                for tag, value in tags.items():
                    # Skip some binary fields that aren't useful for analysis
                    if 'thumbnail' in tag.lower():
                        continue
                    
                    # Clean up tag names
                    key = tag.split("EXIF")[1] if "EXIF" in tag else tag
                    key = key.strip()
                    
                    # Convert value to string
                    metadata[key] = str(value)
                
                return metadata
        except Exception as e:
            return {}
    
    def extract_iptc(self, image_path):
        """Extract IPTC metadata"""
        try:
            # This would use a proper IPTC extraction library in a real implementation
            # For this example, we'll return dummy data
            return {}
        except Exception as e:
            return {}
    
    def extract_xmp(self, image_path):
        """Extract XMP metadata"""
        try:
            # This would use a proper XMP extraction library in a real implementation
            xmp_data = {}
            
            # Try to read the file and look for XMP data
            with open(image_path, 'rb') as f:
                content = f.read()
                
                # Look for XMP packet markers
                xmp_start = content.find(b'<x:xmpmeta')
                xmp_end = content.find(b'</x:xmpmeta')
                
                if xmp_start != -1 and xmp_end != -1:
                    xmp_content = content[xmp_start:xmp_end+12].decode('utf-8', errors='ignore')
                    
                    # Extract basic creator information
                    creator_match = re.search(r'<dc:creator>(.*?)</dc:creator>', xmp_content)
                    if creator_match:
                        xmp_data['Creator'] = creator_match.group(1)
                    
                    # Extract software information
                    software_match = re.search(r'<xmp:CreatorTool>(.*?)</xmp:CreatorTool>', xmp_content)
                    if software_match:
                        xmp_data['Software'] = software_match.group(1)
                    
                    # Look for AI indicators
                    if 'dall-e' in xmp_content.lower() or 'openai' in xmp_content.lower():
                        xmp_data['AI_Generated'] = 'DALL-E (OpenAI)'
                    elif 'midjourney' in xmp_content.lower():
                        xmp_data['AI_Generated'] = 'Midjourney'
                    elif 'stable diffusion' in xmp_content.lower():
                        xmp_data['AI_Generated'] = 'Stable Diffusion'
            
            return xmp_data
        except Exception as e:
            return {}
    
    def extract_image_properties(self, image_path):
        """Extract basic image properties"""
        try:
            with Image.open(image_path) as img:
                properties = {
                    'Format': img.format,
                    'Width': img.width,
                    'Height': img.height,
                    'Mode': img.mode,
                    'FileSize': os.path.getsize(image_path)
                }
                
                # Try to extract creation time from file metadata
                creation_time = os.path.getctime(image_path)
                properties['CreationTime'] = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
                
                return properties
        except Exception as e:
            return {}
    
    def analyze_jpeg_signature(self, image_path):
        """Analyze JPEG signature which can provide source information"""
        try:
            with open(image_path, 'rb') as f:
                # Read first few bytes to check header
                header = f.read(12)
                
                # Check for JPEG signature
                if header[:2] != b'\xFF\xD8':
                    return {}
                
                # Read more of the file to analyze JPEG structure
                f.seek(0)
                content = f.read(2048)  # Read a chunk to analyze markers
                
                # Check for JFIF or Exif APP markers
                has_jfif = b'JFIF' in content
                has_exif = b'Exif' in content
                
                # Check compression type
                # This is a simplified approach - real implementation would parse SOF markers
                is_progressive = b'\xFF\xC2' in content
                is_baseline = b'\xFF\xC0' in content
                
                # Check for camera-specific markers
                has_apple_marker = b'Apple' in content
                has_samsung_marker = b'SAMSUNG' in content or b'SM-' in content
                has_google_marker = b'Google' in content or b'Pixel' in content
                
                # Check for C2PA markers
                has_c2pa_marker = b'c2pa' in content or b'C2PA' in content or b'stds.adobe.com/c2pa' in content
                
                jpeg_info = {
                    'HasJFIF': has_jfif,
                    'HasExif': has_exif,
                    'Progressive': is_progressive,
                    'Baseline': is_baseline,
                    'HasC2PA': has_c2pa_marker
                }
                
                # Add source hints based on markers
                if has_apple_marker:
                    jpeg_info['SourceHint'] = 'Apple iPhone'
                elif has_samsung_marker:
                    jpeg_info['SourceHint'] = 'Samsung'
                elif has_google_marker:
                    jpeg_info['SourceHint'] = 'Google Pixel'
                
                return jpeg_info
        except Exception as e:
            return {}
    
    def extract_c2pa_metadata(self, image_path):
        """Extract C2PA metadata specifically used by OpenAI and other providers"""
        try:
            # Look for C2PA data in the image file
            with open(image_path, 'rb') as f:
                content = f.read()
                
                # C2PA data is often stored in a JUMBF box
                # This is a simplified detection approach
                c2pa_data = {}
                
                # Look for C2PA manifest markers
                c2pa_start = content.find(b'c2pa')
                c2pa_json_start = content.find(b'{"claim":')
                
                if c2pa_start != -1 or c2pa_json_start != -1:
                    c2pa_data['HasC2PAManifest'] = True
                    
                    # Check for OpenAI specific identifiers
                    if b'openai' in content.lower() or b'dall-e' in content.lower():
                        c2pa_data['Provider'] = 'OpenAI'
                        c2pa_data['AI_Generated'] = True
                        
                        # Try to extract model info
                        if b'dall-e-3' in content.lower():
                            c2pa_data['Model'] = 'DALL-E 3'
                        elif b'dall-e-2' in content.lower():
                            c2pa_data['Model'] = 'DALL-E 2'
                    
                    # Look for creation information
                    creation_match = re.search(rb'"created":\s*"([^"]+)"', content)
                    if creation_match:
                        c2pa_data['CreationTime'] = creation_match.group(1).decode('utf-8', errors='ignore')
                    
                    # Look for prompt information (crucial for OpenAI)
                    prompt_match = re.search(rb'"prompt":\s*"([^"]+)"', content)
                    if prompt_match:
                        c2pa_data['Prompt'] = prompt_match.group(1).decode('utf-8', errors='ignore')
                
                return c2pa_data
        except Exception as e:
            return {}
    
    def extract_metadata(self, image_path):
        """Extract all available metadata"""
        # Combine all extraction methods
        exif_data = self.extract_exif(image_path)
        iptc_data = self.extract_iptc(image_path)
        xmp_data = self.extract_xmp(image_path)
        props = self.extract_image_properties(image_path)
        jpeg_info = self.analyze_jpeg_signature(image_path)
        c2pa_data = self.extract_c2pa_metadata(image_path)
        
        # Merge all sources
        all_metadata = {
            **props,
            **exif_data,
            **iptc_data,
            **xmp_data,
            **c2pa_data
        }
        
        # Add JPEG signature info
        if jpeg_info.get('SourceHint'):
            all_metadata['SourceHint'] = jpeg_info['SourceHint']
        
        # Add compression info
        if jpeg_info.get('Progressive'):
            all_metadata['Compression'] = 'Progressive JPEG'
        elif jpeg_info.get('Baseline'):
            all_metadata['Compression'] = 'Baseline JPEG'
        
        return all_metadata
    
    def analyze_noise_pattern(self, img_array):
        """Analyze image noise pattern to identify source device"""
        # Convert to YUV color space
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
            y_channel = yuv[:,:,0]
        else:
            y_channel = img_array
        
        # Extract noise using wavelet-based denoising
        # This is a simplified approach
        blurred = cv2.GaussianBlur(y_channel, (5, 5), 0)
        noise = y_channel.astype(float) - blurred.astype(float)
        
        # Calculate noise statistics at different frequency levels
        noise_low = cv2.blur(noise, (15, 15))
        noise_mid = noise - noise_low
        noise_high = noise - cv2.blur(noise, (3, 3))
        
        # Compute standard deviations at each level
        low_std = np.std(noise_low)
        mid_std = np.std(noise_mid)
        high_std = np.std(noise_high)
        
        return [low_std, mid_std, high_std]
    
    def analyze_color_profile(self, img_array):
        """Analyze color profile to identify source device"""
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            return "Unknown"
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Split channels and compute histograms
        l, a, b = cv2.split(lab)
        l_hist = cv2.calcHist([l], [0], None, [256], [0, 256])
        a_hist = cv2.calcHist([a], [0], None, [256], [0, 256])
        b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
        
        # Normalize histograms
        l_hist = l_hist / np.sum(l_hist)
        a_hist = a_hist / np.sum(a_hist)
        b_hist = b_hist / np.sum(b_hist)
        
        # Calculate statistics
        l_mean, l_std = np.mean(l), np.std(l)
        a_mean, a_std = np.mean(a), np.std(a)
        b_mean, b_std = np.mean(b), np.std(b)
        
        # Compare with known profiles (simplified rules based on color rendition)
        # These would be learned from training data in a real implementation
        if 120 < l_mean < 140 and 6 < a_std < 9 and 12 < b_std < 18:
            return "Apple"
        elif 115 < l_mean < 135 and 8 < a_std < 12 and 15 < b_std < 22:
            return "Samsung"
        elif 125 < l_mean < 145 and 5 < a_std < 8 and 10 < b_std < 16:
            return "Google"
        elif l_std < 5 and a_std < 5 and b_std < 10:
            return "AI"
        
        return "Unknown"
    
    def analyze_compression_signature(self, img_array):
        """Analyze compression signature to identify source device"""
        # Convert to grayscale if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Check for 8x8 block artifacts characteristic of JPEG compression
        h, w = gray.shape
        block_size = 8
        
        # Calculate differences across block boundaries
        block_differences = []
        for y in range(0, h-1, block_size):
            for x in range(0, w-1, block_size):
                if y+block_size < h:
                    # Horizontal block boundary
                    upper_row = gray[y+block_size-1, x:min(x+block_size, w)]
                    lower_row = gray[y+block_size, x:min(x+block_size, w)]
                    block_differences.append(np.mean(np.abs(upper_row.astype(float) - lower_row.astype(float))))
                
                if x+block_size < w:
                    # Vertical block boundary
                    left_col = gray[y:min(y+block_size, h), x+block_size-1]
                    right_col = gray[y:min(y+block_size, h), x+block_size]
                    block_differences.append(np.mean(np.abs(left_col.astype(float) - right_col.astype(float))))
        
        if not block_differences:
            return "unknown"
            
        mean_diff = np.mean(block_differences)
        
        # Simplified classification based on block boundary differences
        if mean_diff < 0.8:
            return "progressive"  # Typical of iPhones, higher-end cameras
        else:
            return "baseline"     # Typical of some Android devices
    
    def recover_metadata(self, img_array):
        """Attempt to recover metadata from image content when original metadata is lost"""
        # Analyze image characteristics to infer metadata
        noise_profile = self.analyze_noise_pattern(img_array)
        color_profile = self.analyze_color_profile(img_array)
        compression = self.analyze_compression_signature(img_array)
        
        # Build recovered metadata dictionary
        recovered = {
            'RecoveryMethod': 'AI Content Analysis',
            'ColorProfile': color_profile,
            'CompressionType': compression
        }
        
        # Try to identify source device
        best_match = "Unknown"
        best_score = float('inf')
        
        for device, signature in self.metadata_signatures.items():
            # Calculate noise profile distance
            noise_distance = sum((a - b)**2 for a, b in zip(noise_profile, signature['noise_profile']))
            
            # Add color profile match
            color_match = 0 if color_profile == signature['color_profile'] else 1
            
            # Add compression match
            compression_match = 0 if compression == signature['compression'] else 1
            
            # Calculate overall score (lower is better)
            score = noise_distance + color_match * 2 + compression_match
            
            if score < best_score:
                best_score = score
                best_match = device
        
        # Add source device information if confident enough
        if best_score < 2.5:  # Threshold for reasonable confidence
            if 'iphone' in best_match:
                recovered['ProbableDevice'] = 'Apple iPhone'
            elif 'samsung' in best_match:
                recovered['ProbableDevice'] = 'Samsung Galaxy'
            elif 'google_pixel' in best_match:
                recovered['ProbableDevice'] = 'Google Pixel'
            elif 'dall_e' in best_match:
                recovered['ProbableDevice'] = 'DALL-E (AI Generated)'
                recovered['C2PANote'] = 'OpenAI now adds C2PA metadata to DALL-E images containing creator, timestamp, and prompt information'
            elif 'midjourney' in best_match:
                recovered['ProbableDevice'] = 'Midjourney (AI Generated)'
            elif 'stable_diffusion' in best_match:
                recovered['ProbableDevice'] = 'Stable Diffusion (AI Generated)'
        
        # Add estimated creation date based on image characteristics
        # This would use more sophisticated analysis in a real implementation
        recovered['EstimatedCreationDate'] = 'Recent (unable to determine exact date)'
        
        return recovered

# Create singleton instance
_analyzer = None

def extract_metadata(image_path):
    """
    Extract all available metadata from an image.
    Returns a dictionary of metadata.
    """
    global _analyzer
    
    if _analyzer is None:
        _analyzer = MetadataAnalyzer()
    
    return _analyzer.extract_metadata(image_path)

def recover_metadata(image_array):
    """
    Attempt to recover metadata from an image that has lost its original metadata.
    Returns a dictionary of recovered metadata.
    """
    global _analyzer
    
    if _analyzer is None:
        _analyzer = MetadataAnalyzer()
    
    return _analyzer.recover_metadata(image_array)
