import easyocr
import cv2
import numpy as np
import re
from typing import Optional
import time

class PlateOCR:
    def __init__(self, languages: list = ['en']):
        """Initialize EasyOCR reader with optimizations"""
        # Use smaller model for faster processing
        self.reader = easyocr.Reader(languages, gpu=False, model_storage_directory='.easyocr_cache')
        self.text_cache = {}
        self.cache_timeout = 2.0  # Cache timeout in seconds
    
    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """Preprocess plate image for better OCR results"""
        if plate_img is None or plate_img.size == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Resize if too small for better OCR
        height, width = gray.shape
        if height < 20 or width < 60:
            scale = max(30/height, 100/width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh
    
    def clean_plate_text(self, text: str) -> str:
        """Clean and format the OCR result"""
        # Remove special characters and spaces
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Common OCR corrections
        corrections = {
            '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B',
            'I': '1', 'O': '0', 'Z': '2', 'S': '5', 'B': '8'
        }
        
        # Apply corrections for common misreads
        corrected = ''
        for char in cleaned:
            if char in corrections:
                # Only correct if it makes sense in context
                if len(corrected) > 0 and corrected[-1].isalpha() and char.isdigit():
                    corrected += corrections[char]
                elif len(corrected) > 0 and corrected[-1].isdigit() and char.isalpha():
                    corrected += corrections[char]
                else:
                    corrected += char
            else:
                corrected += char
        
        return corrected
    
    def get_image_hash(self, plate_img: np.ndarray) -> str:
        """Generate a simple hash for image caching"""
        if plate_img is None:
            return "none"
        
        # Resize to small thumbnail for hashing
        small = cv2.resize(plate_img, (16, 16))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # Simple hash: average pixel value
        avg = np.mean(gray)
        return f"{avg:.2f}"
    
    def read_plate(self, plate_img: np.ndarray) -> Optional[tuple]:
        """
        Read text from plate image with caching and optimizations
        Returns: (plate_text, confidence) or None
        """
        if plate_img is None or plate_img.size == 0:
            return None
        
        # Check cache first
        img_hash = self.get_image_hash(plate_img)
        current_time = time.time()
        
        if img_hash in self.text_cache:
            cached_text, cached_conf, cached_time = self.text_cache[img_hash]
            if current_time - cached_time < self.cache_timeout:
                return cached_text, cached_conf
        
        # Preprocess the image
        processed = self.preprocess_plate(plate_img)
        
        if processed is None:
            return None
        
        # Try OCR on preprocessed image only (faster)
        try:
            results = self.reader.readtext(processed, 
                                          batch_size=1,
                                          allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                                          detail=1)
        except Exception as e:
            print(f"OCR error: {e}")
            return None
        
        if not results:
            return None
        
        # Get the best result
        best_result = max(results, key=lambda x: x[2])
        text, confidence = best_result[1], best_result[2]
        
        # Clean the text
        cleaned_text = self.clean_plate_text(text)
        
        # Validate plate format (typical plate has 6-10 characters)
        if len(cleaned_text) < 4 or len(cleaned_text) > 12:
            return None
        
        # Check for reasonable character distribution
        letters = sum(1 for c in cleaned_text if c.isalpha())
        digits = sum(1 for c in cleaned_text if c.isdigit())
        
        if letters == 0 or digits == 0:
            return None
        
        # Cache the result
        self.text_cache[img_hash] = (cleaned_text, confidence, current_time)
        
        # Clean old cache entries
        for key in list(self.text_cache.keys()):
            if current_time - self.text_cache[key][2] > self.cache_timeout * 2:
                del self.text_cache[key]
        
        return cleaned_text, confidence
