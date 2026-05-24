import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import os
import time

class PlateDetector:
    def __init__(self, model_path: str = "models/best.pt"):
        """Initialize the YOLO plate detector"""
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}. Using default YOLOv8 model.")
            self.model = YOLO('yolov8n.pt')
        else:
            self.model = YOLO(model_path)
        
        self.conf_threshold = 0.3  # Increased for better accuracy
        self.iou_threshold = 0.5   # Non-maximum suppression threshold
        self.last_detection_time = 0
        self.detection_cache = None
        self.cache_valid_for = 0.5  # Cache valid for 0.5 seconds
    
    def detect_plates(self, image: np.ndarray) -> List[Tuple[np.ndarray, float, Tuple[int, int, int, int]]]:
        """
        Detect license plates in an image with optimizations
        Returns: List of (cropped_plate_image, confidence, bbox)
        """
        current_time = time.time()
        
        # Use cache if available and recent
        if (self.detection_cache is not None and 
            current_time - self.last_detection_time < self.cache_valid_for):
            return self.detection_cache
        
        # Resize image for faster processing if it's too large
        height, width = image.shape[:2]
        if width > 1280 or height > 720:
            scale = min(1280/width, 720/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_resized = cv2.resize(image, (new_width, new_height))
        else:
            image_resized = image
        
        # Run detection with optimized parameters
        results = self.model(image_resized, 
                            conf=self.conf_threshold,
                            iou=self.iou_threshold,
                            verbose=False)  # Disable verbose output for speed
        
        plates = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                # Scale coordinates back to original size if image was resized
                if 'image_resized' in locals() and image_resized is not image:
                    scale_x = width / new_width
                    scale_y = height / new_height
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                
                # Ensure coordinates are within bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # Crop the plate region
                plate_img = image[y1:y2, x1:x2]
                
                if plate_img.size > 0 and plate_img.shape[0] > 10 and plate_img.shape[1] > 10:
                    plates.append((plate_img, confidence, (x1, y1, x2, y2)))
        
        # Cache the results
        self.detection_cache = plates
        self.last_detection_time = current_time
        
        return plates
    
    def draw_detections(self, image: np.ndarray, detections: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> np.ndarray:
        """
        Draw bounding boxes and plate numbers on image
        detections: List of (plate_text, confidence, bbox)
        """
        img_copy = image.copy()
        
        for plate_text, confidence, (x1, y1, x2, y2) in detections:
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label
            label = f"{plate_text} ({confidence:.2f})"
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_copy, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return img_copy
