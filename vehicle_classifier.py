"""
Simple vehicle classifier to improve detection accuracy
Uses basic image processing to distinguish vehicles from other motion
"""

import cv2
import numpy as np
from typing import Tuple, Optional

class VehicleClassifier:
    def __init__(self):
        """Initialize vehicle classifier"""
        # Vehicle characteristics (aspect ratios in pixels)
        self.car_aspect_ratio_range = (1.2, 3.0)  # Width/height ratio for cars
        self.truck_aspect_ratio_range = (2.5, 5.0)  # Width/height ratio for trucks
        self.min_vehicle_area = 5000  # Minimum area in pixels to be considered a vehicle
        self.max_vehicle_area = 50000  # Maximum area in pixels
        
        # Color characteristics (vehicles are typically darker than background)
        self.min_brightness = 30  # Minimum average brightness
        self.max_brightness = 200  # Maximum average brightness
        
        # Edge density (vehicles have more edges)
        self.min_edge_density = 0.05  # Minimum edge pixels / total pixels
        self.max_edge_density = 0.3   # Maximum edge pixels / total pixels
    
    def classify_motion_region(self, frame: np.ndarray, motion_mask: np.ndarray) -> Tuple[bool, float]:
        """
        Classify if a motion region contains a vehicle
        Returns: (is_vehicle, confidence)
        """
        if motion_mask is None or np.sum(motion_mask) == 0:
            return False, 0.0
        
        # Find contours in motion mask
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0.0
        
        # Find the largest contour (likely the moving object)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Check area constraints
        if area < self.min_vehicle_area or area > self.max_vehicle_area:
            return False, 0.0
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Check aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate confidence based on aspect ratio
        aspect_confidence = 0.0
        if self.car_aspect_ratio_range[0] <= aspect_ratio <= self.car_aspect_ratio_range[1]:
            aspect_confidence = 0.7
        elif self.truck_aspect_ratio_range[0] <= aspect_ratio <= self.truck_aspect_ratio_range[1]:
            aspect_confidence = 0.8
        else:
            aspect_confidence = 0.2
        
        # Extract region from original frame
        region = frame[y:y+h, x:x+w]
        if region.size == 0:
            return False, 0.0
        
        # Check brightness
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_region)
        
        brightness_confidence = 0.0
        if self.min_brightness <= avg_brightness <= self.max_brightness:
            # Higher confidence for medium brightness (typical vehicles)
            if 50 <= avg_brightness <= 150:
                brightness_confidence = 0.8
            else:
                brightness_confidence = 0.5
        else:
            brightness_confidence = 0.2
        
        # Check edge density
        edges = cv2.Canny(gray_region, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        
        edge_confidence = 0.0
        if self.min_edge_density <= edge_density <= self.max_edge_density:
            edge_confidence = 0.7
        else:
            edge_confidence = 0.3
        
        # Calculate overall confidence
        overall_confidence = (aspect_confidence * 0.4 + 
                             brightness_confidence * 0.3 + 
                             edge_confidence * 0.3)
        
        is_vehicle = overall_confidence >= 0.5
        
        return is_vehicle, overall_confidence
    
    def create_motion_mask(self, current_frame: np.ndarray, previous_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Create motion mask between two frames
        """
        if previous_frame is None:
            return None
        
        # Convert to grayscale
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        gray_current = cv2.GaussianBlur(gray_current, (21, 21), 0)
        gray_previous = cv2.GaussianBlur(gray_previous, (21, 21), 0)
        
        # Compute absolute difference
        frame_delta = cv2.absdiff(gray_previous, gray_current)
        
        # Apply threshold
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate to fill holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        return thresh
    
    def draw_vehicle_region(self, frame: np.ndarray, motion_mask: np.ndarray, is_vehicle: bool, confidence: float):
        """
        Draw vehicle detection region on frame
        """
        if motion_mask is None:
            return frame
        
        # Find contours
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return frame
        
        # Draw the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Choose color based on classification
        if is_vehicle:
            color = (0, 255, 0)  # Green for vehicle
            label = f"Vehicle ({confidence:.2f})"
        else:
            color = (0, 0, 255)  # Red for non-vehicle
            label = f"Non-vehicle ({confidence:.2f})"
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame


# Test function
def test_classifier():
    """Test the vehicle classifier"""
    print("Testing Vehicle Classifier...")
    
    classifier = VehicleClassifier()
    
    # Create test frames
    # Frame 1: Empty
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Frame 2: With "vehicle" (white rectangle)
    frame2 = frame1.copy()
    cv2.rectangle(frame2, (100, 100), (300, 200), (255, 255, 255), -1)
    
    # Create motion mask
    motion_mask = classifier.create_motion_mask(frame2, frame1)
    
    if motion_mask is not None:
        # Classify
        is_vehicle, confidence = classifier.classify_motion_region(frame2, motion_mask)
        
        print(f"Motion mask created: {motion_mask.shape}")
        print(f"Vehicle detected: {is_vehicle}")
        print(f"Confidence: {confidence:.2f}")
        
        # Draw result
        result_frame = classifier.draw_vehicle_region(frame2.copy(), motion_mask, is_vehicle, confidence)
        
        # Save for inspection
        cv2.imwrite('test_vehicle_classification.jpg', result_frame)
        print("Test image saved as 'test_vehicle_classification.jpg'")
    else:
        print("Failed to create motion mask")


if __name__ == "__main__":
    test_classifier()