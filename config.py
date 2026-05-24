"""
Configuration settings for optimized vehicle number plate recognition
"""

class Config:
    # Performance settings
    FRAME_SKIP = 2  # Process every Nth frame when vehicle detected
    MOTION_THRESHOLD = 5000  # Sensitivity for motion detection
    VEHICLE_DETECTION_THRESHOLD = 3  # Consecutive frames with motion to detect vehicle
    
    # Detection settings
    DETECTION_CONFIDENCE = 0.3  # Minimum confidence for plate detection
    OCR_CONFIDENCE = 0.6  # Minimum confidence for OCR result
    CACHE_TIMEOUT = 0.5  # Detection cache timeout in seconds
    
    # Camera settings
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30
    
    # Display settings
    DISPLAY_WIDTH = 900
    DISPLAY_HEIGHT = 600
    SHOW_PERFORMANCE_OVERLAY = True
    
    # Processing settings
    ENABLE_PREPROCESSING = True
    ENABLE_CACHING = True
    MAX_PROCESSING_TIME_MS = 50  # Target max processing time per frame
    
    # File paths
    UPLOAD_DIR = 'uploads'
    RESULTS_DIR = 'results'
    CAPTURES_DIR = 'captures'
    
    @classmethod
    def get_optimized_settings(cls, current_fps):
        """Dynamically adjust settings based on performance"""
        settings = {
            'frame_skip': cls.FRAME_SKIP,
            'motion_threshold': cls.MOTION_THRESHOLD,
            'detection_confidence': cls.DETECTION_CONFIDENCE
        }
        
        # Adjust settings based on FPS
        if current_fps < 15:
            # System is struggling, reduce processing
            settings['frame_skip'] = max(3, cls.FRAME_SKIP + 1)
            settings['motion_threshold'] = cls.MOTION_THRESHOLD * 1.5
            settings['detection_confidence'] = min(0.5, cls.DETECTION_CONFIDENCE + 0.1)
        elif current_fps > 25:
            # System has capacity, increase processing
            settings['frame_skip'] = max(1, cls.FRAME_SKIP - 1)
            settings['motion_threshold'] = max(1000, cls.MOTION_THRESHOLD * 0.8)
            settings['detection_confidence'] = max(0.2, cls.DETECTION_CONFIDENCE - 0.05)
        
        return settings