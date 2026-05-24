from datetime import datetime
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        """Initialize in-memory database (fallback when MongoDB is not available)"""
        self.detections = []
        self.next_id = 1
        print("Using in-memory database (MongoDB not available)")
    
    def save_detection(self, plate_number: str, confidence: float, image_path: str, location: Optional[dict] = None):
        """Save a plate detection to the database"""
        detection = {
            "_id": str(self.next_id),
            "plate_number": plate_number,
            "confidence": confidence,
            "image_path": image_path,
            "location": location,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.detections.append(detection)
        self.next_id += 1
        return detection["_id"]
    
    def get_all_detections(self, limit: int = 100, skip: int = 0) -> List[dict]:
        """Retrieve all detections with pagination"""
        # Sort by timestamp (newest first)
        sorted_detections = sorted(self.detections, key=lambda x: x["timestamp"], reverse=True)
        return sorted_detections[skip:skip + limit]
    
    def get_detection_by_plate(self, plate_number: str) -> List[dict]:
        """Search detections by plate number"""
        return [d for d in self.detections if plate_number.lower() in d["plate_number"].lower()]
    
    def get_detection_count(self) -> int:
        """Get total number of detections"""
        return len(self.detections)
