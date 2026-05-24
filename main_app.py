"""
AI-Enabled Vehicle Number Plate Recognition System
Desktop Application with Live CCTV Detection and Photo Capture
OPTIMIZED VERSION: Fast and Accurate with Motion Detection
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import queue
from datetime import datetime
import os
from pathlib import Path
import numpy as np
import time

from detector import PlateDetector
from ocr import PlateOCR
from database import Database
from config import Config
from vehicle_classifier import VehicleClassifier


class VNPRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Vehicle Number Plate Recognition System (Optimized)")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Initialize components
        self.detector = PlateDetector()
        self.ocr = PlateOCR()
        self.db = Database()
        self.vehicle_classifier = VehicleClassifier()
        self.use_vehicle_classifier = True  # Enable/disable vehicle classification
        
        # Video capture variables
        self.cap = None
        self.is_running = False
        self.current_camera = 0
        self.frame_queue = queue.Queue(maxsize=2)
        self.detection_enabled = True
        self.current_detections = []  # Store current frame detections
        
        # Performance optimization variables
        self.prev_frame = None
        self.prev_color_frame = None
        self.motion_threshold = Config.MOTION_THRESHOLD
        self.skip_frames = 0
        self.frame_counter = 0
        self.processing_time = 0
        self.fps = 0
        self.last_time = time.time()
        self.frame_skip = Config.FRAME_SKIP
        
        # Vehicle detection state
        self.vehicle_detected = False
        self.vehicle_detection_counter = 0
        self.vehicle_detection_threshold = Config.VEHICLE_DETECTION_THRESHOLD
        
        # Create directories
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('captures', exist_ok=True)
        
        # Setup UI
        self.setup_ui()
        
        # Start video thread
        self.video_thread = None
        
    def setup_ui(self):
        """Setup the user interface"""
        # Title Bar
        title_frame = tk.Frame(self.root, bg='#34495e', height=60)
        title_frame.pack(fill=tk.X, side=tk.TOP)
        
        title_label = tk.Label(
            title_frame,
            text="🚗 AI Vehicle Number Plate Recognition System",
            font=('Arial', 20, 'bold'),
            bg='#34495e',
            fg='white'
        )
        title_label.pack(pady=15)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Panel - Video Feed
        left_panel = tk.Frame(main_container, bg='#34495e', relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Video label
        video_label_frame = tk.Frame(left_panel, bg='#34495e')
        video_label_frame.pack(pady=5)
        
        tk.Label(
            video_label_frame,
            text="📹 Live Camera Feed",
            font=('Arial', 14, 'bold'),
            bg='#34495e',
            fg='white'
        ).pack()
        
        # Video display
        self.video_label = tk.Label(left_panel, bg='black')
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Control buttons
        control_frame = tk.Frame(left_panel, bg='#34495e')
        control_frame.pack(pady=10, fill=tk.X, padx=10)
        
        # Camera selection
        camera_frame = tk.Frame(control_frame, bg='#34495e')
        camera_frame.pack(side=tk.LEFT, padx=5)
        
        tk.Label(camera_frame, text="Camera:", bg='#34495e', fg='white', font=('Arial', 10)).pack(side=tk.LEFT)
        self.camera_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_var, width=5, state='readonly')
        camera_combo['values'] = ['0', '1', '2', '3']
        camera_combo.pack(side=tk.LEFT, padx=5)
        
        # Start/Stop button
        self.start_btn = tk.Button(
            control_frame,
            text="▶ Start Camera",
            command=self.toggle_camera,
            bg='#27ae60',
            fg='white',
            font=('Arial', 11, 'bold'),
            width=15,
            cursor='hand2'
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        # Capture button
        self.capture_btn = tk.Button(
            control_frame,
            text="📷 Capture Photo",
            command=self.capture_photo,
            bg='#3498db',
            fg='white',
            font=('Arial', 11, 'bold'),
            width=15,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        # Load Image button
        self.load_btn = tk.Button(
            control_frame,
            text="📁 Load Image",
            command=self.load_image,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 11, 'bold'),
            width=15,
            cursor='hand2'
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        # Detection toggle
        self.detection_var = tk.BooleanVar(value=True)
        detection_check = tk.Checkbutton(
            control_frame,
            text="Enable Detection",
            variable=self.detection_var,
            bg='#34495e',
            fg='white',
            selectcolor='#2c3e50',
            font=('Arial', 10),
            command=self.toggle_detection
        )
        detection_check.pack(side=tk.LEFT, padx=10)
        
        # Right Panel - Results and History
        right_panel = tk.Frame(main_container, bg='#34495e', relief=tk.RAISED, bd=2, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        # Results section
        results_label = tk.Label(
            right_panel,
            text="🎯 Detection Results",
            font=('Arial', 14, 'bold'),
            bg='#34495e',
            fg='white'
        )
        results_label.pack(pady=10)
        
        # Current detection display
        self.current_detection_frame = tk.Frame(right_panel, bg='#2c3e50', relief=tk.SUNKEN, bd=2)
        self.current_detection_frame.pack(padx=10, pady=5, fill=tk.X)
        
        self.current_plate_label = tk.Label(
            self.current_detection_frame,
            text="No detection yet",
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='#ecf0f1',
            pady=20
        )
        self.current_plate_label.pack()
        
        self.current_confidence_label = tk.Label(
            self.current_detection_frame,
            text="",
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#95a5a6'
        )
        self.current_confidence_label.pack()
        
        # Statistics
        stats_frame = tk.Frame(right_panel, bg='#34495e')
        stats_frame.pack(pady=10, fill=tk.X, padx=10)
        
        self.total_detections_label = tk.Label(
            stats_frame,
            text="Total Detections: 0",
            font=('Arial', 11),
            bg='#34495e',
            fg='white'
        )
        self.total_detections_label.pack()
        
        # Performance metrics
        self.fps_label = tk.Label(
            stats_frame,
            text="FPS: 0",
            font=('Arial', 10),
            bg='#34495e',
            fg='#95a5a6'
        )
        self.fps_label.pack()
        
        self.processing_time_label = tk.Label(
            stats_frame,
            text="Processing: 0ms",
            font=('Arial', 10),
            bg='#34495e',
            fg='#95a5a6'
        )
        self.processing_time_label.pack()
        
        self.motion_status_label = tk.Label(
            stats_frame,
            text="Motion: No",
            font=('Arial', 10),
            bg='#34495e',
            fg='#95a5a6'
        )
        self.motion_status_label.pack()
        
        # History section
        history_label = tk.Label(
            right_panel,
            text="📋 Detection History",
            font=('Arial', 12, 'bold'),
            bg='#34495e',
            fg='white'
        )
        history_label.pack(pady=(20, 5))
        
        # History listbox with scrollbar
        history_frame = tk.Frame(right_panel, bg='#34495e')
        history_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(history_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.history_listbox = tk.Listbox(
            history_frame,
            yscrollcommand=scrollbar.set,
            font=('Courier', 10),
            bg='#2c3e50',
            fg='white',
            selectbackground='#3498db',
            relief=tk.FLAT
        )
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.history_listbox.yview)
        
        # Export button
        export_btn = tk.Button(
            right_panel,
            text="💾 Export History",
            command=self.export_history,
            bg='#e67e22',
            fg='white',
            font=('Arial', 10, 'bold'),
            cursor='hand2'
        )
        export_btn.pack(pady=10)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bg='#34495e',
            fg='white',
            anchor=tk.W,
            font=('Arial', 9)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def toggle_camera(self):
        """Start or stop the camera"""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start the camera feed"""
        try:
            camera_index = int(self.camera_var.get())
            self.current_camera = camera_index
            
            # Try to open camera with DirectShow (Windows)
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Cannot open camera {camera_index}")
                return
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.is_running = True
            self.start_btn.config(text="⏸ Stop Camera", bg='#e74c3c')
            self.capture_btn.config(state=tk.NORMAL)
            self.update_status("Camera started")
            
            # Start video thread
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop the camera feed"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(text="▶ Start Camera", bg='#27ae60')
        self.capture_btn.config(state=tk.DISABLED)
        self.video_label.config(image='', bg='black')
        self.update_status("Camera stopped")
    
    def video_loop(self):
        """Main video processing loop with motion detection"""
        frame_count = 0
        
        while self.is_running:
            try:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Clear previous detections
                self.current_detections = []
                
                # Calculate FPS
                current_time = time.time()
                self.fps = 1.0 / (current_time - self.last_time) if current_time > self.last_time else 0
                self.last_time = current_time
                
                # Check for motion detection
                has_motion = self.detect_motion(frame)
                
                # Update vehicle detection state
                if has_motion:
                    self.vehicle_detection_counter += 1
                    if self.vehicle_detection_counter >= self.vehicle_detection_threshold:
                        self.vehicle_detected = True
                else:
                    self.vehicle_detection_counter = max(0, self.vehicle_detection_counter - 1)
                    if self.vehicle_detection_counter == 0:
                        self.vehicle_detected = False
                
                # Only process detection when vehicle is detected and enabled
                if self.detection_enabled and self.vehicle_detected and frame_count % self.frame_skip == 0:
                    self.process_frame(frame)
                
                # Draw detections on frame
                frame = self.draw_detections_on_frame(frame)
                
                # Add performance overlay
                frame = self.add_performance_overlay(frame, has_motion)
                
                # Display frame
                self.display_frame(frame)
                
                # Update performance metrics
                self.processing_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Update UI metrics every 10 frames
                if frame_count % 10 == 0:
                    self.root.after(0, self.update_performance_metrics, has_motion)
                    
                # Dynamically adjust settings based on performance
                if frame_count % 30 == 0:
                    self.adjust_performance_settings()
                
            except Exception as e:
                print(f"Video loop error: {e}")
                break
    
    def process_frame(self, frame):
        """Process frame for plate detection"""
        try:
            # Detect plates
            plates = self.detector.detect_plates(frame)
            
            if plates:
                for plate_img, det_confidence, bbox in plates:
                    # Perform OCR
                    ocr_result = self.ocr.read_plate(plate_img)
                    
                    if ocr_result:
                        plate_text, ocr_confidence = ocr_result
                        combined_confidence = (det_confidence + ocr_confidence) / 2
                        
                        # Save detection
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        image_filename = f"plate_{plate_text}_{timestamp}.jpg"
                        image_path = os.path.join('uploads', image_filename)
                        cv2.imwrite(image_path, plate_img)
                        
                        self.db.save_detection(
                            plate_number=plate_text,
                            confidence=combined_confidence,
                            image_path=image_filename
                        )
                        
                        # Store detection for drawing
                        self.current_detections.append((plate_text, combined_confidence, bbox))
                        
                        # Update UI
                        self.root.after(0, self.update_detection_display, plate_text, combined_confidence)
        
        except Exception as e:
            print(f"Processing error: {e}")
    
    def detect_motion(self, frame):
        """Detect motion between frames with optional vehicle classification"""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Compute absolute difference between current and previous frame
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        
        # Apply threshold
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate the thresholded image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any contour is large enough to be considered motion
        motion_detected = False
        motion_confidence = 0.0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.motion_threshold:
                motion_detected = True
                
                # If vehicle classifier is enabled, check if it's actually a vehicle
                if self.use_vehicle_classifier and self.prev_color_frame is not None:
                    # Create motion mask for this contour
                    contour_mask = np.zeros_like(thresh)
                    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                    
                    # Classify the motion region
                    is_vehicle, confidence = self.vehicle_classifier.classify_motion_region(
                        frame, contour_mask
                    )
                    
                    if is_vehicle:
                        motion_confidence = max(motion_confidence, confidence)
                    else:
                        # If not a vehicle, reduce motion confidence
                        motion_detected = False
                
                break
        
        # Update previous frames
        self.prev_frame = gray
        self.prev_color_frame = frame.copy()
        
        return motion_detected
    
    def draw_detections_on_frame(self, frame):
        """Draw bounding boxes and labels on frame"""
        for plate_text, confidence, bbox in self.current_detections:
            x1, y1, x2, y2 = bbox
            
            # Draw rectangle (green, thick)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Prepare label
            label = f"{plate_text} ({confidence:.2f})"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
            )
            
            # Draw label background (green filled rectangle)
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1  # Filled
            )
            
            # Draw label text (black on green background)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),  # Black text
                2
            )
        
        return frame
    
    def add_performance_overlay(self, frame, has_motion):
        """Add performance metrics overlay to frame"""
        # Create overlay text
        overlay_text = [
            f"FPS: {self.fps:.1f}",
            f"Processing: {self.processing_time:.1f}ms",
            f"Motion: {'YES' if has_motion else 'NO'}",
            f"Vehicle: {'DETECTED' if self.vehicle_detected else 'NO'}",
            f"Detection: {'ON' if self.detection_enabled else 'OFF'}"
        ]
        
        # Draw overlay background
        overlay_height = len(overlay_text) * 25 + 10
        cv2.rectangle(frame, (10, 10), (250, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, overlay_height), (255, 255, 255), 2)
        
        # Draw text
        for i, text in enumerate(overlay_text):
            color = (0, 255, 0) if "YES" in text or "DETECTED" in text or "ON" in text else (0, 165, 255)
            if "NO" in text or "OFF" in text:
                color = (0, 0, 255)
            
            y_pos = 35 + i * 25
            cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def update_performance_metrics(self, has_motion):
        """Update performance metrics in UI"""
        self.fps_label.config(text=f"FPS: {self.fps:.1f}")
        self.processing_time_label.config(text=f"Processing: {self.processing_time:.1f}ms")
        self.motion_status_label.config(text=f"Motion: {'Yes' if has_motion else 'No'}")
        
        # Color code motion status
        if has_motion:
            self.motion_status_label.config(fg='#27ae60')
        else:
            self.motion_status_label.config(fg='#e74c3c')
    
    def adjust_performance_settings(self):
        """Dynamically adjust performance settings based on current FPS"""
        optimized_settings = Config.get_optimized_settings(self.fps)
        
        # Update settings
        self.frame_skip = optimized_settings['frame_skip']
        self.motion_threshold = optimized_settings['motion_threshold']
        
        # Update detector confidence
        self.detector.conf_threshold = optimized_settings['detection_confidence']
        
        # Clear detection cache when settings change
        self.detector.detection_cache = None
    
    def display_frame(self, frame):
        """Display frame in the GUI"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to fit display
            display_width = 900
            display_height = 600
            frame_resized = cv2.resize(frame_rgb, (display_width, display_height))
            
            # Convert to PhotoImage
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update label
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        except Exception as e:
            print(f"Display error: {e}")
    
    def capture_photo(self):
        """Capture current frame and process it"""
        if not self.is_running or self.cap is None:
            messagebox.showwarning("Warning", "Camera is not running")
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture frame")
                return
            
            # Save captured frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            capture_filename = f"capture_{timestamp}.jpg"
            capture_path = os.path.join('captures', capture_filename)
            cv2.imwrite(capture_path, frame)
            
            # Process for detection
            self.process_image(frame, "Captured Photo")
            
            self.update_status(f"Photo captured: {capture_filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Capture failed: {str(e)}")
    
    def load_image(self):
        """Load and process an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Read image
            frame = cv2.imread(file_path)
            if frame is None:
                messagebox.showerror("Error", "Failed to load image")
                return
            
            # Process image
            self.process_image(frame, os.path.basename(file_path))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def process_image(self, frame, source_name):
        """Process a single image for plate detection"""
        try:
            # Detect plates
            plates = self.detector.detect_plates(frame)
            
            if not plates:
                messagebox.showinfo("Result", "No license plates detected")
                return
            
            detections = []
            
            for plate_img, det_confidence, bbox in plates:
                # Perform OCR
                ocr_result = self.ocr.read_plate(plate_img)
                
                if ocr_result:
                    plate_text, ocr_confidence = ocr_result
                    combined_confidence = (det_confidence + ocr_confidence) / 2
                    
                    # Save detection
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    image_filename = f"plate_{plate_text}_{timestamp}.jpg"
                    image_path = os.path.join('uploads', image_filename)
                    cv2.imwrite(image_path, plate_img)
                    
                    self.db.save_detection(
                        plate_number=plate_text,
                        confidence=combined_confidence,
                        image_path=image_filename
                    )
                    
                    detections.append((plate_text, combined_confidence, bbox))
                    
                    # Update UI
                    self.update_detection_display(plate_text, combined_confidence)
                    
                    # Draw on frame with better visibility
                    x1, y1, x2, y2 = bbox
                    
                    # Draw thick green rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    
                    # Prepare label
                    label = f"{plate_text} ({combined_confidence:.2f})"
                    
                    # Get text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
                    )
                    
                    # Draw label background (green filled rectangle)
                    cv2.rectangle(
                        frame,
                        (x1, y1 - text_height - 15),
                        (x1 + text_width + 10, y1),
                        (0, 255, 0),
                        -1  # Filled
                    )
                    
                    # Draw label text (black on green background)
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 0),  # Black text
                        2
                    )
            
            # Save annotated image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"result_{timestamp}.jpg"
            result_path = os.path.join('results', result_filename)
            cv2.imwrite(result_path, frame)
            
            # Show result
            result_msg = f"Detected {len(detections)} plate(s):\n\n"
            for plate_text, conf, _ in detections:
                result_msg += f"• {plate_text} (Confidence: {conf:.2f})\n"
            result_msg += f"\nResult saved: {result_filename}"
            
            messagebox.showinfo("Detection Result", result_msg)
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
    
    def update_detection_display(self, plate_text, confidence):
        """Update the current detection display"""
        self.current_plate_label.config(text=plate_text)
        
        # Color code by confidence
        if confidence >= 0.8:
            color = '#27ae60'  # Green
        elif confidence >= 0.6:
            color = '#f39c12'  # Orange
        else:
            color = '#e74c3c'  # Red
        
        self.current_plate_label.config(fg=color)
        self.current_confidence_label.config(
            text=f"Confidence: {confidence:.2%}",
            fg=color
        )
        
        # Update history
        timestamp = datetime.now().strftime("%H:%M:%S")
        history_entry = f"{timestamp} | {plate_text} | {confidence:.2%}"
        self.history_listbox.insert(0, history_entry)
        
        # Update total count
        total = self.db.get_detection_count()
        self.total_detections_label.config(text=f"Total Detections: {total}")
    
    def toggle_detection(self):
        """Toggle detection on/off"""
        self.detection_enabled = self.detection_var.get()
        status = "enabled" if self.detection_enabled else "disabled"
        self.update_status(f"Detection {status}")
    
    def export_history(self):
        """Export detection history to CSV"""
        try:
            detections = self.db.get_all_detections(limit=10000)
            
            if not detections:
                messagebox.showinfo("Export", "No detections to export")
                return
            
            # Ask for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            if not file_path:
                return
            
            # Write CSV
            with open(file_path, 'w') as f:
                f.write("Timestamp,Plate Number,Confidence,Image Path\n")
                for det in detections:
                    f.write(f"{det['timestamp']},{det['plate_number']},{det['confidence']},{det['image_path']}\n")
            
            messagebox.showinfo("Export", f"Exported {len(detections)} detections to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=f"Status: {message}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self.stop_camera()
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = VNPRApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
