"""
Core Detection and Identification Engine for YOLO Unauthorized Person Detection System
This is the main inference engine that processes camera feed and performs detection
"""

import os
import sys
import cv2
import numpy as np
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from ultralytics import YOLO
import face_recognition

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import (
    load_config, setup_logging, draw_bounding_box, extract_face_region,
    compute_face_encoding, load_authorized_embeddings, recognize_face,
    blur_face, calculate_fps, create_timestamp, format_detection_result,
    validate_config, ensure_directory_exists
)
from alert_system.alert import AlertSystem
from alert_system.logger import DetectionLogger


class PersonDetectionEngine:
    """
    Main person detection and identification engine
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize detection engine
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        if not validate_config(self.config):
            raise ValueError("Invalid configuration")
        
        # Setup logging
        self.logger = setup_logging(self.config)
        self.logger.info("🚀 Initializing Person Detection Engine")
        
        # Configuration sections
        self.model_config = self.config.get('model', {})
        self.camera_config = self.config.get('camera', {})
        self.paths_config = self.config.get('paths', {})
        self.display_config = self.config.get('display', {})
        self.performance_config = self.config.get('performance', {})
        self.security_config = self.config.get('security', {})
        
        # Model settings
        self.model_path = self.model_config.get('yolo_model_path', 'yolov8n.pt')
        self.detection_confidence = self.model_config.get('detection_confidence', 0.5)
        self.face_tolerance = self.model_config.get('face_tolerance', 0.5)
        self.max_faces_per_frame = self.model_config.get('max_faces_per_frame', 10)
        
        # Camera settings
        self.camera_index = self.camera_config.get('camera_index', 0)
        self.frame_width = self.camera_config.get('frame_width', 640)
        self.frame_height = self.camera_config.get('frame_height', 480)
        self.fps = self.camera_config.get('fps', 30)
        
        # Paths
        self.embeddings_file = self.paths_config.get('embeddings_file', 'dataset/face_embeddings.npy')
        self.authorized_persons_file = self.paths_config.get('authorized_persons_file', 'dataset/authorized_persons.json')
        
        # Performance settings
        self.use_gpu = self.performance_config.get('use_gpu', True)
        self.frame_skip = self.performance_config.get('frame_skip', 1)
        
        # Security settings
        self.blur_unauthorized_faces = self.security_config.get('blur_unauthorized_faces', False)
        self.blur_intensity = self.security_config.get('blur_intensity', 50)
        
        # Initialize components
        self.yolo_model = None
        self.authorized_embeddings = []
        self.authorized_names = []
        self.alert_system = None
        self.detection_logger = None
        self.camera = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Detection tracking
        self.unauthorized_attempts = 0
        self.max_unauthorized_attempts = self.security_config.get('max_unauthorized_attempts', 5)
        
        # Initialize all components
        self.initialize_components()
        
        self.logger.info("✅ Person Detection Engine initialized successfully")
    
    def initialize_components(self):
        """
        Initialize all system components
        """
        try:
            # Initialize YOLO model
            self.initialize_yolo_model()
            
            # Load authorized face embeddings
            self.load_authorized_embeddings()
            
            # Initialize alert system
            self.initialize_alert_system()
            
            # Initialize detection logger
            self.initialize_detection_logger()
            
            # Initialize camera
            self.initialize_camera()
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}")
            raise
    
    def initialize_yolo_model(self):
        """
        Initialize YOLO model for person detection
        """
        try:
            self.logger.info(f"🤖 Loading YOLO model: {self.model_path}")
            
            # Load YOLO model
            self.yolo_model = YOLO(self.model_path)
            
            # Test model with dummy image
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = self.yolo_model(dummy_image, verbose=False)
            
            self.logger.info("✅ YOLO model loaded successfully")
            self.logger.info(f"📊 Model classes: {self.yolo_model.names}")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading YOLO model: {e}")
            raise
    
    def load_authorized_embeddings(self):
        """
        Load authorized face embeddings
        """
        try:
            self.logger.info("👤 Loading authorized face embeddings...")
            
            self.authorized_embeddings, self.authorized_names = load_authorized_embeddings(
                self.embeddings_file, self.authorized_persons_file
            )
            
            if self.authorized_embeddings:
                unique_persons = set(self.authorized_names)
                self.logger.info(f"✅ Loaded {len(self.authorized_embeddings)} embeddings for {len(unique_persons)} persons")
                for person in unique_persons:
                    count = self.authorized_names.count(person)
                    self.logger.info(f"   - {person}: {count} embeddings")
            else:
                self.logger.warning("⚠️ No authorized embeddings found. System will treat all persons as unauthorized.")
        
        except Exception as e:
            self.logger.error(f"❌ Error loading authorized embeddings: {e}")
            self.authorized_embeddings = []
            self.authorized_names = []
    
    def initialize_alert_system(self):
        """
        Initialize alert system
        """
        try:
            self.alert_system = AlertSystem()
            self.logger.info("✅ Alert system initialized")
        except Exception as e:
            self.logger.error(f"❌ Error initializing alert system: {e}")
            self.alert_system = None
    
    def initialize_detection_logger(self):
        """
        Initialize detection logger
        """
        try:
            self.detection_logger = DetectionLogger()
            self.logger.info("✅ Detection logger initialized")
        except Exception as e:
            self.logger.error(f"❌ Error initializing detection logger: {e}")
            self.detection_logger = None
    
    def initialize_camera(self):
        """
        Initialize camera
        """
        try:
            self.logger.info(f"📷 Initializing camera {self.camera_index}...")
            
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                raise Exception(f"Cannot open camera {self.camera_index}")
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Test camera
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Cannot read from camera")
            
            self.logger.info(f"✅ Camera initialized successfully")
            self.logger.info(f"📐 Frame size: {frame.shape[1]}x{frame.shape[0]}")
        
        except Exception as e:
            self.logger.error(f"❌ Error initializing camera: {e}")
            raise
    
    def detect_persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect persons in frame using YOLO
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected persons with bounding boxes and confidence
        """
        try:
            # Run YOLO inference
            results = self.yolo_model(frame, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class ID and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Check if it's a person (class 0 in COCO dataset)
                        if class_id == 0 and confidence >= self.detection_confidence:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': 'person'
                            })
            
            return detections
        
        except Exception as e:
            self.logger.error(f"❌ Error in person detection: {e}")
            return []
    
    def identify_person(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[Optional[str], float]:
        """
        Identify person using face recognition
        
        Args:
            frame: Input frame
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            Tuple of (person_name, confidence) or (None, 0.0) if not recognized
        """
        try:
            # Extract face region
            face_region = extract_face_region(frame, bbox)
            if face_region is None:
                return None, 0.0
            
            # Compute face encoding
            face_encoding = compute_face_encoding(face_region)
            if face_encoding is None:
                return None, 0.0
            
            # Recognize face
            person_name, confidence = recognize_face(
                face_encoding, self.authorized_embeddings, self.authorized_names, self.face_tolerance
            )
            
            return person_name, confidence
        
        except Exception as e:
            self.logger.error(f"❌ Error in person identification: {e}")
            return None, 0.0
    
    def process_detection(self, person_name: str, confidence: float, bbox: Tuple[int, int, int, int],
                         frame: np.ndarray) -> bool:
        """
        Process detection result and trigger appropriate actions
        
        Args:
            person_name: Name of detected person
            confidence: Detection confidence
            bbox: Bounding box coordinates
            frame: Current frame
            
        Returns:
            True if person is authorized, False otherwise
        """
        is_authorized = person_name is not None
        
        # Log detection event
        if self.detection_logger:
            alert_triggered = False
            alert_type = ""
            
            if not is_authorized:
                # Trigger alert for unauthorized person
                if self.alert_system:
                    alert_triggered = self.alert_system.trigger_alert(
                        person_name or "Unknown", confidence, frame
                    )
                    alert_type = "sound" if alert_triggered else ""
                
                # Update unauthorized attempts counter
                self.unauthorized_attempts += 1
                
                # Check for lockout
                if self.unauthorized_attempts >= self.max_unauthorized_attempts:
                    self.logger.warning(f"🚫 Maximum unauthorized attempts reached: {self.unauthorized_attempts}")
                    # Here you could implement a lockout mechanism
            
            self.detection_logger.log_detection_event(
                person_name or "Unknown", confidence, is_authorized, bbox,
                camera_index=self.camera_index, frame_size=(frame.shape[1], frame.shape[0]),
                alert_triggered=alert_triggered, alert_type=alert_type
            )
        
        # Print detection result
        result_text = format_detection_result(person_name or "Unknown", confidence, is_authorized)
        print(result_text)
        
        return is_authorized
    
    def draw_detection_results(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detection results on frame
        
        Args:
            frame: Input frame
            detections: List of detection results
            
        Returns:
            Frame with drawn results
        """
        display_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            person_name = detection.get('person_name', 'Unknown')
            is_authorized = detection.get('is_authorized', False)
            
            # Choose color based on authorization status
            colors = self.display_config.get('colors', {})
            if is_authorized:
                color = tuple(colors.get('authorized', [0, 255, 0]))
                label = f"{person_name} (Authorized)"
            else:
                color = tuple(colors.get('unauthorized', [0, 0, 255]))
                label = f"{person_name} (Unauthorized)"
            
            # Draw bounding box
            display_frame = draw_bounding_box(
                display_frame, bbox, label, confidence, color, self.config
            )
            
            # Blur face if unauthorized and blurring is enabled
            if not is_authorized and self.blur_unauthorized_faces:
                display_frame = blur_face(display_frame, bbox, self.blur_intensity)
        
        # Draw FPS counter
        if self.display_config.get('show_fps', True):
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(display_frame, fps_text, (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw detection count
        detection_text = f"Detections: {len(detections)}"
        cv2.putText(display_frame, detection_text, (10, display_frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_frame
    
    def update_fps(self):
        """
        Update FPS calculation
        """
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def run_detection_loop(self):
        """
        Main detection loop
        """
        self.logger.info("🎬 Starting detection loop...")
        print("🎬 Starting YOLO Unauthorized Person Detection System")
        print("📋 Press 'q' to quit, 's' to save frame, 'r' to reload embeddings")
        print("=" * 60)
        
        try:
            while True:
                # Read frame from camera
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.error("❌ Failed to read frame from camera")
                    break
                
                # Skip frames for performance
                if self.frame_count % self.frame_skip != 0:
                    self.frame_count += 1
                    continue
                
                # Detect persons
                detections = self.detect_persons(frame)
                
                # Process each detection
                processed_detections = []
                for detection in detections:
                    bbox = detection['bbox']
                    confidence = detection['confidence']
                    
                    # Identify person
                    person_name, face_confidence = self.identify_person(frame, bbox)
                    
                    # Process detection
                    is_authorized = self.process_detection(person_name, confidence, bbox, frame)
                    
                    # Add to processed detections
                    detection['person_name'] = person_name or "Unknown"
                    detection['is_authorized'] = is_authorized
                    detection['face_confidence'] = face_confidence
                    processed_detections.append(detection)
                
                # Draw results
                display_frame = self.draw_detection_results(frame, processed_detections)
                
                # Update FPS
                self.update_fps()
                
                # Show frame
                cv2.imshow('YOLO Unauthorized Person Detection', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("🛑 Quitting detection loop")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"logs/capture_{timestamp}.jpg"
                    ensure_directory_exists("logs")
                    cv2.imwrite(filename, display_frame)
                    print(f"📸 Frame saved: {filename}")
                elif key == ord('r'):
                    # Reload embeddings
                    self.logger.info("🔄 Reloading authorized embeddings...")
                    self.load_authorized_embeddings()
                    print("✅ Embeddings reloaded")
        
        except KeyboardInterrupt:
            self.logger.info("🛑 Detection loop interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Error in detection loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        Cleanup resources
        """
        try:
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
            
            # Log system shutdown
            if self.detection_logger:
                self.detection_logger.log_system_event(
                    "SYSTEM_SHUTDOWN", "Detection system stopped"
                )
            
            self.logger.info("✅ System cleanup completed")
        
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")


def main():
    """
    Main function for detection engine
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Unauthorized Person Detection System')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--camera', type=int, help='Camera index to use')
    parser.add_argument('--test-camera', action='store_true', help='Test camera connection')
    
    args = parser.parse_args()
    
    try:
        # Create detection engine
        engine = PersonDetectionEngine(args.config)
        
        if args.camera is not None:
            engine.camera_index = args.camera
            engine.initialize_camera()
        
        if args.test_camera:
            # Test camera connection
            ret, frame = engine.camera.read()
            if ret:
                print(f"✅ Camera {engine.camera_index} working correctly")
                print(f"📐 Frame size: {frame.shape[1]}x{frame.shape[0]}")
                cv2.imshow('Camera Test', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"❌ Camera {engine.camera_index} not working")
            return
        
        # Run detection loop
        engine.run_detection_loop()
    
    except Exception as e:
        print(f"❌ Error in detection engine: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
