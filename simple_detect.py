"""
Simplified YOLO Person Detection System (without face recognition for now)
"""
import os
import sys
import cv2
import numpy as np
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from ultralytics import YOLO
import yaml
import json
import logging

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"❌ Error parsing configuration file: {e}")
        return {}

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    log_file = log_config.get('log_file', 'logs/detection.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_config.get('log_level', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if log_config.get('enable_console_log', True) else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)

def draw_bounding_box(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                     label: str, confidence: float, color: Tuple[int, int, int]) -> np.ndarray:
    """Draw bounding box with label and confidence on image"""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    label_text = f"{label}: {confidence:.2f}"
    font_scale = 0.7
    font_thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
    cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    return image

def create_timestamp() -> str:
    """Create timestamp string for logging"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def format_detection_result(name: str, confidence: float, is_authorized: bool) -> str:
    """Format detection result for logging"""
    status = "✅" if is_authorized else "🚫 ALERT"
    auth_status = "Authorized" if is_authorized else "Unauthorized"
    return f"[{auth_status}] {name} - Confidence: {confidence:.2f} {status}"

class SimplePersonDetectionEngine:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config)
        
        # Configuration
        self.model_config = self.config.get('model', {})
        self.camera_config = self.config.get('camera', {})
        self.display_config = self.config.get('display', {})
        
        # Settings
        self.model_path = self.model_config.get('yolo_model_path', 'yolov8n.pt')
        self.detection_confidence = self.model_config.get('detection_confidence', 0.5)
        self.camera_index = self.camera_config.get('camera_index', 0)
        
        # Initialize components
        self.yolo_model = None
        self.camera = None
        
        self.initialize_components()
        self.logger.info("✅ Simple Person Detection Engine initialized successfully")
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            self.initialize_yolo_model()
            self.initialize_camera()
        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}")
            raise
    
    def initialize_yolo_model(self):
        """Initialize YOLO model for person detection"""
        try:
            self.logger.info(f"🤖 Loading YOLO model: {self.model_path}")
            self.yolo_model = YOLO(self.model_path)
            
            # Test model
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = self.yolo_model(dummy_image, verbose=False)
            self.logger.info("✅ YOLO model loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Error loading YOLO model: {e}")
            raise
    
    def initialize_camera(self):
        """Initialize camera"""
        try:
            self.logger.info(f"📷 Initializing camera {self.camera_index}...")
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                raise Exception(f"Cannot open camera {self.camera_index}")
            
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Cannot read from camera")
            
            self.logger.info(f"✅ Camera initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing camera: {e}")
            raise
    
    def detect_persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect persons in frame using YOLO"""
        try:
            results = self.yolo_model(frame, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if class_id == 0 and confidence >= self.detection_confidence:  # Person class
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
    
    def process_detection(self, confidence: float, bbox: Tuple[int, int, int, int]) -> bool:
        """Process detection result - for now, treat all as unauthorized"""
        # For demonstration, we'll treat all detections as unauthorized
        # In a real system, you would do face recognition here
        is_authorized = False
        person_name = "Unknown Person"
        
        # Print detection result
        result_text = format_detection_result(person_name, confidence, is_authorized)
        print(result_text)
        
        return is_authorized
    
    def draw_detection_results(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection results on frame"""
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
            display_frame = draw_bounding_box(display_frame, bbox, label, confidence, color)
        
        # Add FPS counter
        fps_text = f"FPS: {self.calculate_fps():.1f}"
        cv2.putText(display_frame, fps_text, (10, display_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_frame
    
    def calculate_fps(self) -> float:
        """Calculate FPS"""
        current_time = time.time()
        if not hasattr(self, 'last_fps_time'):
            self.last_fps_time = current_time
            self.frame_count = 0
        
        self.frame_count += 1
        
        if current_time - self.last_fps_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
            return fps
        
        return getattr(self, 'current_fps', 0.0)
    
    def run_detection_loop(self):
        """Main detection loop"""
        self.logger.info("🎬 Starting detection loop...")
        print("🎬 Starting YOLO Person Detection System (Simplified)")
        print("📋 Press 'q' to quit, 's' to save frame")
        print("⚠️ Note: All persons are treated as unauthorized (face recognition not available)")
        print("=" * 60)
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.error("❌ Failed to read frame from camera")
                    break
                
                # Detect persons
                detections = self.detect_persons(frame)
                
                # Process each detection
                processed_detections = []
                for detection in detections:
                    bbox = detection['bbox']
                    confidence = detection['confidence']
                    
                    # Process detection (treat all as unauthorized for now)
                    is_authorized = self.process_detection(confidence, bbox)
                    
                    # Add to processed detections
                    detection['person_name'] = "Unknown Person"
                    detection['is_authorized'] = is_authorized
                    processed_detections.append(detection)
                
                # Draw results
                display_frame = self.draw_detection_results(frame, processed_detections)
                
                # Show frame
                cv2.imshow('YOLO Person Detection (Simplified)', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("🛑 Quitting detection loop")
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"logs/capture_{timestamp}.jpg"
                    os.makedirs("logs", exist_ok=True)
                    cv2.imwrite(filename, display_frame)
                    print(f"📸 Frame saved: {filename}")
        
        except KeyboardInterrupt:
            self.logger.info("🛑 Detection loop interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Error in detection loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
            self.logger.info("✅ System cleanup completed")
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Person Detection System (Simplified)')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--camera', type=int, help='Camera index to use')
    parser.add_argument('--test-camera', action='store_true', help='Test camera connection')
    
    args = parser.parse_args()
    
    try:
        engine = SimplePersonDetectionEngine(args.config)
        
        if args.camera is not None:
            engine.camera_index = args.camera
            engine.initialize_camera()
        
        if args.test_camera:
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
        
        engine.run_detection_loop()
    
    except Exception as e:
        print(f"❌ Error in detection engine: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
