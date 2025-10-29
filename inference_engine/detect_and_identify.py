"""
Core Detection and Identification Engine for YOLO Unauthorized Person Detection System
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import (
    load_config, setup_logging, draw_bounding_box, extract_face_region,
    compute_face_encoding, load_authorized_embeddings, recognize_face,
    create_timestamp, format_detection_result
)

class PersonDetectionEngine:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config)
        
        # Configuration
        self.model_config = self.config.get('model', {})
        self.camera_config = self.config.get('camera', {})
        self.paths_config = self.config.get('paths', {})
        self.display_config = self.config.get('display', {})
        
        # Settings
        self.model_path = self.model_config.get('yolo_model_path', 'yolov8n.pt')
        self.detection_confidence = self.model_config.get('detection_confidence', 0.5)
        self.face_tolerance = self.model_config.get('face_tolerance', 0.5)
        self.camera_index = self.camera_config.get('camera_index', 0)
        
        # Paths
        self.embeddings_file = self.paths_config.get('embeddings_file', 'dataset/face_embeddings.npy')
        self.authorized_persons_file = self.paths_config.get('authorized_persons_file', 'dataset/authorized_persons.json')
        
        # Initialize components
        self.yolo_model = None
        self.authorized_embeddings = []
        self.authorized_names = []
        self.camera = None
        
        self.initialize_components()
        self.logger.info("✅ Person Detection Engine initialized successfully")
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            self.initialize_yolo_model()
            self.load_authorized_embeddings()
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
    
    def load_authorized_embeddings(self):
        """Load authorized face embeddings"""
        try:
            self.logger.info("👤 Loading authorized face embeddings...")
            self.authorized_embeddings, self.authorized_names = load_authorized_embeddings(
                self.embeddings_file, self.authorized_persons_file
            )
            
            if self.authorized_embeddings:
                unique_persons = set(self.authorized_names)
                self.logger.info(f"✅ Loaded {len(self.authorized_embeddings)} embeddings for {len(unique_persons)} persons")
            else:
                self.logger.warning("⚠️ No authorized embeddings found. System will treat all persons as unauthorized.")
        except Exception as e:
            self.logger.error(f"❌ Error loading authorized embeddings: {e}")
            self.authorized_embeddings = []
            self.authorized_names = []
    
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
    
    def identify_person(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[Optional[str], float]:
        """Identify person using face recognition"""
        try:
            face_region = extract_face_region(frame, bbox)
            if face_region is None:
                return None, 0.0
            
            face_encoding = compute_face_encoding(face_region)
            if face_encoding is None:
                return None, 0.0
            
            person_name, confidence = recognize_face(
                face_encoding, self.authorized_embeddings, self.authorized_names, self.face_tolerance
            )
            
            return person_name, confidence
        except Exception as e:
            self.logger.error(f"❌ Error in person identification: {e}")
            return None, 0.0
    
    def process_detection(self, person_name: str, confidence: float, bbox: Tuple[int, int, int, int]) -> bool:
        """Process detection result"""
        is_authorized = person_name is not None
        
        # Print detection result
        result_text = format_detection_result(person_name or "Unknown", confidence, is_authorized)
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
        
        return display_frame
    
    def run_detection_loop(self):
        """Main detection loop"""
        self.logger.info("🎬 Starting detection loop...")
        print("🎬 Starting YOLO Unauthorized Person Detection System")
        print("📋 Press 'q' to quit, 's' to save frame, 'r' to reload embeddings")
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
                    
                    # Identify person
                    person_name, face_confidence = self.identify_person(frame, bbox)
                    
                    # Process detection
                    is_authorized = self.process_detection(person_name, confidence, bbox)
                    
                    # Add to processed detections
                    detection['person_name'] = person_name or "Unknown"
                    detection['is_authorized'] = is_authorized
                    detection['face_confidence'] = face_confidence
                    processed_detections.append(detection)
                
                # Draw results
                display_frame = self.draw_detection_results(frame, processed_detections)
                
                # Show frame
                cv2.imshow('YOLO Unauthorized Person Detection', display_frame)
                
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
                elif key == ord('r'):
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
    
    parser = argparse.ArgumentParser(description='YOLO Unauthorized Person Detection System')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--camera', type=int, help='Camera index to use')
    parser.add_argument('--test-camera', action='store_true', help='Test camera connection')
    
    args = parser.parse_args()
    
    try:
        engine = PersonDetectionEngine(args.config)
        
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