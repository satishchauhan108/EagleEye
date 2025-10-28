"""
Utility functions for YOLO Unauthorized Person Detection System
"""
import os
import cv2
import numpy as np
import yaml
import json
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import face_recognition

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

def extract_face_region(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Extract face region from image using bounding box"""
    try:
        x1, y1, x2, y2 = bbox
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        face_region = image[y1:y2, x1:x2]
        return face_region if face_region.size > 0 else None
    except Exception as e:
        print(f"❌ Error extracting face region: {e}")
        return None

def compute_face_encoding(face_image: np.ndarray) -> Optional[np.ndarray]:
    """Compute face encoding for face recognition"""
    try:
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_face)
        return encodings[0] if len(encodings) > 0 else None
    except Exception as e:
        print(f"❌ Error computing face encoding: {e}")
        return None

def load_authorized_embeddings(embeddings_file: str, persons_file: str) -> Tuple[List[np.ndarray], List[str]]:
    """Load authorized face embeddings and person names"""
    try:
        embeddings = []
        if os.path.exists(embeddings_file):
            embeddings = np.load(embeddings_file)
            embeddings = [embeddings[i] for i in range(len(embeddings))]
        
        names = []
        if os.path.exists(persons_file):
            with open(persons_file, 'r') as f:
                persons_data = json.load(f)
            names = [person['name'] for person in persons_data]
        
        min_length = min(len(embeddings), len(names))
        return embeddings[:min_length], names[:min_length]
    except Exception as e:
        print(f"❌ Error loading authorized embeddings: {e}")
        return [], []

def recognize_face(face_encoding: np.ndarray, authorized_embeddings: List[np.ndarray], 
                  authorized_names: List[str], tolerance: float = 0.5) -> Tuple[Optional[str], float]:
    """Recognize face by comparing with authorized embeddings"""
    if not authorized_embeddings or not authorized_names:
        return None, 0.0
    
    try:
        matches = face_recognition.compare_faces(authorized_embeddings, face_encoding, tolerance=tolerance)
        face_distances = face_recognition.face_distance(authorized_embeddings, face_encoding)
        
        if matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                confidence = 1.0 - face_distances[best_match_index]
                return authorized_names[best_match_index], confidence
        
        return None, 0.0
    except Exception as e:
        print(f"❌ Error in face recognition: {e}")
        return None, 0.0

def create_timestamp() -> str:
    """Create timestamp string for logging"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def format_detection_result(name: str, confidence: float, is_authorized: bool) -> str:
    """Format detection result for logging"""
    status = "✅" if is_authorized else "🚫 ALERT"
    auth_status = "Authorized" if is_authorized else "Unauthorized"
    return f"[{auth_status}] {name} - Confidence: {confidence:.2f} {status}"