"""
Utility functions for YOLO Unauthorized Person Detection System
This module contains helper functions used across the project
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
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        print("📝 Please ensure config/config.yaml exists")
        return {}
    except yaml.YAMLError as e:
        print(f"❌ Error parsing configuration file: {e}")
        return {}


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured logger instance
    """
    log_config = config.get('logging', {})
    
    # Create logs directory if it doesn't exist
    log_file = log_config.get('log_file', 'logs/detection.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config.get('log_level', 'INFO')),
        format=log_config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if log_config.get('enable_console_log', True) else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def draw_bounding_box(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    color: Tuple[int, int, int],
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Draw bounding box with label and confidence on image
    
    Args:
        image: Input image
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        label: Label text
        confidence: Confidence score
        color: BGR color tuple
        config: Configuration dictionary
        
    Returns:
        Image with drawn bounding box
    """
    x1, y1, x2, y2 = bbox
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Prepare label text
    display_config = config.get('display', {})
    if display_config.get('show_confidence', True):
        label_text = f"{label}: {confidence:.2f}"
    else:
        label_text = label
    
    # Calculate text size
    font_scale = display_config.get('font_scale', 0.7)
    font_thickness = display_config.get('font_thickness', 2)
    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    # Draw label background
    cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
    
    # Draw label text
    text_color = tuple(display_config.get('colors', {}).get('text', [255, 255, 255]))
    cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    
    return image


def extract_face_region(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Extract face region from image using bounding box
    
    Args:
        image: Input image
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        
    Returns:
        Extracted face region or None if extraction fails
    """
    try:
        x1, y1, x2, y2 = bbox
        
        # Add padding to face region
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        # Extract face region
        face_region = image[y1:y2, x1:x2]
        
        # Check if face region is valid
        if face_region.size == 0:
            return None
            
        return face_region
    except Exception as e:
        print(f"❌ Error extracting face region: {e}")
        return None


def compute_face_encoding(face_image: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute face encoding for face recognition
    
    Args:
        face_image: Face image as numpy array
        
    Returns:
        Face encoding or None if encoding fails
    """
    try:
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Compute face encoding
        encodings = face_recognition.face_encodings(rgb_face)
        
        if len(encodings) > 0:
            return encodings[0]
        else:
            return None
    except Exception as e:
        print(f"❌ Error computing face encoding: {e}")
        return None


def load_authorized_embeddings(embeddings_file: str, persons_file: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load authorized face embeddings and person names
    
    Args:
        embeddings_file: Path to embeddings file
        persons_file: Path to authorized persons metadata file
        
    Returns:
        Tuple of (embeddings list, names list)
    """
    try:
        # Load embeddings
        if os.path.exists(embeddings_file):
            embeddings = np.load(embeddings_file)
            embeddings = [embeddings[i] for i in range(len(embeddings))]
        else:
            embeddings = []
        
        # Load person names
        if os.path.exists(persons_file):
            with open(persons_file, 'r') as f:
                persons_data = json.load(f)
            names = [person['name'] for person in persons_data]
        else:
            names = []
        
        # Ensure lengths match
        min_length = min(len(embeddings), len(names))
        embeddings = embeddings[:min_length]
        names = names[:min_length]
        
        return embeddings, names
    except Exception as e:
        print(f"❌ Error loading authorized embeddings: {e}")
        return [], []


def save_authorized_embeddings(embeddings: List[np.ndarray], names: List[str], 
                              embeddings_file: str, persons_file: str) -> bool:
    """
    Save authorized face embeddings and person names
    
    Args:
        embeddings: List of face embeddings
        names: List of person names
        embeddings_file: Path to save embeddings
        persons_file: Path to save person metadata
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(embeddings_file), exist_ok=True)
        os.makedirs(os.path.dirname(persons_file), exist_ok=True)
        
        # Save embeddings
        if embeddings:
            np.save(embeddings_file, np.array(embeddings))
        
        # Save person metadata
        persons_data = [{'name': name, 'id': i} for i, name in enumerate(names)]
        with open(persons_file, 'w') as f:
            json.dump(persons_data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"❌ Error saving authorized embeddings: {e}")
        return False


def recognize_face(face_encoding: np.ndarray, authorized_embeddings: List[np.ndarray], 
                  authorized_names: List[str], tolerance: float = 0.5) -> Tuple[Optional[str], float]:
    """
    Recognize face by comparing with authorized embeddings
    
    Args:
        face_encoding: Face encoding to recognize
        authorized_embeddings: List of authorized face embeddings
        authorized_names: List of authorized person names
        tolerance: Face recognition tolerance
        
    Returns:
        Tuple of (recognized_name, confidence) or (None, 0.0) if not recognized
    """
    if not authorized_embeddings or not authorized_names:
        return None, 0.0
    
    try:
        # Compare face encoding with authorized embeddings
        matches = face_recognition.compare_faces(authorized_embeddings, face_encoding, tolerance=tolerance)
        face_distances = face_recognition.face_distance(authorized_embeddings, face_encoding)
        
        if matches:
            # Find the best match
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                confidence = 1.0 - face_distances[best_match_index]
                return authorized_names[best_match_index], confidence
        
        return None, 0.0
    except Exception as e:
        print(f"❌ Error in face recognition: {e}")
        return None, 0.0


def blur_face(image: np.ndarray, bbox: Tuple[int, int, int, int], intensity: int = 50) -> np.ndarray:
    """
    Blur face region in image
    
    Args:
        image: Input image
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        intensity: Blur intensity (0-100)
        
    Returns:
        Image with blurred face
    """
    try:
        x1, y1, x2, y2 = bbox
        
        # Extract face region
        face_region = image[y1:y2, x1:x2]
        
        # Apply Gaussian blur
        kernel_size = max(1, intensity // 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred_face = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 0)
        
        # Replace face region with blurred version
        image[y1:y2, x1:x2] = blurred_face
        
        return image
    except Exception as e:
        print(f"❌ Error blurring face: {e}")
        return image


def calculate_fps(start_time: float, frame_count: int) -> float:
    """
    Calculate FPS based on start time and frame count
    
    Args:
        start_time: Start time in seconds
        frame_count: Number of frames processed
        
    Returns:
        FPS value
    """
    try:
        elapsed_time = datetime.now().timestamp() - start_time
        if elapsed_time > 0:
            return frame_count / elapsed_time
        return 0.0
    except Exception:
        return 0.0


def create_timestamp() -> str:
    """
    Create timestamp string for logging
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory_path: Path to directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"❌ Error creating directory {directory_path}: {e}")
        return False


def get_camera_info() -> List[Dict[str, Any]]:
    """
    Get information about available cameras
    
    Returns:
        List of camera information dictionaries
    """
    cameras = []
    
    # Test cameras 0-9
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'available': True
                })
            cap.release()
        else:
            cap.release()
    
    return cameras


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    required_sections = ['model', 'camera', 'paths', 'alerts', 'api', 'logging']
    
    for section in required_sections:
        if section not in config:
            print(f"❌ Missing required configuration section: {section}")
            return False
    
    # Validate specific parameters
    if config['model']['detection_confidence'] < 0 or config['model']['detection_confidence'] > 1:
        print("❌ Invalid detection_confidence value (must be between 0 and 1)")
        return False
    
    if config['camera']['camera_index'] < 0:
        print("❌ Invalid camera_index value (must be >= 0)")
        return False
    
    return True


def format_detection_result(name: str, confidence: float, is_authorized: bool) -> str:
    """
    Format detection result for logging
    
    Args:
        name: Person name or "Unknown"
        confidence: Detection confidence
        is_authorized: Whether person is authorized
        
    Returns:
        Formatted result string
    """
    status = "✅" if is_authorized else "🚫 ALERT"
    auth_status = "Authorized" if is_authorized else "Unauthorized"
    
    return f"[{auth_status}] {name} - Confidence: {confidence:.2f} {status}"
