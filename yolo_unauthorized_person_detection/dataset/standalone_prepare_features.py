"""
Standalone Face Features Preparation Script for YOLO Unauthorized Person Detection System
This script processes captured face images and creates basic face recognition data using OpenCV
"""

import os
import sys
import json
import numpy as np
import cv2
import yaml
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional


class StandaloneFaceProcessor:
    """
    Standalone face processor using OpenCV features
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize face processor
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.paths_config = self.config.get('paths', {})
        
        # Paths
        self.authorized_faces_dir = self.paths_config.get('authorized_faces_dir', 'dataset/authorized_faces')
        self.embeddings_file = self.paths_config.get('embeddings_file', 'dataset/face_embeddings.npy')
        self.authorized_persons_file = self.paths_config.get('authorized_persons_file', 'dataset/authorized_persons.json')
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Processing settings
        self.min_images_per_person = 1  # Minimum images required per person
        
        print("Standalone Face Processor Initialized")
        print(f"Authorized faces directory: {self.authorized_faces_dir}")
        print(f"Embeddings file: {self.embeddings_file}")
        print(f"Persons file: {self.authorized_persons_file}")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
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
            print(f"Configuration file not found: {config_path}")
            print("Using default configuration")
            return {
                'paths': {
                    'authorized_faces_dir': 'dataset/authorized_faces',
                    'embeddings_file': 'dataset/face_embeddings.npy',
                    'authorized_persons_file': 'dataset/authorized_persons.json'
                }
            }
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            return {
                'paths': {
                    'authorized_faces_dir': 'dataset/authorized_faces',
                    'embeddings_file': 'dataset/face_embeddings.npy',
                    'authorized_persons_file': 'dataset/authorized_persons.json'
                }
            }
    
    def ensure_directory_exists(self, directory_path: str) -> bool:
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
            print(f"Error creating directory {directory_path}: {e}")
            return False
    
    def get_person_directories(self) -> List[str]:
        """
        Get list of person directories
        
        Returns:
            List of person directory paths
        """
        if not os.path.exists(self.authorized_faces_dir):
            print(f"ERROR Authorized faces directory not found: {self.authorized_faces_dir}")
            return []
        
        person_dirs = []
        for item in os.listdir(self.authorized_faces_dir):
            person_path = os.path.join(self.authorized_faces_dir, item)
            if os.path.isdir(person_path):
                person_dirs.append(person_path)
        
        return sorted(person_dirs)
    
    def get_person_images(self, person_dir: str) -> List[str]:
        """
        Get list of image files for a person
        
        Args:
            person_dir: Person directory path
            
        Returns:
            List of image file paths
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for file in os.listdir(person_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(person_dir, file))
        
        return sorted(image_files)
    
    def extract_face_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face features using OpenCV
        
        Args:
            image: Input image
            
        Returns:
            Face features vector or None if no face found
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            # Use the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to standard size
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Flatten to create feature vector
            features = face_roi.flatten()
            
            # Normalize features
            features = features.astype(np.float32) / 255.0
            
            return features
            
        except Exception as e:
            print(f"ERROR Error extracting face features: {e}")
            return None
    
    def process_person_images(self, person_dir: str, person_name: str) -> List[np.ndarray]:
        """
        Process all images for a person and extract face features
        
        Args:
            person_dir: Person directory path
            person_name: Person name
            
        Returns:
            List of face feature vectors
        """
        image_files = self.get_person_images(person_dir)
        
        if len(image_files) < self.min_images_per_person:
            print(f"WARNING Insufficient images for {person_name}: {len(image_files)} < {self.min_images_per_person}")
            return []
        
        print(f"PROCESSING Processing {len(image_files)} images for {person_name}...")
        
        features = []
        successful_extractions = 0
        
        for i, image_path in enumerate(image_files):
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"WARNING Could not load image: {image_path}")
                    continue
                
                # Extract face features
                face_features = self.extract_face_features(image)
                if face_features is not None:
                    features.append(face_features)
                    successful_extractions += 1
                    print(f"SUCCESS Processed image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
                else:
                    print(f"WARNING No face found in image: {os.path.basename(image_path)}")
            
            except Exception as e:
                print(f"ERROR Error processing image {image_path}: {e}")
                continue
        
        print(f"SUCCESS Successfully processed {successful_extractions}/{len(image_files)} images for {person_name}")
        
        if successful_extractions < self.min_images_per_person:
            print(f"WARNING Insufficient successful extractions for {person_name}: {successful_extractions} < {self.min_images_per_person}")
            return []
        
        return features
    
    def create_features_dataset(self) -> bool:
        """
        Create features dataset from all authorized persons
        
        Returns:
            True if successful, False otherwise
        """
        person_dirs = self.get_person_directories()
        
        if not person_dirs:
            print("ERROR No person directories found")
            return False
        
        print(f"PROCESSING Processing {len(person_dirs)} authorized persons...")
        
        all_features = []
        all_names = []
        successful_persons = 0
        
        for person_dir in person_dirs:
            person_name = os.path.basename(person_dir).replace('_', ' ').title()
            
            # Process person images
            features = self.process_person_images(person_dir, person_name)
            
            if features:
                # Add all features for this person
                all_features.extend(features)
                all_names.extend([person_name] * len(features))
                successful_persons += 1
                print(f"SUCCESS Added {len(features)} feature vectors for {person_name}")
            else:
                print(f"ERROR Failed to process {person_name}")
        
        if not all_features:
            print("ERROR No features created")
            return False
        
        # Save features and metadata
        success = self.save_features_dataset(all_features, all_names)
        
        if success:
            print(f"SUCCESS Successfully created features dataset")
            print(f"STATS Total feature vectors: {len(all_features)}")
            print(f"PEOPLE Successful persons: {successful_persons}/{len(person_dirs)}")
            print(f"SAVE Saved to: {self.embeddings_file}")
            print(f"PEOPLE Metadata saved to: {self.authorized_persons_file}")
        else:
            print("ERROR Failed to save features dataset")
        
        return success
    
    def save_features_dataset(self, features: List[np.ndarray], names: List[str]) -> bool:
        """
        Save face features and person names
        
        Args:
            features: List of face feature vectors
            names: List of person names
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directories if they don't exist
            self.ensure_directory_exists(os.path.dirname(self.embeddings_file))
            self.ensure_directory_exists(os.path.dirname(self.authorized_persons_file))
            
            # Save features
            if features:
                np.save(self.embeddings_file, np.array(features))
            
            # Load existing persons data
            if os.path.exists(self.authorized_persons_file):
                with open(self.authorized_persons_file, 'r') as f:
                    persons_data = json.load(f)
            else:
                persons_data = []
            
            # Update persons data with new entries
            existing_names = [person['name'] for person in persons_data]
            for name in set(names):
                if name not in existing_names:
                    persons_data.append({
                        'name': name,
                        'id': len(persons_data),
                        'created_at': '2025-01-11 18:30:00',
                        'description': f'Authorized person from provided images',
                        'is_active': True
                    })
            
            # Save updated persons data
            with open(self.authorized_persons_file, 'w') as f:
                json.dump(persons_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"ERROR Error saving features dataset: {e}")
            return False


def main():
    """
    Main function for face features preparation script
    """
    parser = argparse.ArgumentParser(description='Prepare face features for authorized persons')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        processor = StandaloneFaceProcessor(args.config)
        
        # Create features dataset
        print("TARGET Creating features dataset for all authorized persons...")
        success = processor.create_features_dataset()
        sys.exit(0 if success else 1)
    
    except Exception as e:
        print(f"ERROR Error in feature preparation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
