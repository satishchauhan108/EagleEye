"""
Face Embeddings Preparation Script for YOLO Unauthorized Person Detection System
This script processes captured face images and creates face embeddings for recognition
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
import argparse
from typing import List, Dict, Any
import face_recognition

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import (
    load_config, ensure_directory_exists, compute_face_encoding,
    save_authorized_embeddings, load_authorized_embeddings
)


class FaceEmbeddingProcessor:
    """
    Face embedding processor for creating face recognition data
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize face embedding processor
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.paths_config = self.config.get('paths', {})
        
        # Paths
        self.authorized_faces_dir = self.paths_config.get('authorized_faces_dir', 'dataset/authorized_faces')
        self.embeddings_file = self.paths_config.get('embeddings_file', 'dataset/face_embeddings.npy')
        self.authorized_persons_file = self.paths_config.get('authorized_persons_file', 'dataset/authorized_persons.json')
        
        # Processing settings
        self.min_images_per_person = 3  # Minimum images required per person
        self.max_images_per_person = 50  # Maximum images to process per person
        
        print("🧠 Face Embedding Processor Initialized")
        print(f"📁 Authorized faces directory: {self.authorized_faces_dir}")
        print(f"💾 Embeddings file: {self.embeddings_file}")
        print(f"👥 Persons file: {self.authorized_persons_file}")
    
    def get_person_directories(self) -> List[str]:
        """
        Get list of person directories
        
        Returns:
            List of person directory paths
        """
        if not os.path.exists(self.authorized_faces_dir):
            print(f"❌ Authorized faces directory not found: {self.authorized_faces_dir}")
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
    
    def process_person_images(self, person_dir: str, person_name: str) -> List[np.ndarray]:
        """
        Process all images for a person and extract face embeddings
        
        Args:
            person_dir: Person directory path
            person_name: Person name
            
        Returns:
            List of face embeddings
        """
        image_files = self.get_person_images(person_dir)
        
        if len(image_files) < self.min_images_per_person:
            print(f"⚠️ Insufficient images for {person_name}: {len(image_files)} < {self.min_images_per_person}")
            return []
        
        print(f"🔄 Processing {len(image_files)} images for {person_name}...")
        
        embeddings = []
        successful_encodings = 0
        
        # Limit number of images to process
        images_to_process = image_files[:self.max_images_per_person]
        
        for i, image_path in enumerate(images_to_process):
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"⚠️ Could not load image: {image_path}")
                    continue
                
                # Compute face encoding
                encoding = compute_face_encoding(image)
                if encoding is not None:
                    embeddings.append(encoding)
                    successful_encodings += 1
                    print(f"✅ Processed image {i+1}/{len(images_to_process)}: {os.path.basename(image_path)}")
                else:
                    print(f"⚠️ No face found in image: {os.path.basename(image_path)}")
            
            except Exception as e:
                print(f"❌ Error processing image {image_path}: {e}")
                continue
        
        print(f"✅ Successfully processed {successful_encodings}/{len(images_to_process)} images for {person_name}")
        
        if successful_encodings < self.min_images_per_person:
            print(f"⚠️ Insufficient successful encodings for {person_name}: {successful_encodings} < {self.min_images_per_person}")
            return []
        
        return embeddings
    
    def create_embeddings_dataset(self) -> bool:
        """
        Create embeddings dataset from all authorized persons
        
        Returns:
            True if successful, False otherwise
        """
        person_dirs = self.get_person_directories()
        
        if not person_dirs:
            print("❌ No person directories found")
            return False
        
        print(f"🔄 Processing {len(person_dirs)} authorized persons...")
        
        all_embeddings = []
        all_names = []
        successful_persons = 0
        
        for person_dir in person_dirs:
            person_name = os.path.basename(person_dir).replace('_', ' ').title()
            
            # Process person images
            embeddings = self.process_person_images(person_dir, person_name)
            
            if embeddings:
                # Add all embeddings for this person
                all_embeddings.extend(embeddings)
                all_names.extend([person_name] * len(embeddings))
                successful_persons += 1
                print(f"✅ Added {len(embeddings)} embeddings for {person_name}")
            else:
                print(f"❌ Failed to process {person_name}")
        
        if not all_embeddings:
            print("❌ No embeddings created")
            return False
        
        # Save embeddings and metadata
        success = save_authorized_embeddings(
            all_embeddings, all_names,
            self.embeddings_file, self.authorized_persons_file
        )
        
        if success:
            print(f"✅ Successfully created embeddings dataset")
            print(f"📊 Total embeddings: {len(all_embeddings)}")
            print(f"👥 Successful persons: {successful_persons}/{len(person_dirs)}")
            print(f"💾 Saved to: {self.embeddings_file}")
            print(f"👥 Metadata saved to: {self.authorized_persons_file}")
        else:
            print("❌ Failed to save embeddings dataset")
        
        return success
    
    def validate_embeddings(self) -> bool:
        """
        Validate existing embeddings dataset
        
        Returns:
            True if embeddings are valid, False otherwise
        """
        try:
            embeddings, names = load_authorized_embeddings(
                self.embeddings_file, self.authorized_persons_file
            )
            
            if not embeddings or not names:
                print("❌ No embeddings found")
                return False
            
            print(f"✅ Embeddings dataset validation successful")
            print(f"📊 Total embeddings: {len(embeddings)}")
            
            # Count unique persons
            unique_persons = set(names)
            print(f"👥 Unique persons: {len(unique_persons)}")
            
            for person in unique_persons:
                count = names.count(person)
                print(f"   - {person}: {count} embeddings")
            
            return True
        
        except Exception as e:
            print(f"❌ Error validating embeddings: {e}")
            return False
    
    def update_person_embeddings(self, person_name: str) -> bool:
        """
        Update embeddings for a specific person
        
        Args:
            person_name: Name of person to update
            
        Returns:
            True if successful, False otherwise
        """
        person_dir = os.path.join(self.authorized_faces_dir, person_name.lower().replace(' ', '_'))
        
        if not os.path.exists(person_dir):
            print(f"❌ Person directory not found: {person_dir}")
            return False
        
        # Load existing embeddings
        existing_embeddings, existing_names = load_authorized_embeddings(
            self.embeddings_file, self.authorized_persons_file
        )
        
        # Remove existing embeddings for this person
        filtered_embeddings = []
        filtered_names = []
        
        for embedding, name in zip(existing_embeddings, existing_names):
            if name != person_name:
                filtered_embeddings.append(embedding)
                filtered_names.append(name)
        
        # Process new images for this person
        new_embeddings = self.process_person_images(person_dir, person_name)
        
        if not new_embeddings:
            print(f"❌ No valid embeddings created for {person_name}")
            return False
        
        # Combine embeddings
        all_embeddings = filtered_embeddings + new_embeddings
        all_names = filtered_names + [person_name] * len(new_embeddings)
        
        # Save updated embeddings
        success = save_authorized_embeddings(
            all_embeddings, all_names,
            self.embeddings_file, self.authorized_persons_file
        )
        
        if success:
            print(f"✅ Successfully updated embeddings for {person_name}")
            print(f"📊 Total embeddings: {len(all_embeddings)}")
        else:
            print(f"❌ Failed to update embeddings for {person_name}")
        
        return success
    
    def interactive_processing(self):
        """
        Interactive embedding processing session
        """
        print("🧠 Interactive Face Embedding Processing")
        print("=" * 50)
        
        while True:
            print("\n" + "=" * 50)
            print("Options:")
            print("1. Create embeddings dataset (all persons)")
            print("2. Update specific person embeddings")
            print("3. Validate existing embeddings")
            print("4. List authorized persons")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                print("\n🔄 Creating embeddings dataset...")
                success = self.create_embeddings_dataset()
                if success:
                    print("✅ Embeddings dataset created successfully")
                else:
                    print("❌ Failed to create embeddings dataset")
            
            elif choice == '2':
                person_dirs = self.get_person_directories()
                if not person_dirs:
                    print("❌ No person directories found")
                    continue
                
                print("Available persons:")
                for i, person_dir in enumerate(person_dirs, 1):
                    person_name = os.path.basename(person_dir).replace('_', ' ').title()
                    print(f"{i}. {person_name}")
                
                try:
                    person_index = int(input("Select person number: ")) - 1
                    if 0 <= person_index < len(person_dirs):
                        person_name = os.path.basename(person_dirs[person_index]).replace('_', ' ').title()
                        print(f"\n🔄 Updating embeddings for {person_name}...")
                        success = self.update_person_embeddings(person_name)
                        if success:
                            print(f"✅ Successfully updated embeddings for {person_name}")
                        else:
                            print(f"❌ Failed to update embeddings for {person_name}")
                    else:
                        print("❌ Invalid person number")
                except ValueError:
                    print("❌ Invalid input")
            
            elif choice == '3':
                print("\n🔍 Validating embeddings...")
                success = self.validate_embeddings()
                if success:
                    print("✅ Embeddings validation successful")
                else:
                    print("❌ Embeddings validation failed")
            
            elif choice == '4':
                person_dirs = self.get_person_directories()
                if person_dirs:
                    print("\n👥 Authorized persons:")
                    for person_dir in person_dirs:
                        person_name = os.path.basename(person_dir).replace('_', ' ').title()
                        image_count = len(self.get_person_images(person_dir))
                        print(f"   - {person_name}: {image_count} images")
                else:
                    print("👥 No authorized persons found")
            
            elif choice == '5':
                print("👋 Exiting embedding processing session")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-5.")


def main():
    """
    Main function for face embedding preparation script
    """
    parser = argparse.ArgumentParser(description='Prepare face embeddings for authorized persons')
    parser.add_argument('--person', type=str, help='Name of person to update embeddings')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--validate', action='store_true', help='Validate existing embeddings')
    
    args = parser.parse_args()
    
    try:
        processor = FaceEmbeddingProcessor(args.config)
        
        if args.validate:
            # Validate existing embeddings
            success = processor.validate_embeddings()
            sys.exit(0 if success else 1)
        
        elif args.person:
            # Update specific person
            success = processor.update_person_embeddings(args.person)
            sys.exit(0 if success else 1)
        
        elif args.interactive:
            # Interactive mode
            processor.interactive_processing()
        
        else:
            # Default: create embeddings dataset
            print("🎯 Creating embeddings dataset for all authorized persons...")
            success = processor.create_embeddings_dataset()
            sys.exit(0 if success else 1)
    
    except Exception as e:
        print(f"❌ Error in embedding preparation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
