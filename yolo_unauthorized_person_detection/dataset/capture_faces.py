"""
Face Capture Script for YOLO Unauthorized Person Detection System
This script captures face images from webcam for authorized persons
"""

import cv2
import os
import sys
import json
from datetime import datetime
from pathlib import Path
import argparse

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_config, ensure_directory_exists, create_timestamp


class FaceCapture:
    """
    Face capture class for collecting authorized person face data
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize face capture system
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.camera_config = self.config.get('camera', {})
        self.paths_config = self.config.get('paths', {})
        
        # Camera settings
        self.camera_index = self.camera_config.get('camera_index', 0)
        self.frame_width = self.camera_config.get('frame_width', 640)
        self.frame_height = self.camera_config.get('frame_height', 480)
        
        # Paths
        self.authorized_faces_dir = self.paths_config.get('authorized_faces_dir', 'dataset/authorized_faces')
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Capture settings
        self.images_per_person = 20  # Number of images to capture per person
        self.capture_interval = 0.5  # Seconds between captures
        
        print("🎥 Face Capture System Initialized")
        print(f"📁 Authorized faces directory: {self.authorized_faces_dir}")
        print(f"📷 Camera index: {self.camera_index}")
        print(f"🖼️ Images per person: {self.images_per_person}")
    
    def setup_camera(self) -> cv2.VideoCapture:
        """
        Setup camera with specified configuration
        
        Returns:
            Configured camera object
        """
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            raise Exception(f"❌ Cannot open camera {self.camera_index}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        return cap
    
    def detect_faces(self, frame: cv2.Mat) -> list:
        """
        Detect faces in frame using Haar cascade
        
        Args:
            frame: Input frame
            
        Returns:
            List of face bounding boxes
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        return faces
    
    def capture_person_faces(self, person_name: str) -> bool:
        """
        Capture face images for a specific person
        
        Args:
            person_name: Name of the person to capture
            
        Returns:
            True if successful, False otherwise
        """
        # Create person directory
        person_dir = os.path.join(self.authorized_faces_dir, person_name.lower().replace(' ', '_'))
        ensure_directory_exists(person_dir)
        
        print(f"\n👤 Capturing faces for: {person_name}")
        print(f"📁 Saving to: {person_dir}")
        print("📋 Instructions:")
        print("   - Look directly at the camera")
        print("   - Move your head slightly for different angles")
        print("   - Press SPACE to capture image")
        print("   - Press 'q' to quit")
        print("   - Press 's' to skip current person")
        
        cap = self.setup_camera()
        captured_count = 0
        
        try:
            while captured_count < self.images_per_person:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame from camera")
                    break
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Draw face detection rectangles
                display_frame = frame.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add instructions
                cv2.putText(display_frame, f"Person: {person_name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Captured: {captured_count}/{self.images_per_person}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "SPACE: Capture | 'q': Quit | 's': Skip", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Face Capture', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("🛑 Quitting face capture")
                    break
                elif key == ord('s'):
                    print("⏭️ Skipping current person")
                    break
                elif key == ord(' '):  # Space bar
                    if len(faces) > 0:
                        # Use the largest face
                        largest_face = max(faces, key=lambda f: f[2] * f[3])
                        x, y, w, h = largest_face
                        
                        # Extract and save face
                        face_roi = frame[y:y+h, x:x+w]
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{person_name.lower().replace(' ', '_')}_{timestamp}_{captured_count:03d}.jpg"
                        filepath = os.path.join(person_dir, filename)
                        
                        cv2.imwrite(filepath, face_roi)
                        captured_count += 1
                        print(f"✅ Captured image {captured_count}/{self.images_per_person}: {filename}")
                    else:
                        print("⚠️ No face detected. Please position yourself in front of the camera.")
        
        except KeyboardInterrupt:
            print("\n🛑 Face capture interrupted by user")
        except Exception as e:
            print(f"❌ Error during face capture: {e}")
            return False
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"✅ Face capture completed for {person_name}")
        print(f"📊 Total images captured: {captured_count}")
        
        return captured_count > 0
    
    def list_existing_persons(self) -> list:
        """
        List existing authorized persons
        
        Returns:
            List of person names
        """
        if not os.path.exists(self.authorized_faces_dir):
            return []
        
        persons = []
        for item in os.listdir(self.authorized_faces_dir):
            person_path = os.path.join(self.authorized_faces_dir, item)
            if os.path.isdir(person_path):
                persons.append(item.replace('_', ' ').title())
        
        return sorted(persons)
    
    def interactive_capture(self):
        """
        Interactive face capture session
        """
        print("🎥 Interactive Face Capture Session")
        print("=" * 50)
        
        # List existing persons
        existing_persons = self.list_existing_persons()
        if existing_persons:
            print(f"👥 Existing authorized persons: {', '.join(existing_persons)}")
        else:
            print("👥 No existing authorized persons found")
        
        while True:
            print("\n" + "=" * 50)
            print("Options:")
            print("1. Add new person")
            print("2. Re-capture existing person")
            print("3. List existing persons")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                person_name = input("Enter person name: ").strip()
                if person_name:
                    self.capture_person_faces(person_name)
                else:
                    print("❌ Invalid person name")
            
            elif choice == '2':
                if not existing_persons:
                    print("❌ No existing persons to re-capture")
                    continue
                
                print("Existing persons:")
                for i, person in enumerate(existing_persons, 1):
                    print(f"{i}. {person}")
                
                try:
                    person_index = int(input("Select person number: ")) - 1
                    if 0 <= person_index < len(existing_persons):
                        person_name = existing_persons[person_index]
                        self.capture_person_faces(person_name)
                    else:
                        print("❌ Invalid person number")
                except ValueError:
                    print("❌ Invalid input")
            
            elif choice == '3':
                existing_persons = self.list_existing_persons()
                if existing_persons:
                    print(f"👥 Authorized persons: {', '.join(existing_persons)}")
                else:
                    print("👥 No authorized persons found")
            
            elif choice == '4':
                print("👋 Exiting face capture session")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-4.")


def main():
    """
    Main function for face capture script
    """
    parser = argparse.ArgumentParser(description='Capture face images for authorized persons')
    parser.add_argument('--person', type=str, help='Name of person to capture')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    try:
        face_capture = FaceCapture(args.config)
        
        if args.person:
            # Capture specific person
            success = face_capture.capture_person_faces(args.person)
            if success:
                print(f"✅ Successfully captured faces for {args.person}")
            else:
                print(f"❌ Failed to capture faces for {args.person}")
        elif args.interactive:
            # Interactive mode
            face_capture.interactive_capture()
        else:
            # Default: capture example person
            print("🎯 No person specified. Capturing example person: Satish Chauhan")
            success = face_capture.capture_person_faces("Satish Chauhan")
            if success:
                print("✅ Successfully captured example faces")
            else:
                print("❌ Failed to capture example faces")
    
    except Exception as e:
        print(f"❌ Error in face capture: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
