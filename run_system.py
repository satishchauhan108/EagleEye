#!/usr/bin/env python3
"""
Main system runner for YOLO Unauthorized Person Detection System
This script provides an easy way to start the entire system
"""

import os
import sys
import argparse
import subprocess
import threading
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    # Map pip package names to actual Python module names
    required_packages = {
        'ultralytics': 'ultralytics',
        'opencv-python': 'cv2',
        'face_recognition': 'face_recognition',
        'numpy': 'numpy',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'pyyaml': 'yaml'
    }

    missing_packages = []
    for package, module_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("pip install -r project_setup/requirements.txt")
        return False

    return True

def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Camera is working correctly")
                return True
        print("❌ Camera not available or not working")
        return False
    except Exception as e:
        print(f"❌ Error checking camera: {e}")
        return False

def check_embeddings():
    """Check if face embeddings exist"""
    embeddings_file = "dataset/face_embeddings.npy"
    persons_file = "dataset/authorized_persons.json"

    if os.path.exists(embeddings_file) and os.path.exists(persons_file):
        print("✅ Face embeddings found")
        return True
    else:
        print("⚠️ Face embeddings not found")
        print("📝 Run the following to create embeddings:")
        print("python dataset/capture_faces.py --interactive")
        print("python dataset/prepare_embeddings.py")
        return False

def start_detection_engine():
    """Start the detection engine"""
    print("🚀 Starting detection engine...")
    try:
        subprocess.run([sys.executable, "inference_engine/detect_and_identify.py"], check=True)
    except KeyboardInterrupt:
        print("🛑 Detection engine stopped")
    except Exception as e:
        print(f"❌ Error starting detection engine: {e}")

def start_api_server():
    """Start the API server"""
    print("🌐 Starting API server...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "api_server.main:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000"
        ], check=True)
    except KeyboardInterrupt:
        print("🛑 API server stopped")
    except Exception as e:
        print(f"❌ Error starting API server: {e}")

def main():
    parser = argparse.ArgumentParser(description='YOLO Unauthorized Person Detection System Runner')
    parser.add_argument('--mode', choices=['detection', 'api', 'both'], default='detection',
                       help='Run mode: detection only, API only, or both')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check system requirements')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip system checks and start directly')

    args = parser.parse_args()

    print("🛡️ YOLO Unauthorized Person Detection System")
    print("=" * 50)

    if not args.skip_checks:
        print("🔍 Checking system requirements...")

        # Check dependencies
        if not check_dependencies():
            sys.exit(1)

        # Check camera
        if not check_camera():
            print("⚠️ Camera check failed, but continuing...")

        # Check embeddings
        embeddings_ok = check_embeddings()
        if not embeddings_ok:
            response = input("Continue without embeddings? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)

        print("✅ System checks completed")

    if args.check_only:
        print("✅ All checks passed. System is ready to run.")
        return

    print("\n🚀 Starting system...")

    if args.mode == 'detection':
        start_detection_engine()
    elif args.mode == 'api':
        start_api_server()
    elif args.mode == 'both':
        # Start API server in background thread
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()

        # Give API server time to start
        time.sleep(3)
        print("🌐 API server started at http://localhost:8000")
        print("📊 Dashboard available at http://localhost:8000/dashboard")

        # Start detection engine in main thread
        start_detection_engine()

if __name__ == "__main__":
    main()
