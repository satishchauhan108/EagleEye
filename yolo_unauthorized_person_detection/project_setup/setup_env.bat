@echo off
REM YOLO Unauthorized Person Detection System - Windows Setup Script

echo 🚀 Setting up YOLO Unauthorized Person Detection System...

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv yolo_env

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call yolo_env\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install Python dependencies
echo 📚 Installing Python dependencies...
pip install -r project_setup\requirements.txt

REM Create necessary directories
echo 📁 Creating project directories...
if not exist "dataset\authorized_faces" mkdir dataset\authorized_faces
if not exist "dataset\test_images" mkdir dataset\test_images
if not exist "logs" mkdir logs
if not exist "models" mkdir models

REM Download YOLOv8 pretrained model
echo 🤖 Downloading YOLOv8 pretrained model...
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

REM Create example authorized face directory
echo 👤 Creating example authorized face...
if not exist "dataset\authorized_faces\satish_chauhan" mkdir dataset\authorized_faces\satish_chauhan

echo ✅ Setup complete!
echo 🔧 To activate the environment, run: yolo_env\Scripts\activate.bat
echo 🚀 To start detection, run: python inference_engine\detect_and_identify.py
echo 🌐 To start API server, run: uvicorn api_server.main:app --reload

pause
