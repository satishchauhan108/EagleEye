#!/bin/bash

# YOLO Unauthorized Person Detection System - Setup Script
# This script sets up the environment for Linux/Unix systems
# For Windows, use setup_env.bat

echo "🚀 Setting up YOLO Unauthorized Person Detection System..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv yolo_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source yolo_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install system dependencies (Ubuntu/Debian)
echo "🔧 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y cmake build-essential libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r project_setup/requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p dataset/authorized_faces
mkdir -p dataset/test_images
mkdir -p logs
mkdir -p models

# Download YOLOv8 pretrained model
echo "🤖 Downloading YOLOv8 pretrained model..."
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Create example authorized face directory
echo "👤 Creating example authorized face..."
mkdir -p dataset/authorized_faces/satish_chauhan

echo "✅ Setup complete!"
echo "🔧 To activate the environment, run: source yolo_env/bin/activate"
echo "🚀 To start detection, run: python inference_engine/detect_and_identify.py"
echo "🌐 To start API server, run: uvicorn api_server.main:app --reload"
