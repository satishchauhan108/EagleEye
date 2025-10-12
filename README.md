# 🛡️ EagleEye Unauthorized Person Detection System

A complete, modular, and production-ready system for detecting unauthorized persons using YOLOv8 and face recognition. Built with Python, FastAPI, OpenCV, and Ultralytics YOLOv8.

## 🎯 Overview

This system provides real-time person detection and face recognition to identify authorized vs unauthorized individuals. It includes:

- **Real-time YOLOv8 person detection** from camera feed
- **Face recognition** using deep learning embeddings
- **Alert system** with sound, email, and SMS notifications
- **REST API** for system management and monitoring
- **Web dashboard** for real-time monitoring
- **Comprehensive logging** and event tracking
- **Modular architecture** for easy deployment and customization

## 🏗️ System Architecture

```
Camera Feed → YOLOv8 Detection → Face Extraction → Face Recognition
     ↓              ↓                ↓              ↓
  Display      Bounding Boxes    Face Region    Authorized?
     ↓              ↓                ↓              ↓
  Dashboard    Green/Red Boxes   Face Encoding   Alert System
     ↓              ↓                ↓              ↓
  API Server   Event Logging    Database        Notifications
```

## 📁 Project Structure

```
yolo_unauthorized_person_detection/
├── project_setup/
│   ├── requirements.txt          # Python dependencies
│   ├── setup_env.sh             # Linux/Unix setup script
│   └── setup_env.bat            # Windows setup script
├── dataset/
│   ├── authorized_faces/         # Authorized person face images
│   ├── test_images/             # Test images
│   ├── capture_faces.py         # Face capture script
│   └── prepare_embeddings.py    # Face embedding preparation
├── model_training/
│   └── train_yolo.py            # YOLO model training
├── inference_engine/
│   └── detect_and_identify.py   # Core detection engine
├── api_server/
│   └── main.py                  # FastAPI REST API server
├── alert_system/
│   ├── alert.py                 # Alert system (sound, email, SMS)
│   └── logger.py                # Event logging system
├── dashboard/
│   └── index.html               # Web dashboard
├── utils/
│   └── helpers.py               # Utility functions
├── config/
│   └── config.yaml              # System configuration
├── logs/                        # Log files and events
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. System Requirements

- **Python 3.8+**
- **Camera** (webcam or IP camera)
- **4GB+ RAM** (8GB+ recommended)
- **GPU** (optional, for better performance)

### 2. Installation

#### Option A: Automated Setup (Recommended)

**Windows:**
```bash
# Run the setup script
setup_env.bat
```

**Linux/Unix:**
```bash
# Make script executable and run
chmod +x project_setup/setup_env.sh
./project_setup/setup_env.sh
```

#### Option B: Manual Setup

```bash
# 1. Create virtual environment
python -m venv yolo_env

# 2. Activate environment
# Windows:
yolo_env\Scripts\activate
# Linux/Unix:
source yolo_env/bin/activate

# 3. Install dependencies
pip install -r project_setup/requirements.txt

# 4. Create directories
mkdir -p dataset/authorized_faces dataset/test_images logs models

# 5. Download YOLOv8 model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 3. Configuration

Edit `config/config.yaml` to customize system settings:

```yaml
# Camera settings
camera:
  camera_index: 0          # Camera index (0 for default webcam)
  frame_width: 640
  frame_height: 480

# Model settings
model:
  detection_confidence: 0.5    # YOLO detection confidence threshold
  face_tolerance: 0.5          # Face recognition tolerance

# Alert settings
alerts:
  enable_sound: true
  enable_email: false
  enable_sms: false
```

### 4. Add Authorized Persons

#### Method 1: Interactive Face Capture
```bash
# Capture faces interactively
python dataset/capture_faces.py --interactive

# Or capture specific person
python dataset/capture_faces.py --person "John Doe"
```

#### Method 2: Manual Image Addition
1. Create directory: `dataset/authorized_faces/john_doe/`
2. Add face images (JPG/PNG format)
3. Run embedding preparation:
```bash
python dataset/prepare_embeddings.py
```

### 5. Run the System

#### Start Detection Engine
```bash
# Start real-time detection
python inference_engine/detect_and_identify.py

# With custom camera
python inference_engine/detect_and_identify.py --camera 1

# Test camera connection
python inference_engine/detect_and_identify.py --test-camera
```

#### Start API Server (Optional)
```bash
# Start FastAPI server
uvicorn api_server.main:app --reload --host 0.0.0.0 --port 8000

# Access dashboard at: http://localhost:8000/dashboard
```

## 📋 Usage Guide

### Basic Operation

1. **Start Detection**: Run the detection engine
2. **Monitor Feed**: Watch the camera feed with bounding boxes
3. **Check Logs**: Review detection events in `logs/events.csv`
4. **Manage Alerts**: Configure alert settings in `config/config.yaml`

### Key Controls

- **'q'**: Quit detection system
- **'s'**: Save current frame
- **'r'**: Reload face embeddings

### Example Output

```
[Authorized] Satish Chauhan - Confidence: 0.91 ✅
[Unauthorized] Unknown - Confidence: 0.33 🚫 ALERT
```

## 🔧 Configuration Options

### Camera Configuration
```yaml
camera:
  camera_index: 0          # Camera device index
  frame_width: 640         # Frame width
  frame_height: 480        # Frame height
  fps: 30                  # Target FPS
```

### Model Configuration
```yaml
model:
  yolo_model_path: "yolov8n.pt"    # YOLO model path
  detection_confidence: 0.5         # Detection threshold
  face_tolerance: 0.5               # Face recognition tolerance
  max_faces_per_frame: 10           # Max faces to process
```

### Alert Configuration
```yaml
alerts:
  enable_sound: true                # Enable sound alerts
  enable_email: false               # Enable email alerts
  enable_sms: false                 # Enable SMS alerts
  alert_cooldown: 30                # Alert cooldown (seconds)

email:
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  sender_email: "your_email@gmail.com"
  sender_password: "your_app_password"
  recipient_email: "admin@company.com"

sms:
  account_sid: "your_twilio_sid"
  auth_token: "your_twilio_token"
  from_number: "+1234567890"
  to_number: "+0987654321"
```

## 🌐 API Endpoints

### Detection Events
- `GET /api/events` - Get detection events
- `GET /api/events/stats` - Get detection statistics
- `GET /api/events/export` - Export events to file

### Authorized Persons
- `GET /api/authorized-persons` - List authorized persons
- `POST /api/authorized-persons` - Add new authorized person
- `POST /api/authorized-persons/{name}/images` - Upload person images
- `POST /api/embeddings/rebuild` - Rebuild all embeddings

### Alert System
- `GET /api/alerts/config` - Get alert configuration
- `POST /api/alerts/config` - Update alert configuration
- `POST /api/alerts/test` - Test alert system

### Camera
- `GET /api/camera/stream` - Stream camera feed
- `GET /api/camera/frame` - Get latest frame
- `GET /api/camera/info` - Get camera information

### System
- `GET /health` - Health check
- `GET /api/system/info` - System information
- `POST /api/system/reload-config` - Reload configuration

## 📊 Web Dashboard

Access the dashboard at `http://localhost:8000/dashboard` to:

- **View live camera feed** with detection overlays
- **Monitor detection statistics** in real-time
- **Review recent detection events** with filtering
- **Manage authorized persons** and embeddings
- **Test alert systems** and export data
- **Control system settings** and configuration

## 🔍 Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Test camera connection
python inference_engine/detect_and_identify.py --test-camera

# Try different camera indices
python inference_engine/detect_and_identify.py --camera 1
```

#### No Face Recognition
```bash
# Check if embeddings exist
python dataset/prepare_embeddings.py --validate

# Rebuild embeddings
python dataset/prepare_embeddings.py
```

#### Performance Issues
```yaml
# Reduce frame processing
performance:
  frame_skip: 2          # Process every 2nd frame
  use_gpu: true          # Enable GPU acceleration
```

#### Alert System Not Working
```bash
# Test alert system
python alert_system/alert.py --test

# Check configuration
python -c "from utils.helpers import load_config; print(load_config())"
```

### Log Files

- `logs/detection.log` - System logs
- `logs/events.csv` - Detection events
- `logs/detection_events.db` - SQLite database
- `logs/system.log` - System events

## 🚀 Deployment

### Local Development
```bash
# Start detection engine
python inference_engine/detect_and_identify.py

# Start API server (separate terminal)
uvicorn api_server.main:app --reload
```

### Production Deployment
```bash
# Use production WSGI server
gunicorn api_server.main:app -w 4 -k uvicorn.workers.UvicornWorker

# Run detection as service
python inference_engine/detect_and_identify.py --config config/production.yaml
```

### Raspberry Pi Deployment

1. **Install dependencies**:
```bash
sudo apt-get update
sudo apt-get install python3-pip python3-venv
```

2. **Setup GPIO** (optional):
```yaml
raspberry_pi:
  enable_gpio: true
  door_lock_pin: 18
  alarm_pin: 19
  led_pin: 21
```

3. **Run with reduced resources**:
```yaml
performance:
  frame_skip: 3
  use_gpu: false
  num_threads: 2
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r project_setup/requirements.txt

EXPOSE 8000
CMD ["uvicorn", "api_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔒 Security Considerations

### Data Protection
- Face embeddings are stored locally
- No cloud processing of sensitive data
- Configurable face blurring for unauthorized persons
- Secure API key authentication

### Access Control
- Configurable alert thresholds
- Unauthorized attempt tracking
- System lockout mechanisms
- Audit logging for all events

### Privacy
- Face data stored in local database
- Configurable data retention policies
- GDPR-compliant data handling
- Optional face anonymization

## 🧪 Testing

### Unit Tests
```bash
# Run face capture test
python dataset/capture_faces.py --person "Test Person"

# Test embedding preparation
python dataset/prepare_embeddings.py --validate

# Test alert system
python alert_system/alert.py --test

# Test logger
python alert_system/logger.py --test
```

### Integration Tests
```bash
# Test full pipeline
python inference_engine/detect_and_identify.py --test-camera

# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/events/stats
```

## 📈 Performance Optimization

### GPU Acceleration
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Model Optimization
```bash
# Export to ONNX for faster inference
python model_training/train_yolo.py --export onnx

# Use TensorRT for NVIDIA GPUs
python model_training/train_yolo.py --export tensorrt
```

### System Tuning
```yaml
performance:
  frame_skip: 2              # Process every 2nd frame
  num_threads: 4             # CPU threads
  use_gpu: true              # GPU acceleration
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv8 implementation
- **OpenCV** for computer vision capabilities
- **Face Recognition** library for face recognition
- **FastAPI** for the REST API framework
- **Community contributors** for testing and feedback

## 📞 Support

- **Documentation**: Check this README and code comments
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join community discussions
- **Email**: Contact the development team

## 🔄 Changelog

### Version 1.0.0
- Initial release with core functionality
- YOLOv8 person detection
- Face recognition system
- Alert system (sound, email, SMS)
- REST API with comprehensive endpoints
- Web dashboard for monitoring
- Comprehensive logging and event tracking
- Modular architecture for easy deployment

---

**Built with ❤️ for security and safety**

