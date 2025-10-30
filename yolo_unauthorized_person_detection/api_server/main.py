"""
FastAPI Server for YOLO Unauthorized Person Detection System
This module provides REST API endpoints for system management and monitoring
"""

import os
import sys
import cv2
import numpy as np
import json
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import asyncio
import threading
import time
from io import BytesIO
from PIL import Image

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_config, setup_logging, ensure_directory_exists
from alert_system.logger import DetectionLogger
from alert_system.alert import AlertSystem
from dataset.prepare_embeddings import FaceEmbeddingProcessor


# Pydantic models for API requests/responses
class PersonAddRequest(BaseModel):
    name: str
    description: Optional[str] = ""


class DetectionEvent(BaseModel):
    timestamp: str
    person_name: str
    confidence: float
    is_authorized: bool
    camera_index: int
    alert_triggered: bool
    alert_type: str


class SystemStats(BaseModel):
    total_detections: int
    authorized_detections: int
    unauthorized_detections: int
    alerts_triggered: int
    recent_detections: int
    most_detected_person: Optional[str]
    most_detected_count: int


class AlertConfig(BaseModel):
    enable_sound: bool
    enable_email: bool
    enable_sms: bool
    alert_cooldown: int


# Global variables
app = FastAPI(
    title="YOLO Unauthorized Person Detection API",
    description="REST API for managing and monitoring the YOLO-based unauthorized person detection system",
    version="1.0.0"
)

# Configuration
config = load_config()
logger = setup_logging(config)
detection_logger = None
alert_system = None
embedding_processor = None

# Camera streaming
camera_stream = None
streaming_active = False
latest_frame = None
frame_lock = threading.Lock()

# CORS middleware
if config.get('api', {}).get('enable_cors', True):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def initialize_components():
    """
    Initialize system components
    """
    global detection_logger, alert_system, embedding_processor
    
    try:
        detection_logger = DetectionLogger()
        alert_system = AlertSystem()
        embedding_processor = FaceEmbeddingProcessor()
        logger.info("✅ API server components initialized")
    except Exception as e:
        logger.error(f"❌ Error initializing API components: {e}")


def generate_frames():
    """
    Generate camera frames for streaming
    """
    global latest_frame, streaming_active
    
    camera = cv2.VideoCapture(config.get('camera', {}).get('camera_index', 0))
    
    if not camera.isOpened():
        logger.error("❌ Cannot open camera for streaming")
        return
    
    streaming_active = True
    
    try:
        while streaming_active:
            ret, frame = camera.read()
            if not ret:
                break
            
            # Resize frame for streaming
            height, width = frame.shape[:2]
            max_width = 640
            if width > max_width:
                scale = max_width / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            # Update latest frame
            with frame_lock:
                latest_frame = frame_bytes
            
            # Yield frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        logger.error(f"❌ Error in frame generation: {e}")
    finally:
        camera.release()
        streaming_active = False


@app.on_event("startup")
async def startup_event():
    """
    Initialize components on startup
    """
    initialize_components()


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on shutdown
    """
    global streaming_active
    streaming_active = False
    logger.info("🛑 API server shutdown")


# Health check endpoint
@app.get("/")
async def root():
    """
    Root endpoint - API information
    """
    return {
        "message": "YOLO Unauthorized Person Detection API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "detection_logger": detection_logger is not None,
            "alert_system": alert_system is not None,
            "embedding_processor": embedding_processor is not None
        }
    }


# Detection events endpoints
@app.get("/api/events", response_model=List[DetectionEvent])
async def get_detection_events(limit: int = 100, authorized_only: bool = False, unauthorized_only: bool = False):
    """
    Get detection events
    """
    if not detection_logger:
        raise HTTPException(status_code=500, detail="Detection logger not initialized")
    
    events = detection_logger.get_detection_events(limit, authorized_only, unauthorized_only)
    return events


@app.get("/api/events/stats", response_model=SystemStats)
async def get_detection_stats():
    """
    Get detection statistics
    """
    if not detection_logger:
        raise HTTPException(status_code=500, detail="Detection logger not initialized")
    
    stats = detection_logger.get_statistics()
    return SystemStats(**stats)


@app.get("/api/events/export")
async def export_events(format: str = "json"):
    """
    Export detection events
    """
    if not detection_logger:
        raise HTTPException(status_code=500, detail="Detection logger not initialized")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detection_events_{timestamp}.{format}"
    filepath = os.path.join("logs", filename)
    
    ensure_directory_exists("logs")
    
    success = detection_logger.export_events(filepath, format)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to export events")
    
    return {"message": f"Events exported to {filename}", "filepath": filepath}


# Authorized persons management
@app.get("/api/authorized-persons")
async def get_authorized_persons():
    """
    Get list of authorized persons
    """
    if not embedding_processor:
        raise HTTPException(status_code=500, detail="Embedding processor not initialized")
    
    person_dirs = embedding_processor.get_person_directories()
    persons = []
    
    for person_dir in person_dirs:
        person_name = os.path.basename(person_dir).replace('_', ' ').title()
        image_count = len(embedding_processor.get_person_images(person_dir))
        persons.append({
            "name": person_name,
            "directory": person_dir,
            "image_count": image_count
        })
    
    return persons


@app.post("/api/authorized-persons")
async def add_authorized_person(request: PersonAddRequest):
    """
    Add new authorized person
    """
    if not detection_logger:
        raise HTTPException(status_code=500, detail="Detection logger not initialized")
    
    success = detection_logger.add_authorized_person(request.name)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to add authorized person")
    
    return {"message": f"Authorized person {request.name} added successfully"}


@app.post("/api/authorized-persons/{person_name}/images")
async def upload_person_images(person_name: str, files: List[UploadFile] = File(...)):
    """
    Upload images for authorized person
    """
    if not embedding_processor:
        raise HTTPException(status_code=500, detail="Embedding processor not initialized")
    
    # Create person directory
    person_dir = os.path.join(embedding_processor.authorized_faces_dir, person_name.lower().replace(' ', '_'))
    ensure_directory_exists(person_dir)
    
    uploaded_files = []
    
    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                continue
            
            # Save file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{person_name.lower().replace(' ', '_')}_{timestamp}_{file.filename}"
            filepath = os.path.join(person_dir, filename)
            
            with open(filepath, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            uploaded_files.append(filename)
        
        except Exception as e:
            logger.error(f"❌ Error uploading file {file.filename}: {e}")
    
    return {
        "message": f"Uploaded {len(uploaded_files)} images for {person_name}",
        "uploaded_files": uploaded_files
    }


@app.post("/api/authorized-persons/{person_name}/update-embeddings")
async def update_person_embeddings(person_name: str):
    """
    Update embeddings for specific person
    """
    if not embedding_processor:
        raise HTTPException(status_code=500, detail="Embedding processor not initialized")
    
    success = embedding_processor.update_person_embeddings(person_name)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update embeddings")
    
    return {"message": f"Embeddings updated for {person_name}"}


@app.post("/api/embeddings/rebuild")
async def rebuild_all_embeddings():
    """
    Rebuild all face embeddings
    """
    if not embedding_processor:
        raise HTTPException(status_code=500, detail="Embedding processor not initialized")
    
    success = embedding_processor.create_embeddings_dataset()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to rebuild embeddings")
    
    return {"message": "All embeddings rebuilt successfully"}


# Alert system endpoints
@app.get("/api/alerts/config")
async def get_alert_config():
    """
    Get alert system configuration
    """
    if not alert_system:
        raise HTTPException(status_code=500, detail="Alert system not initialized")
    
    return {
        "enable_sound": alert_system.enable_sound,
        "enable_email": alert_system.enable_email,
        "enable_sms": alert_system.enable_sms,
        "alert_cooldown": alert_system.alert_cooldown
    }


@app.post("/api/alerts/config")
async def update_alert_config(config: AlertConfig):
    """
    Update alert system configuration
    """
    if not alert_system:
        raise HTTPException(status_code=500, detail="Alert system not initialized")
    
    alert_system.enable_sound = config.enable_sound
    alert_system.enable_email = config.enable_email
    alert_system.enable_sms = config.enable_sms
    alert_system.alert_cooldown = config.alert_cooldown
    
    return {"message": "Alert configuration updated successfully"}


@app.post("/api/alerts/test")
async def test_alerts():
    """
    Test alert system
    """
    if not alert_system:
        raise HTTPException(status_code=500, detail="Alert system not initialized")
    
    alert_system.test_alerts()
    return {"message": "Alert system test completed"}


@app.get("/api/alerts/stats")
async def get_alert_stats():
    """
    Get alert system statistics
    """
    if not alert_system:
        raise HTTPException(status_code=500, detail="Alert system not initialized")
    
    stats = alert_system.get_alert_stats()
    return stats


# Camera streaming endpoints
@app.get("/api/camera/stream")
async def stream_camera():
    """
    Stream camera feed
    """
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/camera/frame")
async def get_latest_frame():
    """
    Get latest camera frame as base64
    """
    with frame_lock:
        if latest_frame is None:
            raise HTTPException(status_code=404, detail="No frame available")
        
        frame_base64 = base64.b64encode(latest_frame).decode('utf-8')
        return {
            "frame": frame_base64,
            "timestamp": datetime.now().isoformat(),
            "format": "jpeg"
        }


@app.get("/api/camera/info")
async def get_camera_info():
    """
    Get camera information
    """
    camera_index = config.get('camera', {}).get('camera_index', 0)
    
    try:
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            raise HTTPException(status_code=404, detail="Camera not available")
        
        ret, frame = camera.read()
        if not ret:
            raise HTTPException(status_code=500, detail="Cannot read from camera")
        
        height, width = frame.shape[:2]
        camera.release()
        
        return {
            "camera_index": camera_index,
            "width": width,
            "height": height,
            "available": True
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting camera info: {e}")


# System management endpoints
@app.get("/api/system/info")
async def get_system_info():
    """
    Get system information
    """
    return {
        "version": "1.0.0",
        "config": {
            "model_path": config.get('model', {}).get('yolo_model_path', 'yolov8n.pt'),
            "camera_index": config.get('camera', {}).get('camera_index', 0),
            "detection_confidence": config.get('model', {}).get('detection_confidence', 0.5),
            "face_tolerance": config.get('model', {}).get('face_tolerance', 0.5)
        },
        "components": {
            "detection_logger": detection_logger is not None,
            "alert_system": alert_system is not None,
            "embedding_processor": embedding_processor is not None
        },
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/system/reload-config")
async def reload_config():
    """
    Reload system configuration
    """
    global config
    config = load_config()
    
    # Reinitialize components with new config
    initialize_components()
    
    return {"message": "Configuration reloaded successfully"}


@app.post("/api/system/cleanup-logs")
async def cleanup_logs(days: int = 30):
    """
    Cleanup old log entries
    """
    if not detection_logger:
        raise HTTPException(status_code=500, detail="Detection logger not initialized")
    
    success = detection_logger.cleanup_old_events(days)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to cleanup logs")
    
    return {"message": f"Cleaned up logs older than {days} days"}


# GPIO control endpoints (for Raspberry Pi)
@app.post("/api/gpio/trigger-lock")
async def trigger_lock():
    """
    Trigger door lock (placeholder for GPIO control)
    """
    # This is a placeholder for GPIO control
    # In a real implementation, you would control GPIO pins here
    
    logger.info("🔒 Door lock triggered (placeholder)")
    
    return {"message": "Door lock triggered", "status": "success"}


@app.post("/api/gpio/trigger-alarm")
async def trigger_alarm():
    """
    Trigger alarm (placeholder for GPIO control)
    """
    # This is a placeholder for GPIO control
    # In a real implementation, you would control GPIO pins here
    
    logger.info("🚨 Alarm triggered (placeholder)")
    
    return {"message": "Alarm triggered", "status": "success"}


# Static files for dashboard
dashboard_path = os.path.join(os.path.dirname(__file__), "..", "dashboard")
if os.path.exists(dashboard_path):
    app.mount("/dashboard", StaticFiles(directory=dashboard_path), name="dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """
    Serve dashboard HTML
    """
    dashboard_file = os.path.join(dashboard_path, "index.html")
    if os.path.exists(dashboard_file):
        with open(dashboard_file, 'r') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>Dashboard not found</h1>")


def run_server():
    """
    Run the FastAPI server
    """
    api_config = config.get('api', {})
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    
    logger.info(f"🌐 Starting API server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()
