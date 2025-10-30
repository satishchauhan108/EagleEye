"""
Logging System for YOLO Unauthorized Person Detection
This module handles logging of detection events and system activities
"""

import os
import sys
import csv
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_config, create_timestamp, ensure_directory_exists


class DetectionLogger:
    """
    Logger for detection events and system activities
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize detection logger
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.paths_config = self.config.get('paths', {})
        self.logging_config = self.config.get('logging', {})
        
        # Paths
        self.logs_dir = self.paths_config.get('logs_dir', 'logs')
        ensure_directory_exists(self.logs_dir)
        
        # Log files
        self.events_csv = os.path.join(self.logs_dir, 'events.csv')
        self.detection_log = os.path.join(self.logs_dir, 'detection.log')
        self.system_log = os.path.join(self.logs_dir, 'system.log')
        
        # Database settings
        self.db_path = os.path.join(self.logs_dir, 'detection_events.db')
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize database
        self.init_database()
        
        # Initialize CSV file
        self.init_csv_file()
        
        print("📝 Detection Logger Initialized")
        print(f"📁 Logs directory: {self.logs_dir}")
        print(f"💾 Database: {self.db_path}")
        print(f"📊 Events CSV: {self.events_csv}")
    
    def setup_logging(self):
        """
        Setup logging configuration
        """
        import logging
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup detection logger
        self.detection_logger = logging.getLogger('detection')
        self.detection_logger.setLevel(logging.INFO)
        
        # File handler for detection log
        detection_handler = logging.FileHandler(self.detection_log)
        detection_handler.setFormatter(formatter)
        self.detection_logger.addHandler(detection_handler)
        
        # Setup system logger
        self.system_logger = logging.getLogger('system')
        self.system_logger.setLevel(logging.INFO)
        
        # File handler for system log
        system_handler = logging.FileHandler(self.system_log)
        system_handler.setFormatter(formatter)
        self.system_logger.addHandler(system_handler)
        
        # Console handler
        if self.logging_config.get('enable_console_log', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.detection_logger.addHandler(console_handler)
            self.system_logger.addHandler(console_handler)
    
    def init_database(self):
        """
        Initialize SQLite database for detection events
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create detection_events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    person_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_authorized BOOLEAN NOT NULL,
                    camera_index INTEGER,
                    frame_width INTEGER,
                    frame_height INTEGER,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    alert_triggered BOOLEAN DEFAULT FALSE,
                    alert_type TEXT,
                    notes TEXT
                )
            ''')
            
            # Create system_events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_message TEXT NOT NULL,
                    severity TEXT DEFAULT 'INFO',
                    details TEXT
                )
            ''')
            
            # Create authorized_persons table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS authorized_persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen TEXT,
                    total_detections INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT TRUE
                )
            ''')
            
            conn.commit()
            conn.close()
            
            print("✅ Database initialized successfully")
        
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
    
    def init_csv_file(self):
        """
        Initialize CSV file for detection events
        """
        try:
            if not os.path.exists(self.events_csv):
                with open(self.events_csv, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'person_name', 'confidence', 'is_authorized',
                        'camera_index', 'frame_width', 'frame_height',
                        'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                        'alert_triggered', 'alert_type', 'notes'
                    ])
            
            print("✅ CSV file initialized successfully")
        
        except Exception as e:
            print(f"❌ Error initializing CSV file: {e}")
    
    def log_detection_event(self, person_name: str, confidence: float, 
                          is_authorized: bool, bbox: Optional[tuple] = None,
                          camera_index: int = 0, frame_size: Optional[tuple] = None,
                          alert_triggered: bool = False, alert_type: str = '',
                          notes: str = '') -> bool:
        """
        Log detection event to database and CSV
        
        Args:
            person_name: Name of detected person
            confidence: Detection confidence
            is_authorized: Whether person is authorized
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            camera_index: Camera index used
            frame_size: Frame size (width, height)
            alert_triggered: Whether alert was triggered
            alert_type: Type of alert triggered
            notes: Additional notes
            
        Returns:
            True if successful, False otherwise
        """
        timestamp = create_timestamp()
        
        # Prepare data
        bbox_x1 = bbox_y1 = bbox_x2 = bbox_y2 = None
        if bbox:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
        
        frame_width = frame_height = None
        if frame_size:
            frame_width, frame_height = frame_size
        
        try:
            # Log to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO detection_events 
                (timestamp, person_name, confidence, is_authorized, camera_index,
                 frame_width, frame_height, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                 alert_triggered, alert_type, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, person_name, confidence, is_authorized, camera_index,
                frame_width, frame_height, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                alert_triggered, alert_type, notes
            ))
            
            conn.commit()
            conn.close()
            
            # Log to CSV
            with open(self.events_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, person_name, confidence, is_authorized,
                    camera_index, frame_width, frame_height,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    alert_triggered, alert_type, notes
                ])
            
            # Log to detection logger
            status = "AUTHORIZED" if is_authorized else "UNAUTHORIZED"
            alert_info = f" | Alert: {alert_type}" if alert_triggered else ""
            self.detection_logger.info(
                f"[{status}] {person_name} - Confidence: {confidence:.2f}{alert_info}"
            )
            
            return True
        
        except Exception as e:
            print(f"❌ Error logging detection event: {e}")
            return False
    
    def log_system_event(self, event_type: str, message: str, 
                        severity: str = 'INFO', details: str = '') -> bool:
        """
        Log system event
        
        Args:
            event_type: Type of system event
            message: Event message
            severity: Event severity (INFO, WARNING, ERROR)
            details: Additional details
            
        Returns:
            True if successful, False otherwise
        """
        timestamp = create_timestamp()
        
        try:
            # Log to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_events 
                (timestamp, event_type, event_message, severity, details)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, event_type, message, severity, details))
            
            conn.commit()
            conn.close()
            
            # Log to system logger
            if severity == 'ERROR':
                self.system_logger.error(f"[{event_type}] {message}")
            elif severity == 'WARNING':
                self.system_logger.warning(f"[{event_type}] {message}")
            else:
                self.system_logger.info(f"[{event_type}] {message}")
            
            return True
        
        except Exception as e:
            print(f"❌ Error logging system event: {e}")
            return False
    
    def add_authorized_person(self, name: str) -> bool:
        """
        Add authorized person to database
        
        Args:
            name: Name of authorized person
            
        Returns:
            True if successful, False otherwise
        """
        timestamp = create_timestamp()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO authorized_persons 
                (name, created_at, is_active)
                VALUES (?, ?, ?)
            ''', (name, timestamp, True))
            
            conn.commit()
            conn.close()
            
            self.log_system_event('AUTHORIZED_PERSON_ADDED', f'Added authorized person: {name}')
            return True
        
        except Exception as e:
            print(f"❌ Error adding authorized person: {e}")
            return False
    
    def get_detection_events(self, limit: int = 100, 
                           authorized_only: bool = False,
                           unauthorized_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get detection events from database
        
        Args:
            limit: Maximum number of events to return
            authorized_only: Return only authorized detections
            unauthorized_only: Return only unauthorized detections
            
        Returns:
            List of detection events
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM detection_events"
            params = []
            
            if authorized_only:
                query += " WHERE is_authorized = 1"
            elif unauthorized_only:
                query += " WHERE is_authorized = 0"
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            columns = [description[0] for description in cursor.description]
            events = [dict(zip(columns, row)) for row in rows]
            
            conn.close()
            return events
        
        except Exception as e:
            print(f"❌ Error getting detection events: {e}")
            return []
    
    def get_system_events(self, limit: int = 100, 
                         severity: str = None) -> List[Dict[str, Any]]:
        """
        Get system events from database
        
        Args:
            limit: Maximum number of events to return
            severity: Filter by severity level
            
        Returns:
            List of system events
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM system_events"
            params = []
            
            if severity:
                query += " WHERE severity = ?"
                params.append(severity)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            columns = [description[0] for description in cursor.description]
            events = [dict(zip(columns, row)) for row in rows]
            
            conn.close()
            return events
        
        except Exception as e:
            print(f"❌ Error getting system events: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detection statistics
        
        Returns:
            Dictionary with statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total detections
            cursor.execute("SELECT COUNT(*) FROM detection_events")
            total_detections = cursor.fetchone()[0]
            
            # Authorized detections
            cursor.execute("SELECT COUNT(*) FROM detection_events WHERE is_authorized = 1")
            authorized_detections = cursor.fetchone()[0]
            
            # Unauthorized detections
            cursor.execute("SELECT COUNT(*) FROM detection_events WHERE is_authorized = 0")
            unauthorized_detections = cursor.fetchone()[0]
            
            # Alerts triggered
            cursor.execute("SELECT COUNT(*) FROM detection_events WHERE alert_triggered = 1")
            alerts_triggered = cursor.fetchone()[0]
            
            # Recent activity (last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) FROM detection_events 
                WHERE datetime(timestamp) >= datetime('now', '-1 day')
            ''')
            recent_detections = cursor.fetchone()[0]
            
            # Most detected person
            cursor.execute('''
                SELECT person_name, COUNT(*) as count 
                FROM detection_events 
                GROUP BY person_name 
                ORDER BY count DESC 
                LIMIT 1
            ''')
            most_detected = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_detections': total_detections,
                'authorized_detections': authorized_detections,
                'unauthorized_detections': unauthorized_detections,
                'alerts_triggered': alerts_triggered,
                'recent_detections': recent_detections,
                'most_detected_person': most_detected[0] if most_detected else None,
                'most_detected_count': most_detected[1] if most_detected else 0
            }
        
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            return {}
    
    def export_events(self, output_file: str, format: str = 'csv') -> bool:
        """
        Export detection events to file
        
        Args:
            output_file: Output file path
            format: Export format ('csv', 'json')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            events = self.get_detection_events(limit=10000)  # Export all events
            
            if format.lower() == 'csv':
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    if events:
                        writer = csv.DictWriter(f, fieldnames=events[0].keys())
                        writer.writeheader()
                        writer.writerows(events)
            
            elif format.lower() == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(events, f, indent=2, default=str)
            
            else:
                print(f"❌ Unsupported export format: {format}")
                return False
            
            print(f"✅ Events exported to {output_file}")
            return True
        
        except Exception as e:
            print(f"❌ Error exporting events: {e}")
            return False
    
    def cleanup_old_events(self, days: int = 30) -> bool:
        """
        Clean up old detection events
        
        Args:
            days: Number of days to keep
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete old detection events
            cursor.execute('''
                DELETE FROM detection_events 
                WHERE datetime(timestamp) < datetime('now', '-{} days')
            '''.format(days))
            
            deleted_count = cursor.rowcount
            
            # Delete old system events
            cursor.execute('''
                DELETE FROM system_events 
                WHERE datetime(timestamp) < datetime('now', '-{} days')
            '''.format(days))
            
            deleted_system_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            print(f"✅ Cleaned up {deleted_count} detection events and {deleted_system_count} system events")
            return True
        
        except Exception as e:
            print(f"❌ Error cleaning up old events: {e}")
            return False


def main():
    """
    Main function for testing logger
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Test detection logger')
    parser.add_argument('--test', action='store_true', help='Test logging functions')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--export', type=str, help='Export events to file')
    parser.add_argument('--cleanup', type=int, help='Clean up events older than N days')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        logger = DetectionLogger(args.config)
        
        if args.test:
            # Test logging functions
            print("🧪 Testing logging functions...")
            
            # Test detection event logging
            logger.log_detection_event(
                "Satish Chauhan", 0.95, True, (100, 100, 200, 200),
                camera_index=0, frame_size=(640, 480), alert_triggered=False
            )
            
            logger.log_detection_event(
                "Unknown Person", 0.85, False, (150, 150, 250, 250),
                camera_index=0, frame_size=(640, 480), alert_triggered=True, alert_type="sound"
            )
            
            # Test system event logging
            logger.log_system_event("SYSTEM_START", "Detection system started")
            logger.log_system_event("CAMERA_CONNECTED", "Camera 0 connected successfully")
            
            print("✅ Logging test completed")
        
        elif args.stats:
            # Show statistics
            stats = logger.get_statistics()
            print("📊 Detection Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        elif args.export:
            # Export events
            format = 'json' if args.export.endswith('.json') else 'csv'
            success = logger.export_events(args.export, format)
            sys.exit(0 if success else 1)
        
        elif args.cleanup:
            # Cleanup old events
            success = logger.cleanup_old_events(args.cleanup)
            sys.exit(0 if success else 1)
        
        else:
            # Default: show recent events
            events = logger.get_detection_events(limit=10)
            print("📝 Recent Detection Events:")
            for event in events:
                status = "✅" if event['is_authorized'] else "🚫"
                print(f"   {status} {event['timestamp']} - {event['person_name']} ({event['confidence']:.2f})")
    
    except Exception as e:
        print(f"❌ Error in logger: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
