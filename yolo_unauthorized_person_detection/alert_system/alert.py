"""
Alert System for YOLO Unauthorized Person Detection
This module handles various types of alerts (sound, email, SMS)
"""

import os
import sys
import smtplib
import json
import time
from datetime import datetime
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.image import MimeImage
from typing import Optional, Dict, Any
import cv2
import numpy as np

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_config, create_timestamp, ensure_directory_exists


class AlertSystem:
    """
    Alert system for unauthorized person detection
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize alert system
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.alerts_config = self.config.get('alerts', {})
        self.email_config = self.config.get('email', {})
        self.sms_config = self.config.get('sms', {})
        
        # Alert settings
        self.enable_sound = self.alerts_config.get('enable_sound', True)
        self.enable_email = self.alerts_config.get('enable_email', False)
        self.enable_sms = self.alerts_config.get('enable_sms', False)
        self.alert_cooldown = self.alerts_config.get('alert_cooldown', 30)
        
        # Sound settings
        self.alert_sound_path = self.alerts_config.get('alert_sound_path', 'alert_system/alert.wav')
        
        # Email settings
        self.smtp_server = self.email_config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = self.email_config.get('smtp_port', 587)
        self.sender_email = self.email_config.get('sender_email', '')
        self.sender_password = self.email_config.get('sender_password', '')
        self.recipient_email = self.email_config.get('recipient_email', '')
        
        # SMS settings (Twilio)
        self.twilio_account_sid = self.sms_config.get('account_sid', '')
        self.twilio_auth_token = self.sms_config.get('auth_token', '')
        self.twilio_from_number = self.sms_config.get('from_number', '')
        self.twilio_to_number = self.sms_config.get('to_number', '')
        
        # Alert tracking
        self.last_alert_time = 0
        self.alert_count = 0
        
        # Create alert sound if it doesn't exist
        self.create_alert_sound()
        
        print("🚨 Alert System Initialized")
        print(f"🔊 Sound alerts: {'Enabled' if self.enable_sound else 'Disabled'}")
        print(f"📧 Email alerts: {'Enabled' if self.enable_email else 'Disabled'}")
        print(f"📱 SMS alerts: {'Enabled' if self.enable_sms else 'Disabled'}")
    
    def create_alert_sound(self):
        """
        Create alert sound file if it doesn't exist
        """
        if os.path.exists(self.alert_sound_path):
            return
        
        try:
            # Create directory if it doesn't exist
            ensure_directory_exists(os.path.dirname(self.alert_sound_path))
            
            # Generate a simple beep sound using numpy and scipy
            try:
                import scipy.io.wavfile as wavfile
                
                # Generate beep sound
                sample_rate = 44100
                duration = 1.0  # seconds
                frequency = 800  # Hz
                
                t = np.linspace(0, duration, int(sample_rate * duration))
                beep = 0.3 * np.sin(2 * np.pi * frequency * t)
                
                # Convert to 16-bit integers
                beep_int = (beep * 32767).astype(np.int16)
                
                # Save as WAV file
                wavfile.write(self.alert_sound_path, sample_rate, beep_int)
                print(f"✅ Alert sound created: {self.alert_sound_path}")
            
            except ImportError:
                print("⚠️ scipy not available. Using pygame for sound generation.")
                self.create_pygame_alert_sound()
        
        except Exception as e:
            print(f"⚠️ Could not create alert sound: {e}")
            print("ℹ️ You can manually add an alert sound file to: alert_system/alert.wav")
    
    def create_pygame_alert_sound(self):
        """
        Create alert sound using pygame
        """
        try:
            import pygame
            
            # Initialize pygame mixer
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            # Generate beep sound
            sample_rate = 22050
            duration = 1.0
            frequency = 800
            
            frames = int(duration * sample_rate)
            arr = np.zeros((frames, 2), dtype=np.int16)
            
            for i in range(frames):
                wave = 4096 * np.sin(2 * np.pi * frequency * i / sample_rate)
                arr[i][0] = int(wave)  # Left channel
                arr[i][1] = int(wave)  # Right channel
            
            # Create sound object
            sound = pygame.sndarray.make_sound(arr)
            sound.play()
            
            print("✅ Alert sound test successful (pygame)")
        
        except ImportError:
            print("⚠️ pygame not available. Please install pygame or add alert.wav manually.")
        except Exception as e:
            print(f"⚠️ Error creating pygame alert sound: {e}")
    
    def play_sound_alert(self) -> bool:
        """
        Play sound alert
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_sound:
            return False
        
        try:
            # Try playsound first
            try:
                from playsound import playsound
                playsound(self.alert_sound_path)
                return True
            except ImportError:
                pass
            
            # Try pygame
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(self.alert_sound_path)
                pygame.mixer.music.play()
                return True
            except ImportError:
                pass
            
            # Try system beep as fallback
            import winsound if os.name == 'nt' else os.system
            if os.name == 'nt':
                winsound.Beep(1000, 1000)  # Windows beep
            else:
                os.system('echo -e "\a"')  # Unix beep
            
            return True
        
        except Exception as e:
            print(f"❌ Error playing sound alert: {e}")
            return False
    
    def send_email_alert(self, person_name: str, confidence: float, 
                        frame: Optional[np.ndarray] = None) -> bool:
        """
        Send email alert for unauthorized person
        
        Args:
            person_name: Name of detected person
            confidence: Detection confidence
            frame: Optional frame image to attach
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_email or not self.sender_email or not self.recipient_email:
            return False
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = f"🚨 Unauthorized Person Detected - {create_timestamp()}"
            
            # Create email body
            body = f"""
            <html>
            <body>
                <h2>🚨 Security Alert</h2>
                <p><strong>Unauthorized person detected!</strong></p>
                <ul>
                    <li><strong>Person:</strong> {person_name}</li>
                    <li><strong>Confidence:</strong> {confidence:.2f}</li>
                    <li><strong>Time:</strong> {create_timestamp()}</li>
                    <li><strong>Location:</strong> Camera Feed</li>
                </ul>
                <p>Please check the security system immediately.</p>
                <hr>
                <p><em>This is an automated alert from the YOLO Unauthorized Person Detection System.</em></p>
            </body>
            </html>
            """
            
            msg.attach(MimeText(body, 'html'))
            
            # Attach frame if provided
            if frame is not None:
                try:
                    # Save frame temporarily
                    temp_frame_path = "temp_alert_frame.jpg"
                    cv2.imwrite(temp_frame_path, frame)
                    
                    with open(temp_frame_path, 'rb') as f:
                        img_data = f.read()
                    
                    image = MimeImage(img_data)
                    image.add_header('Content-Disposition', 'attachment', filename='detected_person.jpg')
                    msg.attach(image)
                    
                    # Clean up temp file
                    os.remove(temp_frame_path)
                
                except Exception as e:
                    print(f"⚠️ Could not attach frame to email: {e}")
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()
            
            print(f"✅ Email alert sent to {self.recipient_email}")
            return True
        
        except Exception as e:
            print(f"❌ Error sending email alert: {e}")
            return False
    
    def send_sms_alert(self, person_name: str, confidence: float) -> bool:
        """
        Send SMS alert using Twilio
        
        Args:
            person_name: Name of detected person
            confidence: Detection confidence
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_sms or not self.twilio_account_sid:
            return False
        
        try:
            from twilio.rest import Client
            
            # Create Twilio client
            client = Client(self.twilio_account_sid, self.twilio_auth_token)
            
            # Create message
            message_body = f"🚨 UNAUTHORIZED PERSON DETECTED\n\nPerson: {person_name}\nConfidence: {confidence:.2f}\nTime: {create_timestamp()}\n\nCheck security system immediately!"
            
            # Send SMS
            message = client.messages.create(
                body=message_body,
                from_=self.twilio_from_number,
                to=self.twilio_to_number
            )
            
            print(f"✅ SMS alert sent to {self.twilio_to_number}")
            print(f"📱 Message SID: {message.sid}")
            return True
        
        except ImportError:
            print("❌ Twilio not installed. Install with: pip install twilio")
            return False
        except Exception as e:
            print(f"❌ Error sending SMS alert: {e}")
            return False
    
    def trigger_alert(self, person_name: str, confidence: float, 
                     frame: Optional[np.ndarray] = None) -> bool:
        """
        Trigger all enabled alerts
        
        Args:
            person_name: Name of detected person
            confidence: Detection confidence
            frame: Optional frame image
            
        Returns:
            True if at least one alert was successful
        """
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False
        
        print(f"🚨 Triggering alerts for: {person_name} (confidence: {confidence:.2f})")
        
        success_count = 0
        
        # Play sound alert
        if self.play_sound_alert():
            success_count += 1
        
        # Send email alert
        if self.send_email_alert(person_name, confidence, frame):
            success_count += 1
        
        # Send SMS alert
        if self.send_sms_alert(person_name, confidence):
            success_count += 1
        
        # Update alert tracking
        self.last_alert_time = current_time
        self.alert_count += 1
        
        if success_count > 0:
            print(f"✅ {success_count} alert(s) triggered successfully")
            return True
        else:
            print("❌ No alerts were triggered successfully")
            return False
    
    def test_alerts(self):
        """
        Test all alert systems
        """
        print("🧪 Testing alert systems...")
        
        # Test sound alert
        print("🔊 Testing sound alert...")
        if self.play_sound_alert():
            print("✅ Sound alert test successful")
        else:
            print("❌ Sound alert test failed")
        
        # Test email alert
        if self.enable_email:
            print("📧 Testing email alert...")
            if self.send_email_alert("Test Person", 0.95):
                print("✅ Email alert test successful")
            else:
                print("❌ Email alert test failed")
        
        # Test SMS alert
        if self.enable_sms:
            print("📱 Testing SMS alert...")
            if self.send_sms_alert("Test Person", 0.95):
                print("✅ SMS alert test successful")
            else:
                print("❌ SMS alert test failed")
        
        print("🧪 Alert system testing completed")
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """
        Get alert system statistics
        
        Returns:
            Dictionary with alert statistics
        """
        return {
            'total_alerts': self.alert_count,
            'last_alert_time': self.last_alert_time,
            'cooldown_period': self.alert_cooldown,
            'sound_enabled': self.enable_sound,
            'email_enabled': self.enable_email,
            'sms_enabled': self.enable_sms
        }


def main():
    """
    Main function for testing alert system
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Test alert system')
    parser.add_argument('--test', action='store_true', help='Test all alert systems')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        alert_system = AlertSystem(args.config)
        
        if args.test:
            alert_system.test_alerts()
        else:
            # Test with sample data
            print("🎯 Testing alert system with sample data...")
            success = alert_system.trigger_alert("Unknown Person", 0.85)
            if success:
                print("✅ Alert system test successful")
            else:
                print("❌ Alert system test failed")
    
    except Exception as e:
        print(f"❌ Error in alert system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
