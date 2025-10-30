"""
Simple camera test for YOLO Person Detection System
"""
import cv2
import sys

def test_camera():
    """Test camera connection"""
    print("🎥 Testing camera connection...")
    
    # Try different camera indices
    for camera_index in range(3):
        print(f"📷 Trying camera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                print(f"✅ Camera {camera_index} working!")
                print(f"📐 Frame size: {width}x{height}")
                
                # Show frame for 3 seconds
                cv2.imshow(f'Camera {camera_index} Test', frame)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()
                
                cap.release()
                return camera_index
            else:
                print(f"❌ Camera {camera_index} opened but cannot read frames")
        else:
            print(f"❌ Camera {camera_index} not available")
        
        cap.release()
    
    print("❌ No working camera found!")
    return None

if __name__ == "__main__":
    camera_index = test_camera()
    if camera_index is not None:
        print(f"\n🎯 Use camera index {camera_index} for detection")
        print("🚀 Run: python simple_detect.py")
    else:
        print("\n❌ Please check your camera connection")
        sys.exit(1)
