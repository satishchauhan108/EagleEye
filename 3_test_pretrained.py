from ultralytics import YOLO
import cv2
import os

def test_pretrained(video_path, output_path='outputs/pretrained_result.mp4'):
    model = YOLO('yolov8n-cls.pt')
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs('outputs', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    predictions = {'Normal': 0, 'Shoplifting': 0}
    
    print(f"Processing video: {video_path}")
    print("Note: Pretrained model is not trained on shoplifting data!")
    print("It will show generic classifications\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, verbose=False)
        frame_count += 1
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
    
    cap.release()
    out.release()
    
    print(f"\nOutput saved to: {output_path}")
    return predictions

if __name__ == "__main__":
    video_path = 'data/Shoplifting/video1.mp4'
    
    if os.path.exists(video_path):
        test_pretrained(video_path)
    else:
        print(f"Video not found: {video_path}")
        print("Please update the video_path variable")
