from ultralytics import YOLO
import cv2
import os
import numpy as np

def test_custom_model(video_path, model_path='best.pt'):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train the model first using 2_train_model.py")
        return None
    
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = 'outputs/custom_result.mp4'
    os.makedirs('outputs', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    predictions = []
    
    print(f"Processing video with custom model: {video_path}\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, verbose=False)
        
        probs = results[0].probs
        top_class = probs.top1
        top_conf = probs.top1conf.item()
        class_name = model.names[top_class]
        
        predictions.append({
            'class': class_name,
            'confidence': top_conf
        })
        
        color = (0, 255, 0) if class_name == 'Normal' else (0, 0, 255)
        cv2.putText(frame, f'{class_name}: {top_conf:.2f}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, color, 2)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
    
    cap.release()
    out.release()
    
    normal_count = sum(1 for p in predictions if p['class'] == 'Normal')
    shoplifting_count = len(predictions) - normal_count
    avg_conf = np.mean([p['confidence'] for p in predictions])
    
    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"{'='*50}")
    print(f"Total frames: {frame_count}")
    print(f"Normal: {normal_count} ({normal_count/frame_count*100:.1f}%)")
    print(f"Shoplifting: {shoplifting_count} ({shoplifting_count/frame_count*100:.1f}%)")
    print(f"Average confidence: {avg_conf:.2f}")
    print(f"\nOutput saved to: {output_path}")
    
    return predictions

if __name__ == "__main__":
    test_videos = [
        'data/Shoplifting/video1.mp4',
    ]
    
    for video in test_videos:
        if os.path.exists(video):
            print(f"\n{'='*60}")
            print(f"Testing: {video}")
            print(f"{'='*60}")
            test_custom_model(video)
        else:
            print(f"Video not found: {video}")
