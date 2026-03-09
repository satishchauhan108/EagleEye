import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    step = max(1, total_frames // max_frames)
    
    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % step == 0:
            frame_path = os.path.join(output_dir, f'frame_{saved_count:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    return saved_count

def prepare_dataset():
    base_dir = 'data'
    output_dir = 'dataset'
    
    for split in ['train', 'val']:
        for class_name in ['Normal', 'Shoplifting']:
            os.makedirs(f'{output_dir}/{split}/{class_name}', exist_ok=True)
    
    for class_name in ['Normal', 'Shoplifting']:
        video_dir = f'{base_dir}/{class_name}'
        
        if not os.path.exists(video_dir):
            print(f"Warning: {video_dir} not found!")
            continue
            
        videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
        print(f"\nProcessing {len(videos)} videos from {class_name}...")
        
        split_idx = int(len(videos) * 0.8)
        
        for i, video_file in enumerate(videos):
            video_path = os.path.join(video_dir, video_file)
            split = 'train' if i < split_idx else 'val'
            
            video_name = Path(video_file).stem
            output_path = f'{output_dir}/{split}/{class_name}/{video_name}'
            
            frames = extract_frames(video_path, output_path, max_frames=30)
            print(f"  {video_file}: {frames} frames → {split}/{class_name}/")
    
    print("\nDataset preparation complete!")
    print(f"Train: dataset/train/")
    print(f"Val: dataset/val/")

if __name__ == "__main__":
    prepare_dataset()
