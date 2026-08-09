from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8n-cls.pt')
    
    print("Starting training...")
    print("This may take 10-30 minutes depending on your GPU/CPU\n")
    
    results = model.train(
        data='dataset',
        epochs=10,
        imgsz=224,
        batch=16,
        patience=10,
        save=True,
        project='runs/train',
        name='shoplifting_model',
        plots=True
    )
    
    print("\n" + "="*10)
    print("Training Complete!")
    print("="*10)
    print(f"Best model saved at: runs/train/shoplifting_model/weights/best.pt")
    print(f"Training plots saved at: runs/train/shoplifting_model/")
    
    import shutil
    shutil.copy(
        'runs/train/shoplifting_model/weights/best.pt',
        'models/best.pt'
    )
    print("\nBest model copied to: models/best.pt")

if __name__ == "__main__":
    train_model()
