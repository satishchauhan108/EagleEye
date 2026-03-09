from ultralytics import YOLO
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model_path, dataset_path='dataset/val'):
    model = YOLO(model_path)
    y_true = []
    y_pred = []
    classes = ['Normal', 'Shoplifting']
    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_dir):
            continue
        video_dirs = [d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))]
        for video_dir in video_dirs:
            video_path = os.path.join(class_dir, video_dir)
            frames = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
            for frame_file in frames:
                frame_path = os.path.join(video_path, frame_file)
                results = model(frame_path, verbose=False)
                pred_idx = results[0].probs.top1
                pred_class = 'Shoplifting' if pred_idx == 1 else 'Normal'
                y_true.append(class_name)
                y_pred.append(pred_class)
    return y_true, y_pred

def compare_models():
    print("Comparing Models on Validation Set")
    print("="*60)
    if not os.path.exists('best.pt'):
        print("Custom model not found!")
        return
    if not os.path.exists('dataset/val'):
        print("Validation dataset not found!")
        return
    models = {
        'Pretrained': 'yolov8n-cls.pt',
        'Custom': 'best.pt'
    }
    results_dict = {}
    for model_name, model_path in models.items():
        print(f"\nEvaluating {model_name} Model...")
        print("-"*60)
        y_true, y_pred = evaluate_model(model_path)
        if len(y_true) == 0:
            print(f"No data to evaluate for {model_name}")
            continue
        accuracy = np.mean([t == p for t, p in zip(y_true, y_pred)])
        print(f"\nAccuracy: {accuracy*100:.2f}%")
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=['Normal', 'Shoplifting'], target_names=['Normal', 'Shoplifting']))
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred, labels=['Normal', 'Shoplifting'])
        print(f"{'':>12} {'Normal':>12} {'Shoplifting':>12}")
        print(f"{'Normal':<12} {cm[0][0]:>12} {cm[0][1]:>12}")
        print(f"{'Shoplifting':<12} {cm[1][0]:>12} {cm[1][1]:>12}")
        results_dict[model_name] = accuracy
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    for model_name, accuracy in results_dict.items():
        print(f"{model_name:>12}: {accuracy*100:.2f}%")
    if len(results_dict) == 2:
        improvement = (results_dict['Custom'] - results_dict['Pretrained']) * 100
        print(f"\nImprovement: {improvement:+.2f}%")

if __name__ == "__main__":
    compare_models()
