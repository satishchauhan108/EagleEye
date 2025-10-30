"""
YOLO Training Script for Person Detection
This script trains or fine-tunes YOLOv8 model for person detection
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_config, ensure_directory_exists


class YOLOTrainer:
    """
    YOLO model trainer for person detection
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize YOLO trainer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.model_config = self.config.get('model', {})
        self.paths_config = self.config.get('paths', {})
        self.performance_config = self.config.get('performance', {})
        
        # Model settings
        self.model_path = self.model_config.get('yolo_model_path', 'yolov8n.pt')
        self.use_gpu = self.performance_config.get('use_gpu', True)
        
        # Paths
        self.models_dir = self.paths_config.get('models_dir', 'models')
        ensure_directory_exists(self.models_dir)
        
        # Training settings
        self.epochs = 100
        self.batch_size = 16
        self.img_size = 640
        self.device = 'cuda' if torch.cuda.is_available() and self.use_gpu else 'cpu'
        
        print("🤖 YOLO Trainer Initialized")
        print(f"📁 Models directory: {self.models_dir}")
        print(f"🔧 Device: {self.device}")
        print(f"🎯 Model: {self.model_path}")
    
    def check_gpu_availability(self) -> bool:
        """
        Check if GPU is available and configured
        
        Returns:
            True if GPU is available and usable
        """
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available. Using CPU for training.")
            return False
        
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        
        print(f"✅ GPU available: {gpu_name}")
        print(f"🔢 GPU count: {gpu_count}")
        
        # Test GPU memory
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("✅ GPU memory test successful")
            return True
        except Exception as e:
            print(f"❌ GPU memory test failed: {e}")
            return False
    
    def download_pretrained_model(self) -> str:
        """
        Download pretrained YOLOv8 model if not exists
        
        Returns:
            Path to the model file
        """
        model_path = os.path.join(self.models_dir, os.path.basename(self.model_path))
        
        if os.path.exists(model_path):
            print(f"✅ Pretrained model already exists: {model_path}")
            return model_path
        
        print(f"📥 Downloading pretrained model: {self.model_path}")
        
        try:
            # Download model using ultralytics
            model = YOLO(self.model_path)
            model.save(model_path)
            print(f"✅ Model downloaded successfully: {model_path}")
            return model_path
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return self.model_path
    
    def create_dataset_config(self, dataset_path: str) -> str:
        """
        Create YOLO dataset configuration file
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Path to dataset configuration file
        """
        dataset_config = {
            'path': dataset_path,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,  # Number of classes (person only)
            'names': ['person']  # Class names
        }
        
        config_path = os.path.join(dataset_path, 'dataset.yaml')
        
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✅ Dataset configuration created: {config_path}")
        return config_path
    
    def prepare_custom_dataset(self, dataset_path: str) -> bool:
        """
        Prepare custom dataset for training (if needed)
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            True if dataset is prepared successfully
        """
        print("📋 Preparing custom dataset...")
        print("ℹ️ Note: This is a placeholder for custom dataset preparation.")
        print("ℹ️ For person detection, the pretrained YOLOv8 model is usually sufficient.")
        print("ℹ️ Custom training is only needed if you have specific requirements.")
        
        # Create dataset structure
        ensure_directory_exists(os.path.join(dataset_path, 'images', 'train'))
        ensure_directory_exists(os.path.join(dataset_path, 'images', 'val'))
        ensure_directory_exists(os.path.join(dataset_path, 'images', 'test'))
        ensure_directory_exists(os.path.join(dataset_path, 'labels', 'train'))
        ensure_directory_exists(os.path.join(dataset_path, 'labels', 'val'))
        ensure_directory_exists(os.path.join(dataset_path, 'labels', 'test'))
        
        # Create dataset configuration
        self.create_dataset_config(dataset_path)
        
        print("✅ Dataset structure created")
        print("📝 Please add your training images and labels to the appropriate directories")
        print("📝 Images should be in JPG/PNG format")
        print("📝 Labels should be in YOLO format (.txt files)")
        
        return True
    
    def train_model(self, dataset_path: str = None, custom_training: bool = False) -> str:
        """
        Train YOLO model
        
        Args:
            dataset_path: Path to custom dataset (if using custom training)
            custom_training: Whether to use custom dataset
            
        Returns:
            Path to trained model
        """
        print("🚀 Starting YOLO training...")
        
        # Download pretrained model
        model_path = self.download_pretrained_model()
        
        # Load model
        model = YOLO(model_path)
        
        if custom_training and dataset_path:
            # Custom training with dataset
            print("🎯 Custom training mode")
            
            if not os.path.exists(dataset_path):
                print(f"❌ Dataset path not found: {dataset_path}")
                return None
            
            # Prepare dataset
            self.prepare_custom_dataset(dataset_path)
            dataset_config = os.path.join(dataset_path, 'dataset.yaml')
            
            # Train model
            results = model.train(
                data=dataset_config,
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=self.img_size,
                device=self.device,
                project=self.models_dir,
                name='yolo_person_detection',
                save=True,
                save_period=10,
                val=True,
                plots=True,
                verbose=True
            )
        else:
            # Use pretrained model as-is (recommended for person detection)
            print("🎯 Using pretrained model (recommended for person detection)")
            print("ℹ️ YOLOv8n is already trained on COCO dataset which includes person class")
            print("ℹ️ No additional training needed for basic person detection")
            
            # Save model to our models directory
            trained_model_path = os.path.join(self.models_dir, 'yolo_person_detection.pt')
            model.save(trained_model_path)
            print(f"✅ Model saved: {trained_model_path}")
            return trained_model_path
        
        # Get trained model path
        trained_model_path = os.path.join(self.models_dir, 'yolo_person_detection', 'weights', 'best.pt')
        
        if os.path.exists(trained_model_path):
            print(f"✅ Training completed successfully")
            print(f"💾 Trained model saved: {trained_model_path}")
            return trained_model_path
        else:
            print("❌ Training failed or model not found")
            return None
    
    def validate_model(self, model_path: str) -> bool:
        """
        Validate trained model
        
        Args:
            model_path: Path to trained model
            
        Returns:
            True if model is valid
        """
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        try:
            # Load and test model
            model = YOLO(model_path)
            
            # Test with dummy image
            import numpy as np
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            results = model(dummy_image)
            
            print("✅ Model validation successful")
            print(f"📊 Model classes: {model.names}")
            print(f"🔢 Number of classes: {len(model.names)}")
            
            return True
        
        except Exception as e:
            print(f"❌ Model validation failed: {e}")
            return False
    
    def export_model(self, model_path: str, export_format: str = 'onnx') -> str:
        """
        Export model to different formats
        
        Args:
            model_path: Path to trained model
            export_format: Export format ('onnx', 'torchscript', 'tflite')
            
        Returns:
            Path to exported model
        """
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return None
        
        try:
            model = YOLO(model_path)
            
            # Export model
            exported_path = model.export(
                format=export_format,
                imgsz=self.img_size,
                optimize=True,
                half=True if self.device == 'cuda' else False
            )
            
            print(f"✅ Model exported successfully: {exported_path}")
            return exported_path
        
        except Exception as e:
            print(f"❌ Model export failed: {e}")
            return None
    
    def benchmark_model(self, model_path: str) -> dict:
        """
        Benchmark model performance
        
        Args:
            model_path: Path to model
            
        Returns:
            Benchmark results dictionary
        """
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return {}
        
        try:
            model = YOLO(model_path)
            
            # Benchmark model
            results = model.benchmark(
                imgsz=self.img_size,
                device=self.device,
                half=True if self.device == 'cuda' else False
            )
            
            print("📊 Model Benchmark Results:")
            print(f"   Device: {self.device}")
            print(f"   Image size: {self.img_size}")
            print(f"   Results: {results}")
            
            return results
        
        except Exception as e:
            print(f"❌ Model benchmarking failed: {e}")
            return {}


def main():
    """
    Main function for YOLO training script
    """
    parser = argparse.ArgumentParser(description='Train YOLO model for person detection')
    parser.add_argument('--dataset', type=str, help='Path to custom dataset')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--custom', action='store_true', help='Use custom dataset for training')
    parser.add_argument('--validate', type=str, help='Validate existing model')
    parser.add_argument('--export', type=str, help='Export model (onnx, torchscript, tflite)')
    parser.add_argument('--benchmark', type=str, help='Benchmark model performance')
    
    args = parser.parse_args()
    
    try:
        trainer = YOLOTrainer(args.config)
        
        if args.validate:
            # Validate existing model
            success = trainer.validate_model(args.validate)
            sys.exit(0 if success else 1)
        
        elif args.export:
            # Export model
            exported_path = trainer.export_model(args.validate or 'models/yolo_person_detection.pt', args.export)
            sys.exit(0 if exported_path else 1)
        
        elif args.benchmark:
            # Benchmark model
            results = trainer.benchmark_model(args.benchmark)
            sys.exit(0 if results else 1)
        
        else:
            # Train model
            print("🎯 Training YOLO model for person detection...")
            
            if args.custom and args.dataset:
                # Custom training
                model_path = trainer.train_model(args.dataset, custom_training=True)
            else:
                # Use pretrained model (recommended)
                model_path = trainer.train_model()
            
            if model_path:
                print(f"✅ Training completed successfully: {model_path}")
                
                # Validate trained model
                if trainer.validate_model(model_path):
                    print("✅ Model validation successful")
                else:
                    print("⚠️ Model validation failed")
                
                sys.exit(0)
            else:
                print("❌ Training failed")
                sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error in YOLO training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
