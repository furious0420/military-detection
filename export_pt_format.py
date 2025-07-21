#!/usr/bin/env python3
"""
Export YOLO model and metrics to .pt format for cross-device compatibility.
"""

import pickle
import torch
import json
import shutil
from pathlib import Path
from datetime import datetime

def load_model_from_pkl(pkl_file='yolo_model_and_metrics.pkl'):
    """Load the saved model from pickle file."""
    print("📦 Loading saved YOLO model and metrics...")
    try:
        with open(pkl_file, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        print(f"✅ Model loaded successfully!")
        print(f"🏷️  Classes: {list(model.names.values())}")
        print(f"📊 Performance: mAP@0.5 = {model_data['metrics']['best_mAP50']:.3f}")
        return model_data
    except FileNotFoundError:
        print(f"❌ Error: Model file '{pkl_file}' not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def export_to_pt_format(model_data, output_dir="portable_model"):
    """
    Export model to .pt format with all necessary files for another device.
    """
    print(f"\n{'='*60}")
    print("🚀 EXPORTING TO .PT FORMAT FOR CROSS-DEVICE USE")
    print(f"{'='*60}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    model = model_data['model']
    model_info = model_data['model_info']
    metrics = model_data['metrics']
    
    # 1. Copy the original .pt weights file
    print("\n1️⃣ Copying PyTorch model weights...")
    try:
        # Find the best.pt file
        original_weights = Path("unified_detection/multi_class_v15/weights/best.pt")
        if original_weights.exists():
            pt_model_path = output_path / "yolo_multiclass_model.pt"
            shutil.copy2(str(original_weights), str(pt_model_path))
            print(f"   ✅ Model weights: {pt_model_path}")
            print(f"   📏 File size: {pt_model_path.stat().st_size / (1024*1024):.2f} MB")
        else:
            print(f"   ❌ Original weights not found at {original_weights}")
            return None
    except Exception as e:
        print(f"   ❌ Error copying weights: {e}")
        return None
    
    # 2. Create model configuration file
    print("\n2️⃣ Creating model configuration...")
    try:
        config = {
            "model_info": {
                "model_type": "YOLOv8",
                "num_classes": len(model.names),
                "class_names": list(model.names.values()),
                "class_mapping": dict(model.names),
                "input_size": [640, 640],
                "model_architecture": "yolov8n"
            },
            "training_metrics": {
                "final_epoch": metrics['final_epoch'],
                "best_mAP50": float(metrics['best_mAP50']),
                "best_mAP50_95": float(metrics['best_mAP50_95']),
                "final_precision": float(metrics['final_precision']),
                "final_recall": float(metrics['final_recall']),
                "final_mAP50": float(metrics['final_mAP50']),
                "final_mAP50_95": float(metrics['final_mAP50_95']),
                "training_time_hours": float(metrics['training_time_total'] / 3600)
            },
            "usage_info": {
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "input_format": "RGB",
                "input_normalization": "0-1",
                "export_timestamp": datetime.now().isoformat()
            }
        }
        
        config_path = output_path / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   ✅ Configuration: {config_path}")
        
    except Exception as e:
        print(f"   ❌ Error creating config: {e}")
        return None
    
    # 3. Create class names file (for easy loading)
    print("\n3️⃣ Creating class names file...")
    try:
        classes_path = output_path / "classes.txt"
        with open(classes_path, 'w') as f:
            for class_name in model.names.values():
                f.write(f"{class_name}\n")
        
        print(f"   ✅ Class names: {classes_path}")
        
    except Exception as e:
        print(f"   ❌ Error creating classes file: {e}")
    
    # 4. Export training metrics to CSV
    print("\n4️⃣ Exporting training metrics...")
    try:
        if 'all_metrics' in metrics:
            import pandas as pd
            metrics_df = pd.DataFrame(metrics['all_metrics'])
            metrics_path = output_path / "training_history.csv"
            metrics_df.to_csv(metrics_path, index=False)
            print(f"   ✅ Training history: {metrics_path}")
            print(f"   📊 {len(metrics_df)} epochs of training data")
        else:
            print(f"   ⚠️  Full training history not available")
    except Exception as e:
        print(f"   ❌ Error exporting metrics: {e}")
    
    # 5. Create usage example script
    print("\n5️⃣ Creating usage example...")
    try:
        example_script = f'''#!/usr/bin/env python3
"""
Example script to use the exported YOLO model on another device.
"""

from ultralytics import YOLO
import cv2
import json

def load_model():
    """Load the exported YOLO model."""
    # Load model
    model = YOLO('yolo_multiclass_model.pt')
    
    # Load configuration
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    
    print("✅ Model loaded successfully!")
    print(f"🏷️  Classes: {{config['model_info']['class_names']}}")
    print(f"📊 Best mAP@0.5: {{config['training_metrics']['best_mAP50']:.3f}}")
    
    return model, config

def detect_objects(model, image_path, conf_threshold=0.25):
    """Run detection on an image."""
    print(f"🔍 Running detection on: {{image_path}}")
    
    # Run inference
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=0.45,
        save=True,
        show_labels=True,
        show_conf=True
    )
    
    # Process results
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            classes = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            
            print(f"✅ Found {{len(boxes)}} objects:")
            
            class_counts = {{}}
            for cls_id, conf in zip(classes, confidences):
                class_name = model.names[int(cls_id)]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                print(f"  - {{class_name}}: {{conf:.3f}} ({{conf*100:.1f}}%)")
            
            print("\\n📊 Summary:")
            for class_name, count in class_counts.items():
                print(f"  {{class_name}}: {{count}} detected")
                
            return results
        else:
            print("❌ No objects detected")
            return None
    else:
        print("❌ No results returned")
        return None

def main():
    """Main function - example usage."""
    try:
        # Load model
        model, config = load_model()
        
        # Example detection (replace with your image path)
        image_path = "test_image.jpg"  # Change this to your image
        
        # Run detection
        results = detect_objects(model, image_path, conf_threshold=0.25)
        
        if results:
            print("\\n🎉 Detection completed! Check the 'runs' folder for output.")
        
    except Exception as e:
        print(f"❌ Error: {{e}}")

if __name__ == "__main__":
    main()
'''
        
        example_path = output_path / "example_usage.py"
        with open(example_path, 'w') as f:
            f.write(example_script)
        
        print(f"   ✅ Usage example: {example_path}")
        
    except Exception as e:
        print(f"   ❌ Error creating example: {e}")
    
    # 6. Create README file
    print("\n6️⃣ Creating README documentation...")
    try:
        readme_content = f"""# YOLO Multi-Class Detection Model (.pt format)

## 📋 Model Information
- **Model Type**: YOLOv8n
- **Classes**: {len(model.names)} ({', '.join(model.names.values())})
- **Input Size**: 640×640 pixels
- **Best mAP@0.5**: {metrics['best_mAP50']:.3f}
- **Best mAP@0.5:0.95**: {metrics['best_mAP50_95']:.3f}
- **Training Time**: {metrics['training_time_total']/3600:.1f} hours

## 📁 Files Included
- `yolo_multiclass_model.pt` - Main model weights (PyTorch format)
- `model_config.json` - Model configuration and metadata
- `classes.txt` - List of class names
- `training_history.csv` - Complete training metrics
- `example_usage.py` - Example Python script
- `README.md` - This documentation

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install ultralytics opencv-python
```

### 2. Load and Use Model
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolo_multiclass_model.pt')

# Run detection
results = model.predict('your_image.jpg', conf=0.25)

# Process results
for result in results:
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            class_name = model.names[class_id]
            print(f"{{class_name}}: {{confidence:.3f}}")
```

### 3. Run Example Script
```bash
python example_usage.py
```

## 🎯 Class Mapping
{chr(10).join([f"- {idx}: {name}" for idx, name in model.names.items()])}

## 📊 Performance Metrics
- **Final Precision**: {metrics['final_precision']:.3f}
- **Final Recall**: {metrics['final_recall']:.3f}
- **Final mAP@0.5**: {metrics['final_mAP50']:.3f}
- **Final mAP@0.5:0.95**: {metrics['final_mAP50_95']:.3f}

## 🔧 Recommended Settings
- **Confidence Threshold**: 0.25 (adjust based on your needs)
- **IoU Threshold**: 0.45
- **Input Format**: RGB images
- **Supported Formats**: JPG, PNG, BMP, TIFF

## 💡 Usage Tips
1. **Lower confidence** (0.1-0.2) for more detections
2. **Higher confidence** (0.4-0.6) for fewer false positives
3. **Thermal images**: May need lower confidence thresholds
4. **Multiple objects**: Model supports simultaneous multi-class detection

## 🖥️ System Requirements
- **Python**: 3.8+
- **PyTorch**: 1.9+
- **RAM**: 4GB+ recommended
- **GPU**: Optional (CUDA-compatible for faster inference)

## 📞 Support
This model was trained for multi-class object detection including:
humans, animals, vehicles, drones, and weapons.

For best results, ensure input images are well-lit and objects are clearly visible.
"""
        
        readme_path = output_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"   ✅ Documentation: {readme_path}")
        
    except Exception as e:
        print(f"   ❌ Error creating README: {e}")
    
    return output_path

def main():
    """Main export function."""
    
    try:
        # Load model data
        model_data = load_model_from_pkl()
        if model_data is None:
            return
        
        # Export to .pt format
        output_dir = export_to_pt_format(model_data)
        
        if output_dir:
            # Summary
            print(f"\n{'='*60}")
            print("🎉 .PT FORMAT EXPORT COMPLETED!")
            print(f"{'='*60}")
            print(f"📁 Output Directory: {output_dir}")
            
            # List all files
            print(f"\n📦 EXPORTED FILES:")
            for file_path in sorted(output_dir.glob('*')):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024*1024)
                    print(f"   📄 {file_path.name} ({size_mb:.2f} MB)")
            
            print(f"\n🚀 READY FOR DEPLOYMENT:")
            print(f"   1. Copy entire '{output_dir}' folder to target device")
            print(f"   2. Install: pip install ultralytics opencv-python")
            print(f"   3. Run: python example_usage.py")
            
            print(f"\n💡 USAGE ON ANOTHER DEVICE:")
            print(f"   from ultralytics import YOLO")
            print(f"   model = YOLO('yolo_multiclass_model.pt')")
            print(f"   results = model.predict('image.jpg')")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    main()
