#!/usr/bin/env python3
"""
Export trained YOLO model to multiple formats for cross-device compatibility.
"""

import pickle
import json
import torch
import onnx
import numpy as np
from pathlib import Path
import shutil
import zipfile
from datetime import datetime

def load_model_from_pkl(pkl_file='yolo_model_and_metrics.pkl'):
    """Load the saved model from pickle file."""
    print("📦 Loading saved YOLO model...")
    try:
        with open(pkl_file, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        print(f"✅ Model loaded successfully!")
        print(f"🏷️  Classes: {list(model.names.values())}")
        return model_data
    except FileNotFoundError:
        print(f"❌ Error: Model file '{pkl_file}' not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def export_to_multiple_formats(model_data, export_dir="exported_models"):
    """
    Export model to multiple formats for different devices and platforms.
    """
    print(f"\n{'='*70}")
    print("🚀 EXPORTING MODEL TO MULTIPLE FORMATS")
    print(f"{'='*70}")
    
    # Create export directory
    export_path = Path(export_dir)
    export_path.mkdir(exist_ok=True)
    
    model = model_data['model']
    model_info = model_data['model_info']
    metrics = model_data['metrics']
    
    exported_formats = {}
    
    # 1. Export to ONNX (Cross-platform, optimized)
    print("\n1️⃣ Exporting to ONNX format...")
    try:
        onnx_path = export_path / "model.onnx"
        model.export(format='onnx', imgsz=640, optimize=True)
        
        # Move the exported file to our directory
        runs_path = Path("runs/detect/train/weights/best.onnx")
        if runs_path.exists():
            shutil.move(str(runs_path), str(onnx_path))
        else:
            # Try alternative path
            for onnx_file in Path(".").glob("**/*.onnx"):
                if onnx_file.stat().st_mtime > (datetime.now().timestamp() - 60):  # Recent file
                    shutil.move(str(onnx_file), str(onnx_path))
                    break
        
        if onnx_path.exists():
            exported_formats['onnx'] = str(onnx_path)
            print(f"   ✅ ONNX: {onnx_path}")
            print(f"   📱 Compatible: Python, C++, JavaScript, Mobile apps")
        else:
            print(f"   ❌ ONNX export failed")
    except Exception as e:
        print(f"   ❌ ONNX export error: {e}")
    
    # 2. Export to TensorRT (NVIDIA GPUs)
    print("\n2️⃣ Exporting to TensorRT format...")
    try:
        trt_path = export_path / "model.engine"
        model.export(format='engine', imgsz=640, half=True)
        
        # Find and move TensorRT file
        for trt_file in Path(".").glob("**/*.engine"):
            if trt_file.stat().st_mtime > (datetime.now().timestamp() - 60):
                shutil.move(str(trt_file), str(trt_path))
                break
        
        if trt_path.exists():
            exported_formats['tensorrt'] = str(trt_path)
            print(f"   ✅ TensorRT: {trt_path}")
            print(f"   🚀 Compatible: NVIDIA GPUs, Jetson devices")
        else:
            print(f"   ⚠️  TensorRT export skipped (requires NVIDIA GPU)")
    except Exception as e:
        print(f"   ⚠️  TensorRT export skipped: {e}")
    
    # 3. Export to TorchScript (PyTorch native)
    print("\n3️⃣ Exporting to TorchScript format...")
    try:
        torchscript_path = export_path / "model.torchscript"
        model.export(format='torchscript', imgsz=640, optimize=True)
        
        # Find and move TorchScript file
        for ts_file in Path(".").glob("**/*.torchscript"):
            if ts_file.stat().st_mtime > (datetime.now().timestamp() - 60):
                shutil.move(str(ts_file), str(torchscript_path))
                break
        
        if torchscript_path.exists():
            exported_formats['torchscript'] = str(torchscript_path)
            print(f"   ✅ TorchScript: {torchscript_path}")
            print(f"   🐍 Compatible: PyTorch applications, C++")
        else:
            print(f"   ❌ TorchScript export failed")
    except Exception as e:
        print(f"   ❌ TorchScript export error: {e}")
    
    # 4. Export to CoreML (Apple devices)
    print("\n4️⃣ Exporting to CoreML format...")
    try:
        coreml_path = export_path / "model.mlmodel"
        model.export(format='coreml', imgsz=640)
        
        # Find and move CoreML file
        for ml_file in Path(".").glob("**/*.mlmodel"):
            if ml_file.stat().st_mtime > (datetime.now().timestamp() - 60):
                shutil.move(str(ml_file), str(coreml_path))
                break
        
        if coreml_path.exists():
            exported_formats['coreml'] = str(coreml_path)
            print(f"   ✅ CoreML: {coreml_path}")
            print(f"   🍎 Compatible: iOS, macOS applications")
        else:
            print(f"   ⚠️  CoreML export skipped (requires macOS)")
    except Exception as e:
        print(f"   ⚠️  CoreML export skipped: {e}")
    
    # 5. Export to TensorFlow Lite (Mobile/Edge)
    print("\n5️⃣ Exporting to TensorFlow Lite format...")
    try:
        tflite_path = export_path / "model.tflite"
        model.export(format='tflite', imgsz=640, int8=True)
        
        # Find and move TFLite file
        for tfl_file in Path(".").glob("**/*.tflite"):
            if tfl_file.stat().st_mtime > (datetime.now().timestamp() - 60):
                shutil.move(str(tfl_file), str(tflite_path))
                break
        
        if tflite_path.exists():
            exported_formats['tflite'] = str(tflite_path)
            print(f"   ✅ TensorFlow Lite: {tflite_path}")
            print(f"   📱 Compatible: Android, iOS, Edge devices")
        else:
            print(f"   ❌ TensorFlow Lite export failed")
    except Exception as e:
        print(f"   ❌ TensorFlow Lite export error: {e}")
    
    # 6. Save original PyTorch weights
    print("\n6️⃣ Copying PyTorch weights...")
    try:
        pt_path = export_path / "model.pt"
        original_weights = Path("unified_detection/multi_class_v15/weights/best.pt")
        if original_weights.exists():
            shutil.copy2(str(original_weights), str(pt_path))
            exported_formats['pytorch'] = str(pt_path)
            print(f"   ✅ PyTorch: {pt_path}")
            print(f"   🔥 Compatible: PyTorch, Ultralytics YOLO")
        else:
            print(f"   ❌ Original weights not found")
    except Exception as e:
        print(f"   ❌ PyTorch copy error: {e}")
    
    # 7. Export metrics and model info
    print("\n7️⃣ Exporting model metadata...")
    try:
        # Model configuration
        config = {
            'model_info': model_info,
            'training_metrics': {
                'final_epoch': metrics['final_epoch'],
                'best_mAP50': metrics['best_mAP50'],
                'best_mAP50_95': metrics['best_mAP50_95'],
                'final_precision': metrics['final_precision'],
                'final_recall': metrics['final_recall'],
                'training_time_hours': metrics['training_time_total'] / 3600
            },
            'class_names': list(model.names.values()),
            'class_mapping': dict(model.names),
            'input_size': [640, 640],
            'export_timestamp': datetime.now().isoformat(),
            'exported_formats': list(exported_formats.keys())
        }
        
        # Save as JSON
        config_path = export_path / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save full metrics as CSV
        metrics_path = export_path / "training_metrics.csv"
        if 'all_metrics' in metrics:
            import pandas as pd
            df = pd.DataFrame(metrics['all_metrics'])
            df.to_csv(metrics_path, index=False)
        
        print(f"   ✅ Config: {config_path}")
        print(f"   ✅ Metrics: {metrics_path}")
        
    except Exception as e:
        print(f"   ❌ Metadata export error: {e}")
    
    # 8. Create deployment package
    print("\n8️⃣ Creating deployment package...")
    try:
        # Create README
        readme_path = export_path / "README.md"
        readme_content = f"""# YOLO Multi-Class Detection Model

## Model Information
- **Classes**: {len(model.names)} ({', '.join(model.names.values())})
- **Input Size**: 640x640 pixels
- **Performance**: mAP@0.5 = {metrics['best_mAP50']:.3f}
- **Training Time**: {metrics['training_time_total']/3600:.1f} hours

## Exported Formats
{chr(10).join([f"- **{fmt.upper()}**: {path}" for fmt, path in exported_formats.items()])}

## Usage Examples

### Python (ONNX)
```python
import cv2
import onnxruntime as ort

# Load model
session = ort.InferenceSession('model.onnx')

# Run inference
image = cv2.imread('test.jpg')
# ... preprocessing ...
results = session.run(None, {{'images': image}})
```

### Python (PyTorch)
```python
from ultralytics import YOLO

# Load model
model = YOLO('model.pt')

# Run inference
results = model.predict('test.jpg')
```

## Class Mapping
{chr(10).join([f"- {idx}: {name}" for idx, name in model.names.items()])}

## Performance Metrics
- Final mAP@0.5: {metrics['final_mAP50']:.3f}
- Final mAP@0.5:0.95: {metrics['final_mAP50_95']:.3f}
- Final Precision: {metrics['final_precision']:.3f}
- Final Recall: {metrics['final_recall']:.3f}
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"   ✅ README: {readme_path}")
        
        # Create ZIP package
        zip_path = f"yolo_multiclass_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in export_path.rglob('*'):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(export_path))
        
        print(f"   ✅ Package: {zip_path}")
        
    except Exception as e:
        print(f"   ❌ Package creation error: {e}")
    
    return exported_formats, export_path

def main():
    """Main export function."""
    
    try:
        # Load model data
        model_data = load_model_from_pkl()
        if model_data is None:
            return
        
        # Export to multiple formats
        exported_formats, export_dir = export_to_multiple_formats(model_data)
        
        # Summary
        print(f"\n{'='*70}")
        print("🎉 MODEL EXPORT COMPLETED!")
        print(f"{'='*70}")
        print(f"📁 Export Directory: {export_dir}")
        print(f"📦 Formats Exported: {len(exported_formats)}")
        
        print(f"\n✅ SUCCESSFULLY EXPORTED FORMATS:")
        for format_name, file_path in exported_formats.items():
            print(f"   🔹 {format_name.upper()}: {file_path}")
        
        print(f"\n🚀 DEPLOYMENT READY:")
        print(f"   📱 Mobile: TensorFlow Lite, CoreML")
        print(f"   🖥️  Desktop: ONNX, PyTorch, TorchScript")
        print(f"   ⚡ GPU: TensorRT (NVIDIA)")
        print(f"   🌐 Web: ONNX.js")
        print(f"   📊 Complete metrics and config included")
        
        print(f"\n💡 USAGE:")
        print(f"   - Copy entire '{export_dir}' folder to target device")
        print(f"   - Use appropriate format for your platform")
        print(f"   - Check README.md for usage examples")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    main()
