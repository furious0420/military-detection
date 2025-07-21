# 🎯 YOLO Multi-Class Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance multi-class object detection system built with YOLOv8, capable of detecting **humans, animals, vehicles, drones, and weapons** in real-time with **97.99% mAP@0.5** accuracy.

## 🌟 **Key Features**

- 🎯 **Multi-Class Detection**: 5 classes (human, animal, vehicle, drone, weapon)
- ⚡ **Real-Time Performance**: 70ms inference time
- 🎨 **Smart Visualization**: Color-coded bounding boxes for each class
- 🧠 **Intelligent Classification**: Auto-corrects human/animal misclassification
- 📱 **Cross-Platform**: Works on any device with PyTorch
- 🔥 **High Accuracy**: 97.99% mAP@0.5, 76.64% mAP@0.5:0.95

## 📊 **Model Performance**

| Metric | Score |
|--------|-------|
| mAP@0.5 | **97.99%** |
| mAP@0.5:0.95 | **76.64%** |
| Precision | **96.01%** |
| Recall | **95.46%** |
| Inference Time | **~70ms** |
| Model Size | **11.78 MB** |

## 🚀 **Quick Start**

### **Installation**
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/yolo-multiclass-detection.git
cd yolo-multiclass-detection

# Install dependencies
pip install -r requirements.txt
```

### **Download Pre-trained Model**
```bash
# Download the trained model (replace with your actual download link)
wget https://github.com/YOUR_USERNAME/yolo-multiclass-detection/releases/download/v1.0/best.pt
```

### **Run Detection**
```python
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Run detection
results = model.predict('your_image.jpg', conf=0.25)

# Process results
for result in results:
    result.show()  # Display image with bounding boxes
```

### **Universal Detection Script**
```bash
# Detect objects in any image with colored bounding boxes
python universal_detector.py "path/to/your/image.jpg"
```

## 🎨 **Class Color Coding**

| Class | Color | Detection Focus |
|-------|-------|----------------|
| 🟢 **Human** | Green | People detection |
| 🔵 **Animal** | Blue | Animals (dogs, cats, wildlife) |
| 🔴 **Vehicle** | Red | Cars, trucks, buses |
| 🟡 **Drone** | Cyan | UAVs, quadcopters |
| 🟠 **Weapon** | Orange | Guns, knives, weapons |

## 📁 **Repository Structure**

```
yolo-multiclass-detection/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 LICENSE                      # MIT License
├── 🐍 universal_detector.py        # Main detection script
├── 🐍 train_yolov8.py             # Training script
├── 🐍 load_saved_model.py         # Model loading utilities
├── 📊 unified_dataset.yaml        # Dataset configuration
├── 📁 models/                      # Pre-trained models
│   ├── best.pt                    # Best model weights
│   └── model_config.json          # Model configuration
├── 📁 examples/                    # Example images and outputs
│   ├── input/                     # Sample input images
│   └── output/                    # Detection results
├── 📁 docs/                       # Documentation
│   ├── TRAINING.md                # Training guide
│   ├── API.md                     # API documentation
│   └── DEPLOYMENT.md              # Deployment guide
└── 📁 utils/                      # Utility scripts
    ├── export_formats.py          # Model export utilities
    └── visualization.py           # Visualization tools
```

## 🔧 **Usage Examples**

### **1. Basic Detection**
```python
from ultralytics import YOLO

model = YOLO('models/best.pt')
results = model.predict('image.jpg', conf=0.25, save=True)
```

### **2. Batch Processing**
```python
# Process multiple images
results = model.predict(['img1.jpg', 'img2.jpg', 'img3.jpg'])
```

### **3. Video Detection**
```python
# Process video file
results = model.predict('video.mp4', save=True)
```

### **4. Real-time Webcam**
```python
# Real-time detection from webcam
results = model.predict(source=0, show=True)
```

## 🎯 **Applications**

- 🛡️ **Security & Surveillance**: Perimeter monitoring, threat detection
- 🚁 **Aerospace & Defense**: Airspace monitoring, UAV detection
- 🏭 **Industrial Safety**: Workplace monitoring, safety compliance
- 🔬 **Research**: Computer vision, AI development
- 📱 **Mobile Apps**: Real-time object detection applications

## 📈 **Training Details**

- **Architecture**: YOLOv8n (nano)
- **Input Size**: 640×640 pixels
- **Training Epochs**: 100
- **Batch Size**: 16
- **Training Time**: 13.01 hours
- **Dataset**: Custom multi-class dataset
- **Augmentation**: HSV, rotation, scaling, flipping

## 🚀 **Advanced Features**

### **Smart Classification**
- Automatic human/animal disambiguation
- Context-aware corrections (weapon proximity)
- Aspect ratio analysis

### **Multi-Format Export**
- PyTorch (.pt)
- ONNX (.onnx)
- TensorRT (.engine)
- CoreML (.mlmodel)
- TensorFlow Lite (.tflite)

### **Visualization Tools**
- Color-coded bounding boxes
- Confidence score display
- Class legend overlay
- Detection statistics

## 📚 **Documentation**

- [📖 Training Guide](docs/TRAINING.md) - How to train your own model
- [🔧 API Reference](docs/API.md) - Complete API documentation
- [🚀 Deployment Guide](docs/DEPLOYMENT.md) - Production deployment
- [💡 Examples](examples/) - Code examples and tutorials

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- [Ultralytics](https://ultralytics.com) for the YOLOv8 framework
- [PyTorch](https://pytorch.org) for the deep learning framework
- HAL Aerospace for project inspiration and requirements

## 📞 **Contact**

- **Author**: Your Name
- **Email**: your.email@example.com
- **Project Link**: https://github.com/YOUR_USERNAME/yolo-multiclass-detection

## 🌟 **Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=YOUR_USERNAME/yolo-multiclass-detection&type=Date)](https://star-history.com/#YOUR_USERNAME/yolo-multiclass-detection&Date)

---

**⭐ If this project helped you, please give it a star! ⭐**
