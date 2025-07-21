# 🎯 Military Object Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-furious0420/military--detection-blue.svg)](https://github.com/furious0420/military-detection)

A high-performance military-grade object detection system built with YOLOv8, capable of detecting **humans, animals, vehicles, drones, and weapons** in real-time with **97.99% mAP@0.5** accuracy. Designed for defense, surveillance, and security applications.

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
git clone https://github.com/furious0420/military-detection.git
cd military-detection

# Install dependencies
pip install -r requirements.txt
```

### **Download Pre-trained Model**
```bash
# Download the trained model from releases
wget https://github.com/furious0420/military-detection/releases/download/v1.0/best.pt
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

### **Interactive Web Application**
```bash
# Launch the Streamlit web interface
streamlit run lit.py
```

The web application provides:
- 🖥️ Interactive UI for model testing
- 📊 Real-time detection visualization
- 📷 Support for image, video, and webcam input
- 🔄 Automatic thermal/visible image classification
- 📈 Detection analytics and statistics
- 🎛️ Adjustable confidence thresholds
- 🎨 Image enhancement tools

## 🎨 **Class Color Coding**

| Class | Color | Detection Focus |
|-------|-------|----------------|
| 🟢 **Human** | Green | People detection |
| 🔵 **Animal** | Blue | Animals (dogs, cats, wildlife) |
| 🔴 **Vehicle** | Red | Cars, trucks, buses |
| 🟡 **Drone** | Cyan | UAVs, quadcopters |
| 🟠 **Weapon** | Orange | Guns, knives, weapons |

## 🌐 **Interactive Web Application**

The `lit.py` file provides a comprehensive **Streamlit web interface** for the YOLO detection system:

### **🎯 Key Features:**
- **🔄 Multi-Model Support**: Supports visible, thermal, and aerial detection models
- **🧠 Smart Classification**: Automatic thermal/visible image type detection
- **📷 Multiple Input Types**: Images, videos, and real-time webcam streams
- **🎛️ Interactive Controls**: Adjustable confidence, NMS, and detection thresholds
- **📊 Real-time Analytics**: Live detection statistics and performance metrics
- **🎨 Image Enhancement**: Brightness, contrast, sharpness, and saturation controls
- **📈 Advanced Visualization**: Interactive charts, detection timelines, and heatmaps
- **🌓 Dual Themes**: Dark/Light mode with animated UI elements
- **🔍 Debug Mode**: Detailed frame-by-frame detection analysis
- **⚡ Performance Optimization**: Frame skipping and GPU acceleration
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices
- **🎬 Video Processing**: Batch video analysis with progress tracking

### **🚀 Launch Web App:**
```bash
streamlit run lit.py
```

### **📱 Web Interface Tabs:**
1. **🏠 Home**: Model overview, performance metrics, and system status
2. **🖼️ Image Detection**:
   - Drag & drop image upload
   - Automatic thermal/visible classification
   - Real-time confidence adjustment
   - Smart classification corrections
   - Downloadable results
3. **🎬 Video Processing**:
   - Video file upload and analysis
   - Frame-by-frame detection
   - Confidence threshold testing
   - Detection timeline visualization
   - Class distribution analytics
4. **📷 Real-time Detection**:
   - Live webcam streaming
   - Real-time object detection
   - Performance monitoring
   - FPS optimization controls
5. **🛩️ Aerial Surveillance**:
   - Specialized aerial model
   - Enhanced detection for drones/aircraft
   - Large-scale area monitoring
   - Multi-object tracking
6. **📊 Analytics**:
   - Detection performance metrics
   - Historical data analysis
   - Model comparison charts
   - Export capabilities

### **🔧 What lit.py Does:**

**Core Functionality:**
- **Multi-Model Management**: Loads and manages multiple YOLO models (visible, thermal, aerial)
- **Intelligent Image Classification**: Automatically detects if an image is thermal or visible using advanced algorithms
- **Real-time Processing**: Handles live video streams with optimized frame processing
- **Interactive UI**: Provides a user-friendly web interface with real-time controls
- **Performance Monitoring**: Tracks detection speed, accuracy, and system resources

**Advanced Features:**
- **Smart Detection Correction**: Applies post-processing rules to improve accuracy
- **Video Analysis**: Processes video files with batch detection and analytics
- **Confidence Testing**: Tests multiple confidence levels to find optimal settings
- **Debug Mode**: Provides detailed detection information for troubleshooting
- **Export Capabilities**: Saves results in multiple formats (images, CSV, JSON)

**Technical Implementation:**
- **Streamlit Framework**: Modern web application with reactive components
- **WebRTC Integration**: Real-time video streaming capabilities
- **OpenCV Processing**: Advanced image and video manipulation
- **Plotly Visualization**: Interactive charts and graphs
- **Threading Support**: Concurrent processing for better performance

## 📁 **Repository Structure**

```
military-detection/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 LICENSE                      # MIT License
├── 🌐 lit.py                       # Streamlit web application
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

### **1. Web Application (Recommended)**
```bash
# Launch interactive web interface
streamlit run lit.py

# Features:
# - Drag & drop image/video upload
# - Real-time webcam detection
# - Adjustable confidence thresholds
# - Detection analytics and charts
# - Automatic thermal/visible classification
```

### **2. Command Line Detection**
```bash
# Detect objects in any image with colored bounding boxes
python universal_detector.py "path/to/your/image.jpg"

# With custom confidence threshold
python universal_detector.py "image.jpg" 0.1
```

### **3. Python API**
```python
from ultralytics import YOLO

# Load model
model = YOLO('models/best.pt')

# Basic detection
results = model.predict('image.jpg', conf=0.25, save=True)

# Batch processing
results = model.predict(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# Video detection
results = model.predict('video.mp4', save=True)

# Real-time webcam
results = model.predict(source=0, show=True)
```

### **4. Advanced Configuration**
```python
# Custom detection parameters
results = model.predict(
    source='image.jpg',
    conf=0.25,          # Confidence threshold
    iou=0.45,           # NMS IoU threshold
    max_det=100,        # Maximum detections
    save=True,          # Save results
    show_labels=True,   # Show class labels
    show_conf=True      # Show confidence scores
)
```

## 🎯 **Applications**

### **🛡️ Security & Surveillance**
- **Perimeter monitoring** with real-time alerts
- **Threat detection** (weapons, unauthorized personnel)
- **Access control** with facial recognition integration
- **Incident analysis** with video playback

### **🚁 Aerospace & Defense**
- **Airspace monitoring** for UAV detection
- **Border surveillance** with thermal imaging
- **Military reconnaissance** and target identification
- **Search and rescue** operations

### **🏭 Industrial & Commercial**
- **Workplace safety** monitoring
- **Equipment inspection** and maintenance
- **Quality control** in manufacturing
- **Inventory management** with automated counting

### **🌐 Web-Based Solutions**
- **Remote monitoring** dashboards
- **Cloud-based** detection services
- **Multi-user** collaboration platforms
- **API integration** for third-party applications

### **🔬 Research & Development**
- **Computer vision** research
- **AI model** development and testing
- **Dataset annotation** and validation
- **Performance benchmarking**

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

- **Author**: furious0420
- **GitHub**: [@furious0420](https://github.com/furious0420)
- **Project Link**: https://github.com/furious0420/military-detection

## 🌟 **Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=furious0420/military-detection&type=Date)](https://star-history.com/#furious0420/military-detection&Date)

---

**⭐ If this military detection system helped you, please give it a star! ⭐**
