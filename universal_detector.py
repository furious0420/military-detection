#!/usr/bin/env python3
"""
Universal Multi-Class Detection Script
Usage: python universal_detector.py "path/to/your/image.jpg"
"""

import sys
import pickle
import cv2
import numpy as np
from pathlib import Path

def load_model_from_pkl(pkl_file='yolo_model_and_metrics.pkl'):
    """Load the saved model from pickle file."""
    try:
        with open(pkl_file, 'rb') as f:
            model_data = pickle.load(f)
        model = model_data['model']
        return model
    except FileNotFoundError:
        print(f"❌ Error: Model file '{pkl_file}' not found!")
        print("   Make sure you have the trained model pickle file in the current directory.")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def get_class_colors():
    """Define distinct colors for each class (BGR format for OpenCV)."""
    class_colors = {
        'human': (0, 255, 0),      # Green
        'animal': (255, 0, 0),     # Blue  
        'vehicle': (0, 0, 255),    # Red
        'drone': (255, 255, 0),    # Cyan
        'weapon': (0, 165, 255)    # Orange
    }
    return class_colors

def apply_smart_classification(detections, model):
    """Apply intelligent classification rules to improve accuracy."""
    corrected_detections = []
    
    for detection in detections:
        cls_id, conf, coord = detection
        class_name = model.names[int(cls_id)]
        x1, y1, x2, y2 = coord
        
        # Calculate object properties
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = height / width if width > 0 else 1
        
        # Apply smart classification rules
        corrected_class = class_name
        correction_reason = ""
        
        # Rule: Correct animal->human misclassification
        if class_name == 'animal':
            # Human-like aspect ratio (tall and narrow)
            if 1.8 < aspect_ratio < 4.0:
                # Check for weapon proximity
                weapon_nearby = False
                for other_detection in detections:
                    other_cls_id, other_conf, other_coord = other_detection
                    other_class = model.names[int(other_cls_id)]
                    
                    if other_class == 'weapon':
                        ox1, oy1, ox2, oy2 = other_coord
                        # Check overlap or proximity
                        overlap_x = max(0, min(x2, ox2) - max(x1, ox1))
                        overlap_y = max(0, min(y2, oy2) - max(y1, oy1))
                        
                        if overlap_x > 0 and overlap_y > 0:
                            weapon_nearby = True
                            break
                
                # Apply correction based on aspect ratio and context
                if weapon_nearby or aspect_ratio > 2.5:
                    corrected_class = 'human'
                    correction_reason = f"Aspect ratio: {aspect_ratio:.2f}"
                    if weapon_nearby:
                        correction_reason += " + weapon proximity"
        
        corrected_detections.append({
            'class': corrected_class,
            'original_class': class_name,
            'confidence': conf,
            'coordinates': coord,
            'correction_reason': correction_reason,
            'aspect_ratio': aspect_ratio
        })
    
    return corrected_detections

def detect_and_visualize(image_path, confidence_threshold=0.25):
    """
    Universal detection function for any image.
    
    Args:
        image_path (str): Path to the input image
        confidence_threshold (float): Minimum confidence for detection
    
    Returns:
        str: Path to the output image with bounding boxes
    """
    
    print(f"\n{'='*70}")
    print(f"🔍 UNIVERSAL MULTI-CLASS DETECTION")
    print(f"{'='*70}")
    print(f"📁 Input Image: {Path(image_path).name}")
    print(f"🎯 Confidence Threshold: {confidence_threshold}")
    
    # Load model
    print("\n📦 Loading YOLO model...")
    model = load_model_from_pkl()
    if model is None:
        return None
    
    print(f"✅ Model loaded successfully!")
    print(f"🏷️  Available classes: {list(model.names.values())}")
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return None
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Error: Could not load image: {image_path}")
        return None
    
    print(f"📐 Image dimensions: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Run inference
    print("\n🔍 Running multi-class detection...")
    results = model.predict(
        source=image_path,
        save=False,
        conf=confidence_threshold,
        iou=0.45,
        verbose=False
    )
    
    # Get class colors
    class_colors = get_class_colors()
    
    if results and len(results) > 0:
        result = results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            classes = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            coordinates = boxes.xyxy.cpu().numpy()
            
            # Prepare detections for smart classification
            raw_detections = []
            for cls_id, conf, coord in zip(classes, confidences, coordinates):
                raw_detections.append((cls_id, conf, coord))
            
            # Apply smart classification
            smart_detections = apply_smart_classification(raw_detections, model)
            
            print(f"\n✅ DETECTION RESULTS:")
            print(f"{'='*50}")
            print(f"🎯 Found {len(smart_detections)} objects:")
            
            # Draw boxes and labels
            for i, detection in enumerate(smart_detections):
                class_name = detection['class']
                original_class = detection['original_class']
                conf = detection['confidence']
                x1, y1, x2, y2 = detection['coordinates']
                correction_reason = detection['correction_reason']
                aspect_ratio = detection['aspect_ratio']
                
                # Get color for this class
                color = class_colors.get(class_name, (128, 128, 128))
                
                # Draw bounding box with thick border
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                
                # Prepare label
                label = f"{class_name.upper()}: {conf:.2f}"
                
                # Calculate label dimensions
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )
                
                # Draw label background
                cv2.rectangle(
                    image,
                    (int(x1), int(y1) - label_height - 15),
                    (int(x1) + label_width + 10, int(y1)),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    image,
                    label,
                    (int(x1) + 5, int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
                
                # Print detection details
                print(f"\n  📦 Detection {i+1}:")
                print(f"     🏷️  Class: {class_name.upper()}")
                if correction_reason:
                    print(f"     🔄 Corrected from: {original_class} ({correction_reason})")
                print(f"     📊 Confidence: {conf:.3f} ({conf*100:.1f}%)")
                print(f"     📐 Bounding Box: ({int(x1)}, {int(y1)}) → ({int(x2)}, {int(y2)})")
                print(f"     📏 Size: {int(x2-x1)} × {int(y2-y1)} pixels")
                print(f"     🎨 Box Color: {color} (BGR)")
                print(f"     📊 Aspect Ratio: {aspect_ratio:.2f}")
            
            # Add legend to image
            legend_start_y = 40
            legend_bg_height = len(class_colors) * 35 + 50
            
            # Draw legend background
            cv2.rectangle(image, (10, 10), (250, legend_bg_height), (0, 0, 0), -1)
            cv2.rectangle(image, (10, 10), (250, legend_bg_height), (255, 255, 255), 2)
            
            # Legend title
            cv2.putText(image, "CLASS LEGEND:", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw legend items
            legend_y = legend_start_y + 25
            for class_name, color in class_colors.items():
                # Draw color box
                cv2.rectangle(image, (20, legend_y - 12), (45, legend_y + 8), color, -1)
                cv2.rectangle(image, (20, legend_y - 12), (45, legend_y + 8), (255, 255, 255), 1)
                
                # Draw class name
                cv2.putText(image, class_name.upper(), (55, legend_y + 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                legend_y += 30
            
            # Generate output filename
            input_stem = Path(image_path).stem
            output_path = f"detected_{input_stem}_multiclass.jpg"
            
            # Save result
            cv2.imwrite(output_path, image)
            
            # Summary
            print(f"\n📊 DETECTION SUMMARY:")
            print(f"{'='*40}")
            class_counts = {}
            for detection in smart_detections:
                class_name = detection['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in class_counts.items():
                color = class_colors.get(class_name, (128, 128, 128))
                print(f"  🏷️  {class_name.upper()}: {count} detected (Color: {color})")
            
            print(f"\n💾 OUTPUT SAVED:")
            print(f"✅ File: {output_path}")
            print(f"✅ Multi-class bounding boxes drawn")
            print(f"✅ Color legend included")
            print(f"✅ Confidence scores displayed")
            
            return output_path
            
        else:
            print("\n❌ No objects detected in the image.")
            print(f"💡 Try lowering the confidence threshold (current: {confidence_threshold})")
            return None
    else:
        print("\n❌ No detection results returned.")
        return None

def main():
    """Main function with command line argument support."""
    
    # Default image path
    default_image = r"B:\eoir\hu_gun.jpg"
    
    # Check for command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = default_image
        print(f"💡 No image path provided. Using default: {Path(image_path).name}")
        print(f"💡 Usage: python universal_detector.py \"path/to/your/image.jpg\"")
    
    # Optional confidence threshold
    confidence = 0.25
    if len(sys.argv) > 2:
        try:
            confidence = float(sys.argv[2])
        except ValueError:
            print(f"⚠️  Invalid confidence value. Using default: {confidence}")
    
    try:
        # Run detection
        output_path = detect_and_visualize(image_path, confidence)
        
        if output_path:
            print(f"\n🎉 SUCCESS! Multi-class detection completed!")
            print(f"📁 Check output: {output_path}")
        else:
            print(f"\n❌ Detection failed or no objects found.")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
