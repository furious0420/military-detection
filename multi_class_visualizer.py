#!/usr/bin/env python3
"""
Multi-class detection visualizer with distinct colored boxes for each class.
"""

import pickle
import cv2
import numpy as np
from pathlib import Path

def load_model_from_pkl(pkl_file='yolo_model_and_metrics.pkl'):
    """Load the saved model from pickle file."""
    print("Loading saved YOLO model...")
    with open(pkl_file, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    print(f"✓ Model loaded successfully!")
    print(f"✓ Classes: {list(model.names.values())}")
    return model

def get_class_colors():
    """Define distinct colors for each class."""
    class_colors = {
        'human': (0, 255, 0),      # Green
        'animal': (255, 0, 0),     # Blue  
        'vehicle': (0, 0, 255),    # Red
        'drone': (255, 255, 0),    # Cyan
        'weapon': (0, 165, 255)    # Orange
    }
    return class_colors

def apply_smart_classification(detections, model):
    """Apply smart classification rules."""
    corrected_detections = []
    
    for detection in detections:
        cls_id, conf, coord = detection
        class_name = model.names[int(cls_id)]
        x1, y1, x2, y2 = coord
        
        # Calculate properties
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = height / width if width > 0 else 1
        
        # Smart classification
        corrected_class = class_name
        correction_applied = False
        
        if class_name == 'animal':
            # Rule 1: Human-like aspect ratio
            if 1.8 < aspect_ratio < 4.0:
                # Rule 2: Check for weapon proximity
                weapon_nearby = False
                for other_detection in detections:
                    other_cls_id, other_conf, other_coord = other_detection
                    other_class = model.names[int(other_cls_id)]
                    
                    if other_class == 'weapon':
                        ox1, oy1, ox2, oy2 = other_coord
                        # Check overlap
                        overlap_x = max(0, min(x2, ox2) - max(x1, ox1))
                        overlap_y = max(0, min(y2, oy2) - max(y1, oy1))
                        
                        if overlap_x > 0 and overlap_y > 0:
                            weapon_nearby = True
                            break
                
                if weapon_nearby or aspect_ratio > 2.5:
                    corrected_class = 'human'
                    correction_applied = True
        
        corrected_detections.append({
            'class': corrected_class,
            'original_class': class_name,
            'confidence': conf,
            'coordinates': coord,
            'corrected': correction_applied
        })
    
    return corrected_detections

def draw_multi_class_boxes(image_path, model, output_path=None):
    """
    Draw colored bounding boxes for multi-class detection.
    """
    print(f"\n{'='*60}")
    print(f"MULTI-CLASS VISUALIZATION: {Path(image_path).name}")
    print(f"{'='*60}")
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return None
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Error: Could not load image: {image_path}")
        return None
    
    original_image = image.copy()
    
    # Run inference
    print("🔍 Running multi-class detection...")
    results = model.predict(
        source=image_path,
        save=False,
        conf=0.3,
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
            
            # Prepare detections
            raw_detections = []
            for cls_id, conf, coord in zip(classes, confidences, coordinates):
                raw_detections.append((cls_id, conf, coord))
            
            # Apply smart classification
            smart_detections = apply_smart_classification(raw_detections, model)
            
            print(f"✓ Found {len(smart_detections)} objects:")
            
            # Draw boxes for each detection
            for i, detection in enumerate(smart_detections):
                class_name = detection['class']
                original_class = detection['original_class']
                conf = detection['confidence']
                x1, y1, x2, y2 = detection['coordinates']
                corrected = detection['corrected']
                
                # Get color for this class
                color = class_colors.get(class_name, (128, 128, 128))  # Default gray
                
                # Draw bounding box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                
                # Prepare label
                label = f"{class_name.upper()}: {conf:.2f}"
                if corrected:
                    label += f" (was {original_class})"
                
                # Calculate label size and position
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw label background
                cv2.rectangle(
                    image,
                    (int(x1), int(y1) - label_height - 10),
                    (int(x1) + label_width, int(y1)),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    image,
                    label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),  # White text
                    2
                )
                
                # Print detection info
                print(f"  {i+1}. {class_name.upper()}")
                if corrected:
                    print(f"     🔄 Corrected from: {original_class}")
                print(f"     Confidence: {conf:.3f} ({conf*100:.1f}%)")
                print(f"     Box Color: RGB{color}")
                print(f"     Coordinates: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
                print()
            
            # Create legend
            legend_y = 30
            cv2.putText(image, "CLASS LEGEND:", (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            legend_y += 30
            for class_name, color in class_colors.items():
                # Draw color box
                cv2.rectangle(image, (10, legend_y - 15), (30, legend_y + 5), color, -1)
                # Draw class name
                cv2.putText(image, class_name.upper(), (40, legend_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                legend_y += 25
            
            # Save output image
            if output_path is None:
                output_path = f"multi_class_detection_{Path(image_path).stem}.jpg"
            
            cv2.imwrite(output_path, image)
            
            print(f"📊 MULTI-CLASS DETECTION SUMMARY:")
            print(f"{'='*40}")
            class_counts = {}
            for detection in smart_detections:
                class_name = detection['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in class_counts.items():
                color = class_colors.get(class_name, (128, 128, 128))
                print(f"  {class_name.upper()}: {count} detected (Color: RGB{color})")
            
            print(f"\n💾 VISUALIZATION SAVED:")
            print(f"✓ Output image: {output_path}")
            print(f"✓ Original image preserved")
            print(f"✓ Multi-class boxes with distinct colors")
            print(f"✓ Legend included in image")
            
            return output_path
            
        else:
            print("❌ No objects detected in the image.")
            return None
    else:
        print("❌ No detection results.")
        return None

def main():
    """Main function."""
    
    # Image path
    image_path = r"B:\eoir\hu_gun.jpg"
    output_path = "multi_class_thermal_detection.jpg"
    
    try:
        # Load the saved model
        model = load_model_from_pkl()
        
        # Create multi-class visualization
        result_path = draw_multi_class_boxes(image_path, model, output_path)
        
        if result_path:
            print(f"\n🎉 Multi-class visualization completed!")
            print(f"📁 Check the output image: {result_path}")
            print(f"🎨 Each class has a distinct colored bounding box")
            print(f"📋 Legend shows color coding for all classes")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
