#!/usr/bin/env python3
"""
Smart detection script that corrects human/animal misclassification in post-processing.
"""

import pickle
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

def smart_classify_detections(detections, model):
    """
    Apply intelligent classification rules to correct misclassifications.
    """
    corrected_detections = []
    
    for detection in detections:
        cls_id, conf, coord = detection
        class_name = model.names[int(cls_id)]
        x1, y1, x2, y2 = coord
        
        # Calculate object properties
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = height / width if width > 0 else 1
        area = width * height
        
        # Smart classification rules
        corrected_class = class_name
        correction_reason = ""
        
        if class_name == 'animal':
            # Rule 1: Human-like aspect ratio (tall and narrow)
            if 1.8 < aspect_ratio < 4.0:
                corrected_class = 'human'
                correction_reason = f"Human-like aspect ratio ({aspect_ratio:.2f})"
            
            # Rule 2: Check for weapon proximity
            weapon_nearby = False
            for other_detection in detections:
                other_cls_id, other_conf, other_coord = other_detection
                other_class = model.names[int(other_cls_id)]
                
                if other_class == 'weapon':
                    ox1, oy1, ox2, oy2 = other_coord
                    
                    # Check overlap or proximity
                    overlap_x = max(0, min(x2, ox2) - max(x1, ox1))
                    overlap_y = max(0, min(y2, oy2) - max(y1, oy1))
                    
                    # Check if weapon is within or overlapping the animal detection
                    if overlap_x > 0 and overlap_y > 0:
                        weapon_nearby = True
                        break
                    
                    # Check proximity (weapon center near animal bounds)
                    weapon_center_x = (ox1 + ox2) / 2
                    weapon_center_y = (oy1 + oy2) / 2
                    
                    if x1 <= weapon_center_x <= x2 and y1 <= weapon_center_y <= y2:
                        weapon_nearby = True
                        break
            
            if weapon_nearby and corrected_class == 'animal':
                corrected_class = 'human'
                correction_reason = "Weapon detected in proximity"
            elif weapon_nearby and corrected_class == 'human':
                correction_reason += " + weapon proximity"
        
        corrected_detections.append({
            'original_class': class_name,
            'corrected_class': corrected_class,
            'confidence': conf,
            'coordinates': coord,
            'correction_reason': correction_reason,
            'aspect_ratio': aspect_ratio,
            'area': area
        })
    
    return corrected_detections

def detect_with_smart_classification(model, image_path, save_results=True):
    """
    Run detection with smart classification correction.
    """
    print(f"\n{'='*60}")
    print(f"SMART DETECTION ON: {Path(image_path).name}")
    print(f"{'='*60}")
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return None
    
    # Run inference
    print("🔍 Running inference...")
    results = model.predict(
        source=image_path,
        save=save_results,
        conf=0.3,    # Good confidence threshold
        iou=0.45,
        show_labels=True,
        show_conf=True,
        line_width=2
    )
    
    # Process results
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
            print("🧠 Applying smart classification rules...")
            smart_detections = smart_classify_detections(raw_detections, model)
            
            # Display results
            print(f"\n📊 SMART DETECTION RESULTS:")
            print(f"{'='*40}")
            print(f"✓ Found {len(smart_detections)} objects:")
            print()
            
            corrections_made = 0
            
            for i, detection in enumerate(smart_detections):
                original_class = detection['original_class']
                corrected_class = detection['corrected_class']
                conf = detection['confidence']
                x1, y1, x2, y2 = detection['coordinates']
                reason = detection['correction_reason']
                aspect_ratio = detection['aspect_ratio']
                
                print(f"  {i+1}. {corrected_class.upper()}", end="")
                
                if original_class != corrected_class:
                    print(f" (corrected from {original_class})")
                    print(f"     🔄 Correction reason: {reason}")
                    corrections_made += 1
                else:
                    print()
                
                print(f"     Confidence: {conf:.3f} ({conf*100:.1f}%)")
                print(f"     Bounding Box: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
                print(f"     Size: {int(x2-x1)} x {int(y2-y1)} pixels")
                print(f"     Aspect Ratio: {aspect_ratio:.2f}")
                print()
            
            # Summary by corrected class
            print(f"📋 CORRECTED SUMMARY:")
            print(f"{'='*30}")
            corrected_class_counts = {}
            original_class_counts = {}
            
            for detection in smart_detections:
                corrected_class = detection['corrected_class']
                original_class = detection['original_class']
                
                corrected_class_counts[corrected_class] = corrected_class_counts.get(corrected_class, 0) + 1
                original_class_counts[original_class] = original_class_counts.get(original_class, 0) + 1
            
            for class_name, count in corrected_class_counts.items():
                print(f"  {class_name}: {count} detected")
            
            if corrections_made > 0:
                print(f"\n🔄 CORRECTIONS APPLIED: {corrections_made}")
                print("Original detections:")
                for class_name, count in original_class_counts.items():
                    print(f"  {class_name}: {count}")
            
            # Show where results are saved
            if save_results:
                print(f"\n💾 RESULTS SAVED:")
                print(f"✓ Annotated image saved in: runs/detect/predict/")
                print("⚠️  Note: Saved image shows original classifications")
                print("    Smart corrections are applied in this analysis only")
        
        else:
            print("❌ No objects detected in the image.")
    
    return results

def main():
    """Main function."""
    
    # Image path
    image_path = r"B:\eoir\hu_gun.jpg"
    
    try:
        # Load the saved model
        model = load_model_from_pkl()
        
        # Run smart detection
        results = detect_with_smart_classification(model, image_path, save_results=True)
        
        print(f"\n🎉 Smart detection completed!")
        print(f"📁 Check the 'runs/detect/predict/' folder for the annotated image.")
        print(f"🧠 Smart classification corrections applied in analysis above.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
