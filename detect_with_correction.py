#!/usr/bin/env python3
"""
Detection script with human/animal classification correction.
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

def correct_human_animal_classification(results, model):
    """
    Apply heuristics to correct human/animal misclassification.
    """
    if not results or len(results) == 0:
        return results
    
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return results
    
    boxes = result.boxes
    classes = boxes.cls.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    coordinates = boxes.xyxy.cpu().numpy()
    
    corrected_classes = []
    correction_applied = False
    
    for i, (cls_id, conf, coord) in enumerate(zip(classes, confidences, coordinates)):
        class_name = model.names[int(cls_id)]
        x1, y1, x2, y2 = coord
        
        # Calculate aspect ratio and size
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = height / width if width > 0 else 1
        
        # Heuristics for human detection
        is_likely_human = False
        
        # Rule 1: If detected as "animal" with human-like proportions
        if class_name == 'animal':
            # Human-like aspect ratio (taller than wide)
            if aspect_ratio > 1.5 and aspect_ratio < 4.0:
                is_likely_human = True
                print(f"   🔄 Correction applied: animal → human (aspect ratio: {aspect_ratio:.2f})")
            
            # Rule 2: If there's a weapon nearby, likely human
            weapon_nearby = False
            for j, (other_cls_id, other_coord) in enumerate(zip(classes, coordinates)):
                if i != j and model.names[int(other_cls_id)] == 'weapon':
                    # Check if weapon is close to this detection
                    ox1, oy1, ox2, oy2 = other_coord
                    overlap_x = max(0, min(x2, ox2) - max(x1, ox1))
                    overlap_y = max(0, min(y2, oy2) - max(y1, oy1))
                    if overlap_x > 0 and overlap_y > 0:
                        weapon_nearby = True
                        break
            
            if weapon_nearby:
                is_likely_human = True
                print(f"   🔄 Correction applied: animal → human (weapon detected nearby)")
        
        # Apply correction
        if is_likely_human:
            corrected_classes.append(0)  # human class ID
            correction_applied = True
        else:
            corrected_classes.append(int(cls_id))
    
    # Update the results if corrections were applied
    if correction_applied:
        import torch
        result.boxes.cls = torch.tensor(corrected_classes, dtype=torch.float32)
        print(f"✅ Applied {sum(1 for i, orig in enumerate(classes) if corrected_classes[i] != orig)} corrections")
    
    return results

def detect_with_correction(model, image_path, save_results=True):
    """
    Run detection with human/animal classification correction.
    """
    print(f"\n{'='*50}")
    print(f"RUNNING CORRECTED DETECTION ON: {Path(image_path).name}")
    print(f"{'='*50}")
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return None
    
    # Run inference
    print("🔍 Running inference...")
    results = model.predict(
        source=image_path,
        save=False,  # We'll save after correction
        conf=0.3,    # Reasonable confidence threshold
        iou=0.45,
        verbose=False
    )
    
    # Apply corrections
    print("🔄 Applying classification corrections...")
    corrected_results = correct_human_animal_classification(results, model)
    
    # Display results
    if corrected_results and len(corrected_results) > 0:
        result = corrected_results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            classes = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            coordinates = boxes.xyxy.cpu().numpy()
            
            print(f"\n📊 CORRECTED DETECTION RESULTS:")
            print(f"{'='*35}")
            print(f"✓ Found {len(boxes)} objects:")
            print()
            
            # Display each detection
            for i, (cls_id, conf, coord) in enumerate(zip(classes, confidences, coordinates)):
                class_name = model.names[int(cls_id)]
                x1, y1, x2, y2 = coord
                
                print(f"  {i+1}. {class_name.upper()}")
                print(f"     Confidence: {conf:.3f} ({conf*100:.1f}%)")
                print(f"     Bounding Box: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
                print(f"     Size: {int(x2-x1)} x {int(y2-y1)} pixels")
                print()
            
            # Summary by class
            print(f"📋 SUMMARY BY CLASS:")
            print(f"{'='*25}")
            class_counts = {}
            for cls_id in classes:
                class_name = model.names[int(cls_id)]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count} detected")
            
            # Save corrected results
            if save_results:
                print(f"\n💾 SAVING CORRECTED RESULTS...")
                final_results = model.predict(
                    source=image_path,
                    save=True,
                    conf=0.3,
                    iou=0.45,
                    show_labels=True,
                    show_conf=True,
                    line_width=2
                )
                print(f"✓ Results saved to runs/detect/predict/ folder")
        
        else:
            print("❌ No objects detected in the image.")
    
    return corrected_results

def main():
    """Main function."""
    
    # Image path
    image_path = r"B:\eoir\hu_gun.jpg"
    
    try:
        # Load the saved model
        model = load_model_from_pkl()
        
        # Run detection with correction
        results = detect_with_correction(model, image_path, save_results=True)
        
        print(f"\n🎉 Corrected detection completed!")
        print(f"📁 Check the 'runs/detect/predict/' folder for the annotated image.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
