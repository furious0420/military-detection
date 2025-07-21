#!/usr/bin/env python3
"""
Script to test different confidence thresholds and find optimal settings.
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

def test_multiple_thresholds(model, image_path):
    """
    Test multiple confidence thresholds to find optimal detection.
    """
    print(f"\n{'='*60}")
    print(f"TESTING MULTIPLE THRESHOLDS ON: {Path(image_path).name}")
    print(f"{'='*60}")
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return None
    
    # Test different confidence thresholds
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    best_threshold = None
    best_detections = None
    human_found = False
    
    for conf_thresh in thresholds:
        print(f"\n🎯 Testing confidence threshold: {conf_thresh}")
        print("-" * 40)
        
        # Run inference
        results = model.predict(
            source=image_path,
            save=False,
            conf=conf_thresh,
            iou=0.45,
            verbose=False
        )
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                classes = boxes.cls.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                
                print(f"   Found {len(boxes)} objects:")
                
                class_counts = {}
                for cls_id, conf in zip(classes, confidences):
                    class_name = model.names[int(cls_id)]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    print(f"   - {class_name}: {conf:.3f} ({conf*100:.1f}%)")
                    
                    # Check if human is detected
                    if class_name == 'human' and not human_found:
                        human_found = True
                        best_threshold = conf_thresh
                        best_detections = (classes, confidences, boxes)
                        print(f"   ✅ HUMAN DETECTED! Best threshold so far: {conf_thresh}")
                
                # Summary for this threshold
                summary = ", ".join([f"{cls}: {count}" for cls, count in class_counts.items()])
                print(f"   Summary: {summary}")
                
            else:
                print("   No objects detected")
        else:
            print("   No results")
    
    # Final recommendation
    print(f"\n{'='*60}")
    print("THRESHOLD ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    if human_found:
        print(f"✅ HUMAN DETECTED at threshold: {best_threshold}")
        print(f"🎯 RECOMMENDED THRESHOLD: {best_threshold}")
        
        # Show final detection with recommended threshold
        print(f"\n🔍 Final detection with threshold {best_threshold}:")
        results = model.predict(
            source=image_path,
            save=True,
            conf=best_threshold,
            iou=0.45,
            show_labels=True,
            show_conf=True,
            line_width=2
        )
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                classes = boxes.cls.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                coordinates = boxes.xyxy.cpu().numpy()
                
                for i, (cls_id, conf, coord) in enumerate(zip(classes, confidences, coordinates)):
                    class_name = model.names[int(cls_id)]
                    x1, y1, x2, y2 = coord
                    
                    print(f"  {i+1}. {class_name.upper()}")
                    print(f"     Confidence: {conf:.3f} ({conf*100:.1f}%)")
                    print(f"     Bounding Box: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
                    print()
        
    else:
        print("❌ NO HUMAN DETECTED at any threshold")
        print("🔍 This might indicate:")
        print("   - The model needs more human training data")
        print("   - The image type (thermal) is different from training data")
        print("   - The human is being misclassified as 'animal'")
        
        # Show what was detected instead
        print(f"\n📊 What was detected instead:")
        results = model.predict(
            source=image_path,
            save=True,
            conf=0.1,  # Low threshold to see everything
            iou=0.45,
            show_labels=True,
            show_conf=True,
            line_width=2
        )
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                classes = boxes.cls.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                
                class_summary = {}
                for cls_id, conf in zip(classes, confidences):
                    class_name = model.names[int(cls_id)]
                    if class_name not in class_summary:
                        class_summary[class_name] = []
                    class_summary[class_name].append(conf)
                
                for class_name, confs in class_summary.items():
                    avg_conf = sum(confs) / len(confs)
                    print(f"   {class_name}: {len(confs)} detections, avg confidence: {avg_conf:.3f}")
    
    return best_threshold

def main():
    """Main function to test thresholds."""
    
    # Image path - you can change this
    image_path = r"B:\eoir\hu_gun.jpg"
    
    try:
        # Load the saved model
        model = load_model_from_pkl()
        
        # Test multiple thresholds
        best_threshold = test_multiple_thresholds(model, image_path)
        
        print(f"\n🎉 Threshold testing completed!")
        if best_threshold:
            print(f"🎯 Use confidence threshold: {best_threshold} for best human detection")
        else:
            print("🔧 Consider retraining with more diverse human data")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
