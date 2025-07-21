#!/usr/bin/env python3
"""
Script to run inference on a specific image using the saved YOLO model.
"""

import pickle
from pathlib import Path
import cv2

def load_model_from_pkl(pkl_file='yolo_model_and_metrics.pkl'):
    """Load the saved model from pickle file."""
    print("Loading saved YOLO model...")
    with open(pkl_file, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    print(f"✓ Model loaded successfully!")
    print(f"✓ Classes: {list(model.names.values())}")
    return model

def detect_objects_in_image(model, image_path, save_results=True):
    """
    Run object detection on the specified image.
    
    Args:
        model: Loaded YOLO model
        image_path: Path to the image file
        save_results: Whether to save the annotated image
    """
    print(f"\n{'='*50}")
    print(f"RUNNING DETECTION ON: {image_path}")
    print(f"{'='*50}")
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return None
    
    # Run inference with multiple confidence levels
    print("🔍 Running inference...")

    # First try with very low confidence to see all detections
    print("📊 Testing with confidence threshold 0.05...")
    results_low = model.predict(
        source=image_path,
        save=False,  # Don't save yet
        conf=0.05,   # Very low confidence threshold
        iou=0.45,    # IoU threshold for NMS
        verbose=False
    )

    # Show all detections found
    if results_low and len(results_low) > 0:
        result = results_low[0]
        if result.boxes is not None and len(result.boxes) > 0:
            print("🔍 All detections found (conf >= 0.05):")
            boxes = result.boxes
            classes = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()

            for i, (cls_id, conf) in enumerate(zip(classes, confidences)):
                class_name = model.names[int(cls_id)]
                print(f"   {class_name}: {conf:.3f} ({conf*100:.1f}%)")

    # Now run with optimal threshold
    print("\n🎯 Running with optimized threshold...")
    results = model.predict(
        source=image_path,
        save=save_results,
        conf=0.3,    # Higher confidence threshold for final results
        iou=0.45,    # IoU threshold for NMS
        show_labels=True,
        show_conf=True,
        line_width=2
    )
    
    # Process results
    if results and len(results) > 0:
        result = results[0]  # Get first (and only) result
        
        print(f"\n📊 DETECTION RESULTS:")
        print(f"{'='*30}")
        
        if result.boxes is not None and len(result.boxes) > 0:
            # Get detection details
            boxes = result.boxes
            classes = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            coordinates = boxes.xyxy.cpu().numpy()
            
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
            
        else:
            print("❌ No objects detected in the image.")
            print("   Try lowering the confidence threshold or check if the image contains the trained classes:")
            print(f"   Trained classes: {list(model.names.values())}")
        
        # Show where results are saved
        if save_results:
            print(f"\n💾 SAVED RESULTS:")
            print(f"{'='*20}")
            print(f"✓ Annotated image saved in: runs/detect/predict/")
            print(f"✓ Check the 'runs' folder for the output image with bounding boxes")
    
    return results

def main():
    """Main function to run detection on the specified image."""
    
    # Image path
    image_path = r"B:\eoir\hu_gun.jpg"
    
    try:
        # Load the saved model
        model = load_model_from_pkl()
        
        # Run detection
        results = detect_objects_in_image(model, image_path, save_results=True)
        
        if results:
            print(f"\n🎉 Detection completed successfully!")
            print(f"📁 Check the 'runs/detect/predict/' folder for the annotated image.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        if "yolo_model_and_metrics.pkl" in str(e):
            print("   Make sure you have run the model saving script first.")
        else:
            print("   Check the image path and make sure the file exists.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
