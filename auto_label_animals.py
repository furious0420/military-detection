from ultralytics import YOLO
from pathlib import Path
import cv2

def auto_label_animals():
    base_path = Path("b:/eoir/datasets/animals")
    output_path = base_path / "train"
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "images").mkdir(exist_ok=True)
    (output_path / "labels").mkdir(exist_ok=True)
    
    # Load pre-trained YOLO model
    model = YOLO('yolov8n.pt')  # Pre-trained on COCO dataset
    
    # COCO animal classes (convert to class 1)
    animal_classes = [15, 16, 17, 18, 19, 20, 21, 22, 23]  # bird, cat, dog, horse, sheep, cow, elephant, bear, zebra
    
    image_files = list(base_path.glob("*.jpg")) + list(base_path.glob("*.png"))
    labeled_count = 0
    
    for img_file in image_files:
        # Run detection
        results = model(str(img_file))
        
        # Check if any animals detected
        animal_detections = []
        for r in results:
            for box in r.boxes:
                if int(box.cls) in animal_classes:
                    # Convert to YOLO format
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    img_w, img_h = r.orig_shape[1], r.orig_shape[0]
                    
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    animal_detections.append(f"1 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        if animal_detections:
            # Copy image and create label
            new_img_path = output_path / "images" / img_file.name
            new_label_path = output_path / "labels" / f"{img_file.stem}.txt"
            
            # Copy image
            import shutil
            shutil.copy2(img_file, new_img_path)
            
            # Write labels
            with open(new_label_path, 'w') as f:
                for detection in animal_detections:
                    f.write(f"{detection}\n")
            
            labeled_count += 1
    
    print(f"Auto-labeled {labeled_count} animal images")

if __name__ == "__main__":
    auto_label_animals()