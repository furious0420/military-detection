import os
import shutil
from pathlib import Path
import random

def update_label_class_ids(label_file, new_class_id):
    """Update class IDs in YOLO label files to match unified mapping"""
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    updated_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # Replace class ID with new unified class ID
            parts[0] = str(new_class_id)
            updated_lines.append(' '.join(parts) + '\n')
    
    return updated_lines

def create_unified_dataset():
    base_path = Path("b:/eoir/datasets/")
    output_path = Path("unified_dataset")
    
    # Check what folders actually exist
    if base_path.exists():
        existing_folders = [f.name for f in base_path.iterdir() if f.is_dir()]
        print(f"Found folders: {existing_folders}")
    else:
        print(f"Base path {base_path} does not exist!")
        return
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{output_path}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_path}/{split}/labels", exist_ok=True)
    
    # Map folder names to class IDs
    class_mapping = {
        'human': 0, 'humans': 0,
        'animal': 1, 'animals': 1,
        'vehicle': 2, 'vehicles': 2,
        'drone': 3, 'drones': 3, 'uav': 3,
        'weapon': 4, 'weapons': 4
    }
    
    all_files = []
    
    for folder in existing_folders:
        dataset_path = base_path / folder
        print(f"\nProcessing {folder}...")
        
        # Get class ID for this folder
        class_id = None
        folder_lower = folder.lower()
        for key, value in class_mapping.items():
            if key in folder_lower:
                class_id = value
                break
        
        if class_id is None:
            print(f"Warning: Could not map {folder} to a class")
            continue
        
        image_count = 0
        unlabeled_count = 0
        
        # Try standard YOLO structure: train/images, train/labels
        for split in ['train', 'val', 'test']:
            images_dir = dataset_path / split / "images"
            labels_dir = dataset_path / split / "labels"
            
            if images_dir.exists() and labels_dir.exists():
                for pattern in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                    for img_file in images_dir.glob(pattern):
                        label_file = labels_dir / f"{img_file.stem}.txt"
                        if label_file.exists():
                            all_files.append((img_file, label_file, class_id))
                            image_count += 1
                        else:
                            unlabeled_count += 1
        
        # Try inverted structure: images/train, labels/train
        if image_count == 0:
            for split in ['train', 'val', 'test']:
                images_dir = dataset_path / "images" / split
                labels_dir = dataset_path / "labels" / split
                
                if images_dir.exists() and labels_dir.exists():
                    for pattern in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                        for img_file in images_dir.glob(pattern):
                            label_file = labels_dir / f"{img_file.stem}.txt"
                            if label_file.exists():
                                all_files.append((img_file, label_file, class_id))
                                image_count += 1
                            else:
                                unlabeled_count += 1
        
        # Check for loose images (not in proper structure)
        if image_count == 0:
            for pattern in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                for img_file in dataset_path.rglob(pattern):
                    unlabeled_count += 1
        
        print(f"Found {image_count} image-label pairs in {folder}")
        if unlabeled_count > 0:
            print(f"  Warning: {unlabeled_count} unlabeled images found (skipped)")
    
    if not all_files:
        print("No valid image-label pairs found!")
        return
    
    # Shuffle and split
    random.shuffle(all_files)
    total = len(all_files)
    train_split = int(0.7 * total)
    val_split = int(0.9 * total)
    
    for i, (img_file, label_file, class_id) in enumerate(all_files):
        if i < train_split:
            split = 'train'
        elif i < val_split:
            split = 'val'
        else:
            split = 'test'
            
        # Copy files with new names to avoid conflicts
        new_name = f"{class_id}_{img_file.stem}{img_file.suffix}"
        new_img_path = f"{output_path}/{split}/images/{new_name}"
        new_label_path = f"{output_path}/{split}/labels/{new_name.replace(img_file.suffix, '.txt')}"
        
        # Copy image
        shutil.copy2(img_file, new_img_path)
        
        # Update and copy label with correct class ID
        updated_lines = update_label_class_ids(label_file, class_id)
        with open(new_label_path, 'w') as f:
            f.writelines(updated_lines)
    
    print(f"\nCreated unified dataset with {len(all_files)} samples")
    print(f"Train: {train_split}, Val: {val_split - train_split}, Test: {total - val_split}")
    
    # Create YOLOv8 dataset.yaml file
    yaml_content = f"""path: {output_path.absolute()}
train: {output_path.absolute()}/train/images
val: {output_path.absolute()}/val/images
test: {output_path.absolute()}/test/images

names:
  0: human
  1: animal
  2: vehicle
  3: drone
  4: weapon

nc: 5
"""
    
    with open("unified_dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    print(f"Created unified_dataset.yaml for YOLOv8 training")

if __name__ == "__main__":
    create_unified_dataset()




