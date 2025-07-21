from pathlib import Path

def check_animals_dataset():
    base_path = Path("b:/eoir/datasets/animals")
    
    if not base_path.exists():
        print(f"Animals folder not found at {base_path}")
        return
    
    print(f"Checking animals dataset: {base_path}")
    
    # Check if train structure exists (from auto-labeling)
    train_path = base_path / "train"
    if train_path.exists():
        print(f"\n=== TRAIN STRUCTURE (AUTO-LABELED) ===")
        img_dir = train_path / "images"
        lbl_dir = train_path / "labels"
        
        if img_dir.exists() and lbl_dir.exists():
            images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            labels = list(lbl_dir.glob("*.txt"))
            
            # Count matching pairs
            pairs = 0
            for img in images:
                label_file = lbl_dir / f"{img.stem}.txt"
                if label_file.exists():
                    pairs += 1
            
            print(f"Images: {len(images)}")
            print(f"Labels: {len(labels)}")
            print(f"Matched pairs: {pairs}")
            
            # Sample a few labels to check format
            if labels:
                print(f"\n=== SAMPLE LABELS ===")
                for i, label_file in enumerate(labels[:3]):
                    print(f"{label_file.name}:")
                    with open(label_file, 'r') as f:
                        content = f.read().strip()
                        if content:
                            print(f"  {content}")
                        else:
                            print("  (empty)")
                    if i >= 2:
                        break
        else:
            print("No train/images or train/labels folders found")
    
    # Check original loose files
    loose_images = list(base_path.glob("*.jpg")) + list(base_path.glob("*.png"))
    print(f"\n=== ORIGINAL LOOSE FILES ===")
    print(f"Loose images: {len(loose_images)}")

if __name__ == "__main__":
    check_animals_dataset()
