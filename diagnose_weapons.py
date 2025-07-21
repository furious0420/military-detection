from pathlib import Path

def diagnose_weapons():
    base_path = Path("b:/eoir/datasets/weapons")
    print(f"Diagnosing weapons folder: {base_path}")
    print(f"Folder exists: {base_path.exists()}")
    
    if not base_path.exists():
        return
    
    print(f"\n=== FULL DIRECTORY TREE ===")
    for item in base_path.rglob("*"):
        if item.is_file():
            print(f"FILE: {item.relative_to(base_path)} (size: {item.stat().st_size} bytes)")
        elif item.is_dir():
            print(f"DIR:  {item.relative_to(base_path)}/")
    
    print(f"\n=== IMAGE FILES ===")
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    for ext in image_extensions:
        images.extend(list(base_path.rglob(f"*{ext}")))
        images.extend(list(base_path.rglob(f"*{ext.upper()}")))
    
    for img in images:
        print(f"IMAGE: {img.relative_to(base_path)}")
    
    print(f"\n=== LABEL FILES ===")
    labels = list(base_path.rglob("*.txt"))
    for lbl in labels:
        print(f"LABEL: {lbl.relative_to(base_path)} (size: {lbl.stat().st_size} bytes)")
        # Show first few lines of label file
        try:
            with open(lbl, 'r') as f:
                content = f.read().strip()
                if content:
                    print(f"  Content: {content[:100]}...")
                else:
                    print(f"  Content: EMPTY")
        except Exception as e:
            print(f"  Error reading: {e}")
    
    print(f"\n=== MATCHING PAIRS ===")
    for img in images:
        potential_label = img.with_suffix('.txt')
        if potential_label.exists():
            print(f"PAIR: {img.name} <-> {potential_label.name}")
        else:
            print(f"MISSING LABEL: {img.name} (looking for {potential_label.name})")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total images: {len(images)}")
    print(f"Total labels: {len(labels)}")
    print(f"Potential pairs: {sum(1 for img in images if img.with_suffix('.txt').exists())}")

if __name__ == "__main__":
    diagnose_weapons()