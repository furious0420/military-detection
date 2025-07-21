from pathlib import Path

base_path = Path("b:/eoir/datasets/weapons")
print(f"Checking weapons dataset structure...")

# Check for different possible structures
structures = [
    "train/images and train/labels",
    "images/train and labels/train", 
    "loose files"
]

for structure in structures:
    if structure == "train/images and train/labels":
        img_dir = base_path / "train" / "images"
        lbl_dir = base_path / "train" / "labels"
    elif structure == "images/train and labels/train":
        img_dir = base_path / "images" / "train"
        lbl_dir = base_path / "labels" / "train"
    else:
        img_dir = base_path
        lbl_dir = base_path
    
    if img_dir.exists():
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        if lbl_dir.exists():
            labels = list(lbl_dir.glob("*.txt"))
            pairs = 0
            for img in images:
                if (lbl_dir / f"{img.stem}.txt").exists():
                    pairs += 1
            print(f"{structure}: {len(images)} images, {len(labels)} labels, {pairs} pairs")
        else:
            print(f"{structure}: {len(images)} images, no labels folder")