import os
import json
import xml.etree.ElementTree as ET
import cv2
from pathlib import Path
import shutil
import yaml

class DatasetConverter:
    def __init__(self, base_path="b:/eoir/datasets/"):
        self.base_path = Path(base_path)
        self.output_path = Path("unified_dataset")
        self.class_mapping = {
            'human': 0, 'person': 0, 'people': 0,
            'animal': 1, 'dog': 1, 'cat': 1, 'bird': 1,
            'vehicle': 2, 'car': 2, 'truck': 2, 'bus': 2,
            'drone': 3, 'uav': 3, 'quadcopter': 3,
            'weapon': 4, 'gun': 4, 'knife': 4, 'rifle': 4
        }
        
    def convert_coco_to_yolo(self, coco_file, img_dir, output_dir):
        """Convert COCO format to YOLO"""
        with open(coco_file, 'r') as f:
            data = json.load(f)
        
        # Create output directories
        os.makedirs(f"{output_dir}/labels", exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        
        for annotation in data['annotations']:
            img_info = next(img for img in data['images'] if img['id'] == annotation['image_id'])
            
            # Convert bbox to YOLO format
            x, y, w, h = annotation['bbox']
            img_w, img_h = img_info['width'], img_info['height']
            
            x_center = (x + w/2) / img_w
            y_center = (y + h/2) / img_h
            width = w / img_w
            height = h / img_h
            
            # Map class
            class_name = data['categories'][annotation['category_id']]['name'].lower()
            class_id = self.class_mapping.get(class_name, 0)
            
            # Write YOLO label
            label_file = f"{output_dir}/labels/{Path(img_info['file_name']).stem}.txt"
            with open(label_file, 'a') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def convert_pascal_to_yolo(self, xml_dir, img_dir, output_dir):
        """Convert Pascal VOC format to YOLO"""
        os.makedirs(f"{output_dir}/labels", exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        
        for xml_file in Path(xml_dir).glob("*.xml"):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            img_w = int(root.find('size/width').text)
            img_h = int(root.find('size/height').text)
            
            label_file = f"{output_dir}/labels/{xml_file.stem}.txt"
            
            with open(label_file, 'w') as f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text.lower()
                    class_id = self.class_mapping.get(class_name, 0)
                    
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    
                    x_center = ((xmin + xmax) / 2) / img_w
                    y_center = ((ymin + ymax) / 2) / img_h
                    width = (xmax - xmin) / img_w
                    height = (ymax - ymin) / img_h
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def process_all_datasets(self):
        """Process all 5 datasets"""
        datasets = ['humans', 'animals', 'vehicles', 'drones', 'weapons']
        
        for dataset in datasets:
            dataset_path = self.base_path / dataset
            print(f"Processing {dataset}...")
            
            # Check format and convert accordingly
            if (dataset_path / "annotations.json").exists():
                self.convert_coco_to_yolo(
                    dataset_path / "annotations.json",
                    dataset_path / "images",
                    self.output_path / "train"
                )
            elif list(dataset_path.glob("**/*.xml")):
                xml_dir = next(dataset_path.rglob("**/annotations"))
                img_dir = next(dataset_path.rglob("**/images"))
                self.convert_pascal_to_yolo(xml_dir, img_dir, self.output_path / "train")
            else:
                # Assume already YOLO format
                self.copy_yolo_format(dataset_path, self.output_path / "train")

    def copy_yolo_format(self, dataset_path, output_dir):
        """Copy existing YOLO format dataset"""
        os.makedirs(f"{output_dir}/labels", exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        
        # Find labels and images directories
        labels_dir = None
        images_dir = None
        
        for subdir in dataset_path.rglob("*"):
            if subdir.is_dir() and "label" in subdir.name.lower():
                labels_dir = subdir
            elif subdir.is_dir() and "image" in subdir.name.lower():
                images_dir = subdir
        
        # Copy labels
        if labels_dir and labels_dir.exists():
            for label_file in labels_dir.glob("*.txt"):
                shutil.copy2(label_file, f"{output_dir}/labels/")
        
        # Copy images
        if images_dir and images_dir.exists():
            for img_file in images_dir.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    shutil.copy2(img_file, f"{output_dir}/images/")
        
        print(f"Copied YOLO format from {dataset_path}")

    def split_dataset(self):
        """Split unified dataset into train/val/test"""
        import random
        
        # Get all image files
        train_images = list((self.output_path / "train" / "images").glob("*"))
        random.shuffle(train_images)
        
        # Split ratios: 70% train, 20% val, 10% test
        total = len(train_images)
        train_split = int(0.7 * total)
        val_split = int(0.9 * total)
        
        # Create directories
        for split in ['val', 'test']:
            os.makedirs(f"{self.output_path}/{split}/images", exist_ok=True)
            os.makedirs(f"{self.output_path}/{split}/labels", exist_ok=True)
        
        # Move files
        for i, img_file in enumerate(train_images):
            label_file = self.output_path / "train" / "labels" / f"{img_file.stem}.txt"
            
            if i >= val_split:  # test set
                dest_dir = "test"
            elif i >= train_split:  # val set
                dest_dir = "val"
            else:
                continue  # keep in train
            
            # Move image and label
            shutil.move(str(img_file), f"{self.output_path}/{dest_dir}/images/")
            if label_file.exists():
                shutil.move(str(label_file), f"{self.output_path}/{dest_dir}/labels/")

if __name__ == "__main__":
    converter = DatasetConverter()
    converter.process_all_datasets()
    converter.split_dataset()
    print("Dataset conversion and splitting completed!")


