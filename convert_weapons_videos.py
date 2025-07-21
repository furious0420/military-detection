import cv2
import json
import os
from pathlib import Path

def extract_frames_and_convert_labels():
    base_path = Path("b:/eoir/datasets/weapons")
    output_path = base_path / "train"
    
    # Create output directories
    os.makedirs(output_path / "images", exist_ok=True)
    os.makedirs(output_path / "labels", exist_ok=True)
    
    video_folders = [d for d in base_path.rglob("*") if d.is_dir() and "L1_I1" in d.name]
    
    total_frames = 0
    total_annotations = 0
    
    for folder in video_folders:
        video_file = folder / "video.mp4"
        label_file = folder / "label.json"
        
        if not video_file.exists() or not label_file.exists():
            continue
            
        print(f"Processing {folder.name}...")
        
        # Load JSON labels
        with open(label_file, 'r') as f:
            labels_data = json.load(f)
        
        # Create mapping from image_id to frame info
        frame_info = {frame['id']: frame for frame in labels_data['video']}
        
        # Group annotations by image_id (frame)
        frame_annotations = {}
        for ann in labels_data['annotations']:
            image_id = ann['image_id']
            if image_id not in frame_annotations:
                frame_annotations[image_id] = []
            frame_annotations[image_id].append(ann)
        
        # Open video
        cap = cv2.VideoCapture(str(video_file))
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Find corresponding frame info (assuming sequential order)
            frame_id = None
            for fid, finfo in frame_info.items():
                if frame_count == fid or abs(frame_count - fid) < 2:  # Allow small offset
                    frame_id = fid
                    break
            
            if frame_id is None:
                frame_count += 1
                continue
                
            # Save frame
            frame_name = f"{folder.name}_frame_{frame_count:04d}.jpg"
            frame_path = output_path / "images" / frame_name
            cv2.imwrite(str(frame_path), frame)
            
            # Convert labels for this frame
            label_path = output_path / "labels" / f"{folder.name}_frame_{frame_count:04d}.txt"
            
            h, w = frame.shape[:2]
            frame_labels = []
            
            if frame_id in frame_annotations:
                for ann in frame_annotations[frame_id]:
                    # Extract position (bounding box)
                    if 'position' in ann and ann['position']:
                        pos = ann['position']
                        
                        # Handle different position formats
                        if isinstance(pos, dict):
                            if 'x' in pos and 'y' in pos and 'width' in pos and 'height' in pos:
                                x, y, width, height = pos['x'], pos['y'], pos['width'], pos['height']
                            elif 'left' in pos and 'top' in pos and 'width' in pos and 'height' in pos:
                                x, y, width, height = pos['left'], pos['top'], pos['width'], pos['height']
                            else:
                                continue
                        elif isinstance(pos, list) and len(pos) >= 4:
                            x, y, width, height = pos[0], pos[1], pos[2], pos[3]
                        else:
                            continue
                        
                        # Convert to YOLO format (normalized center coordinates)
                        x_center = (x + width/2) / w
                        y_center = (y + height/2) / h
                        norm_width = width / w
                        norm_height = height / h
                        
                        # Ensure values are within [0,1]
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        norm_width = max(0, min(1, norm_width))
                        norm_height = max(0, min(1, norm_height))
                        
                        frame_labels.append(f"4 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
                        total_annotations += 1
            
            # Write labels (even if empty)
            with open(label_path, 'w') as f:
                for label in frame_labels:
                    f.write(f"{label}\n")
            
            frame_count += 1
            total_frames += 1
            
            # Limit frames per video to avoid too many
            if frame_count >= 50:  # Extract max 50 frames per video
                break
        
        cap.release()
        print(f"  Extracted {frame_count} frames with {len([f for f in frame_annotations.values() if f])} annotated frames")
    
    print(f"\nTotal frames extracted: {total_frames}")
    print(f"Total annotations: {total_annotations}")
    print(f"Conversion complete! Check {output_path}")

if __name__ == "__main__":
    extract_frames_and_convert_labels()
