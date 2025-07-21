from pathlib import Path

def fix_animal_class_ids():
    labels_dir = Path("b:/eoir/datasets/animals/train/labels")
    
    if not labels_dir.exists():
        print("Labels directory not found!")
        return
    
    fixed_count = 0
    
    for label_file in labels_dir.glob("*.txt"):
        # Read current content
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Fix class IDs (change 0 to 1)
        fixed_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == '0':
                parts[0] = '1'  # Change to animal class
                fixed_lines.append(' '.join(parts) + '\n')
            else:
                fixed_lines.append(line)
        
        # Write back
        with open(label_file, 'w') as f:
            f.writelines(fixed_lines)
        
        fixed_count += 1
    
    print(f"Fixed class IDs in {fixed_count} label files")

if __name__ == "__main__":
    fix_animal_class_ids()