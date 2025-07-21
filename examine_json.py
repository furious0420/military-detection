import json
from pathlib import Path

def examine_json_structure():
    base_path = Path("b:/eoir/datasets/weapons")
    
    # Find first JSON file
    json_file = None
    for folder in base_path.rglob("*"):
        if folder.is_dir() and "L1_I1" in folder.name:
            potential_json = folder / "label.json"
            if potential_json.exists():
                json_file = potential_json
                break
    
    if not json_file:
        print("No JSON file found!")
        return
    
    print(f"Examining: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"JSON structure:")
    print(f"Keys: {list(data.keys())}")
    
    for key, value in data.items():
        if isinstance(value, list):
            print(f"{key}: list with {len(value)} items")
            if value:
                print(f"  First item: {type(value[0])}")
                if isinstance(value[0], dict):
                    print(f"  First item keys: {list(value[0].keys())}")
        elif isinstance(value, dict):
            print(f"{key}: dict with keys: {list(value.keys())}")
        else:
            print(f"{key}: {type(value)} = {value}")

if __name__ == "__main__":
    examine_json_structure()