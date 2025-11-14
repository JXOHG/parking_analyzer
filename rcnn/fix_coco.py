"""
Quick fix to add missing 'info' field to COCO JSON files
"""
import json
import datetime
from pathlib import Path

def fix_coco_json(json_path):
    """Add missing 'info' and 'licenses' fields to COCO JSON"""
    print(f"🔧 Fixing {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Add 'info' field if missing
    if 'info' not in data:
        data['info'] = {
            'description': 'CARPK Parking Dataset',
            'url': 'https://huggingface.co/datasets/backseollgi/parking_dataset',
            'version': '1.0',
            'year': datetime.datetime.now().year,
            'contributor': 'CARPK Dataset',
            'date_created': datetime.datetime.now().strftime('%Y/%m/%d')
        }
    
    # Add 'licenses' field if missing
    if 'licenses' not in data:
        data['licenses'] = [{
            'id': 1,
            'name': 'Unknown',
            'url': ''
        }]
    
    # Ensure other required fields exist
    if 'images' not in data:
        data['images'] = []
    if 'annotations' not in data:
        data['annotations'] = []
    if 'categories' not in data:
        data['categories'] = [{'id': 1, 'name': 'car', 'supercategory': 'vehicle'}]
    
    # Save back
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Fixed {json_path}")
    print(f"   - Images: {len(data['images'])}")
    print(f"   - Annotations: {len(data['annotations'])}")
    print(f"   - Categories: {len(data['categories'])}")

if __name__ == '__main__':
    data_dir = Path('./data')
    
    for split in ['train', 'val', 'test']:
        json_file = data_dir / f'{split}.json'
        if json_file.exists():
            fix_coco_json(json_file)
        else:
            print(f"⚠️  {json_file} not found, skipping")
    
    print("\n✅ All COCO JSON files fixed!")
    print("   You can now resume training")