# Check training data
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Load train JSON
with open('../yolo/prepared_data/coco/train.json', 'r') as f:
    train_data = json.load(f)

# Get first image
img_info = train_data['images'][0]
img_path = Path('../yolo/prepared_data/coco/train') / img_info['file_name']

# Load image
img = Image.open(img_path)

# Get annotations
anns = [ann for ann in train_data['annotations'] if ann['image_id'] == img_info['id']]

# Visualize
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.imshow(img)
ax.set_title(f"Training Data Ground Truth: {len(anns)} boxes")

for ann in anns[:50]:
    x, y, w, h = ann['bbox']
    rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                             edgecolor='lime', facecolor='none')
    ax.add_patch(rect)

plt.axis('off')
plt.tight_layout()
plt.savefig('./runs/rcnn/train_gt_check.png', dpi=150)
plt.show()

print(f"Check if training ground truth is also broken!")