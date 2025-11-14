from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Load dataset
ds = load_dataset("backseollgi/parking_dataset", "carpk", streaming=False)
print("Dataset structure:", ds)

# Get first sample
if 'train' in ds:
    sample = ds['train'][0]
else:
    key = list(ds.keys())[0]
    sample = ds[key][0]

print("\nSample keys:", sample.keys())
print("Image type:", type(sample['image']))
print("Number of boxes:", len(sample['bboxes']))
print("First box:", sample['bboxes'][0])

# Visualize
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
img = sample['image']
if isinstance(img, str):
    img = Image.open(img)
ax.imshow(img)

for bbox in sample['bboxes'][:10]:  # First 10 boxes
    if len(bbox) == 4:
        # Try to auto-detect format
        x1, y1, x2, y2 = bbox
        if x2 < x1 or y2 < y1:  # Likely [x, y, w, h] format
            x, y, w, h = bbox
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        else:  # Likely [x1, y1, x2, y2] format
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

plt.savefig('test_sample.png')
print("\nSaved test_sample.png - check if boxes align correctly!")