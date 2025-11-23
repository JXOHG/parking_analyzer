import torch
import numpy as np
from pathlib import Path

# Load model
checkpoint = torch.load('./runs/rcnn/best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

# Get one validation sample
test_images, test_targets = next(iter(val_loader))

print("="*70)
print("Prediction Format Check")
print("="*70)

with torch.no_grad():
    test_images_gpu = [test_images[0].to(device)]
    outputs = model(test_images_gpu)

# Check prediction
pred_boxes = outputs[0]['boxes'].cpu().numpy()
pred_scores = outputs[0]['scores'].cpu().numpy()

print(f"\nPredictions made: {len(pred_boxes)}")
print(f"Max score: {pred_scores.max():.4f}")

# Check ground truth
gt_boxes = test_targets[0]['boxes'].cpu().numpy()
img_id = test_targets[0]['image_id'].item()
img_info = val_dataset.coco.loadImgs(img_id)[0]

print(f"\nGround truth boxes: {len(gt_boxes)}")
print(f"Image size: {img_info['width']}x{img_info['height']}")

# Check the actual annotations in COCO
ann_ids = val_dataset.coco.getAnnIds(imgIds=img_id)
coco_anns = val_dataset.coco.loadAnns(ann_ids)

print(f"\nCOCO annotations: {len(coco_anns)}")
print(f"\nFirst COCO annotation (from JSON):")
print(f"  bbox: {coco_anns[0]['bbox']} (format: [x, y, w, h])")

print(f"\nFirst ground truth box (loaded by dataset):")
print(f"  bbox: {gt_boxes[0]} (format: [x1, y1, x2, y2])")

print(f"\nFirst prediction:")
print(f"  bbox: {pred_boxes[0]} (format: [x1, y1, x2, y2])")

# Now convert prediction to COCO format (what evaluation does)
x1, y1, x2, y2 = pred_boxes[0]
coco_pred = [x1, y1, x2-x1, y2-y1]  # Convert to [x, y, w, h]
print(f"  COCO format: {coco_pred} (format: [x, y, w, h])")

# CRITICAL: Check if the conversion matches what evaluation script does
print("\n" + "="*70)
print("CRITICAL CHECK: Bbox Conversion in Evaluation")
print("="*70)

print("\nIn your evaluation code, predictions are converted like this:")
print("  x1, y1, x2, y2 = box")
print("  w = x2 - x1")
print("  h = y2 - y1")
print("  coco_bbox = [float(x1), float(y1), float(w), float(h)]")

print("\n⚠️  WAIT! This is WRONG!")
print("COCO format is: [x, y, width, height]")
print("Your code does:  [x1, y1, width, height]")
print("But x1, y1 are already top-left corner, so this should be correct...")

# Actually test IoU
def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

# Compute IoUs between predictions and ground truth
print("\n" + "="*70)
print("IoU Analysis")
print("="*70)

high_conf_mask = pred_scores > 0.5
high_conf_preds = pred_boxes[high_conf_mask]

print(f"\nHigh confidence predictions: {len(high_conf_preds)}")
print(f"Ground truth boxes: {len(gt_boxes)}")

# Find best IoU for each GT box
ious_matrix = np.zeros((len(gt_boxes), len(high_conf_preds)))

for i, gt_box in enumerate(gt_boxes):
    for j, pred_box in enumerate(high_conf_preds):
        ious_matrix[i, j] = compute_iou(gt_box, pred_box)

if ious_matrix.size > 0:
    print(f"\nIoU statistics:")
    print(f"  Max IoU: {ious_matrix.max():.4f}")
    print(f"  Mean IoU (best match per GT): {ious_matrix.max(axis=1).mean():.4f}")
    print(f"  GTs with IoU>0.5: {(ious_matrix.max(axis=1) > 0.5).sum()}/{len(gt_boxes)}")
    print(f"  GTs with IoU>0.75: {(ious_matrix.max(axis=1) > 0.75).sum()}/{len(gt_boxes)}")
    
    if ious_matrix.max() < 0.3:
        print("\n❌ PROBLEM FOUND: IoU is very low!")
        print("   Predictions don't overlap with ground truth at all!")
        print("   This suggests a coordinate system mismatch.")