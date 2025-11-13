"""
Diagnostic script to check COCO annotation format and predictions
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def check_coco_format(json_path, image_dir):
    """Check COCO format and visualize annotations"""
    
    print("="*70)
    print("COCO FORMAT DIAGNOSTIC")
    print("="*70)
    
    # Load annotations
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"\n📋 Dataset Structure:")
    print(f"   Images: {len(data['images'])}")
    print(f"   Annotations: {len(data['annotations'])}")
    print(f"   Categories: {len(data['categories'])}")
    
    # Check categories
    print(f"\n🏷️  Categories:")
    for cat in data['categories']:
        print(f"   ID {cat['id']}: {cat['name']}")
    
    # Check first image and its annotations
    first_img = data['images'][0]
    print(f"\n🖼️  First Image:")
    print(f"   ID: {first_img['id']}")
    print(f"   File: {first_img['file_name']}")
    print(f"   Size: {first_img['width']} x {first_img['height']}")
    
    # Get annotations for this image
    img_anns = [ann for ann in data['annotations'] if ann['image_id'] == first_img['id']]
    print(f"\n📦 Annotations for first image: {len(img_anns)}")
    
    if len(img_anns) > 0:
        print(f"\n   Sample annotation:")
        ann = img_anns[0]
        print(f"   Bbox: {ann['bbox']}")
        print(f"   Category: {ann['category_id']}")
        print(f"   Area: {ann.get('area', 'N/A')}")
        
        # Check bbox format
        x, y, w, h = ann['bbox']
        print(f"\n   Bbox details:")
        print(f"   - X: {x}, Y: {y}")
        print(f"   - Width: {w}, Height: {h}")
        print(f"   - X range: [{x}, {x+w}]")
        print(f"   - Y range: [{y}, {y+h}]")
        
        # Check if coordinates are normalized or absolute
        if x <= 1.0 and y <= 1.0 and w <= 1.0 and h <= 1.0:
            print(f"   ⚠️  WARNING: Coordinates appear to be NORMALIZED (0-1)")
            print(f"   Expected: Absolute pixel coordinates")
        else:
            print(f"   ✅ Coordinates appear to be absolute pixels")
        
        # Check if bbox is within image bounds
        img_w, img_h = first_img['width'], first_img['height']
        if x + w > img_w or y + h > img_h:
            print(f"   ⚠️  WARNING: Bbox extends beyond image bounds!")
            print(f"   Image size: {img_w}x{img_h}")
            print(f"   Bbox end: ({x+w}, {y+h})")
    
    # Visualize first image with annotations
    img_path = Path(image_dir) / first_img['file_name']
    if img_path.exists():
        print(f"\n📊 Creating visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Load image
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Original image
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        # Image with ground truth annotations
        axes[1].imshow(img_array)
        axes[1].set_title(f'Ground Truth Annotations ({len(img_anns)} boxes)', fontweight='bold')
        
        # Draw all annotations
        for ann in img_anns:
            x, y, w, h = ann['bbox']
            
            # Check if normalized and convert if needed
            if x <= 1.0 and y <= 1.0:
                # Normalized coordinates - convert to pixels
                x = x * first_img['width']
                y = y * first_img['height']
                w = w * first_img['width']
                h = h * first_img['height']
                print(f"   Converting normalized coords to pixels")
            
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            axes[1].add_patch(rect)
        
        axes[1].axis('off')
        
        plt.tight_layout()
        output_path = 'coco_diagnostic.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved to: {output_path}")
        plt.close()
    else:
        print(f"   ⚠️  Image not found: {img_path}")
    
    return data

def check_predictions(pred_path, gt_path, image_dir):
    """Compare predictions with ground truth"""
    
    print(f"\n" + "="*70)
    print("PREDICTION vs GROUND TRUTH COMPARISON")
    print("="*70)
    
    # Load predictions
    with open(pred_path, 'r') as f:
        predictions = json.load(f)
    
    # Load ground truth
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    
    print(f"\n📊 Prediction Statistics:")
    print(f"   Total predictions: {len(predictions)}")
    
    if len(predictions) > 0:
        # Get first image's predictions
        first_img_id = gt_data['images'][0]['id']
        img_preds = [p for p in predictions if p['image_id'] == first_img_id]
        img_gts = [a for a in gt_data['annotations'] if a['image_id'] == first_img_id]
        
        print(f"\n   First image (ID {first_img_id}):")
        print(f"   - Predictions: {len(img_preds)}")
        print(f"   - Ground truth: {len(img_gts)}")
        
        if len(img_preds) > 0 and len(img_gts) > 0:
            pred = img_preds[0]
            gt = img_gts[0]
            
            print(f"\n   Sample prediction bbox: {pred['bbox']}")
            print(f"   Sample GT bbox: {gt['bbox']}")
            
            # Calculate IoU
            def calculate_iou(box1, box2):
                """Calculate IoU between two boxes [x, y, w, h]"""
                x1, y1, w1, h1 = box1
                x2, y2, w2, h2 = box2
                
                # Convert to [x1, y1, x2, y2]
                box1_x2 = x1 + w1
                box1_y2 = y1 + h1
                box2_x2 = x2 + w2
                box2_y2 = y2 + h2
                
                # Intersection
                xi1 = max(x1, x2)
                yi1 = max(y1, y2)
                xi2 = min(box1_x2, box2_x2)
                yi2 = min(box1_y2, box2_y2)
                
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                
                # Union
                box1_area = w1 * h1
                box2_area = w2 * h2
                union_area = box1_area + box2_area - inter_area
                
                return inter_area / union_area if union_area > 0 else 0
            
            # Find best matching GT for first prediction
            best_iou = 0
            best_gt = None
            best_gt_idx = -1
            
            print(f"\n   Checking all {len(img_gts)} ground truth boxes against first prediction...")
            for idx, gt in enumerate(img_gts):
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
                    best_gt_idx = idx
            
            print(f"\n   First prediction bbox: {pred['bbox']}")
            print(f"   Best matching GT (index {best_gt_idx}): {best_gt['bbox']}")
            print(f"   Best IoU: {best_iou:.4f}")
            
            # Also check average IoU across all predictions
            total_iou = 0
            matched = 0
            for pred_item in img_preds:
                best_match_iou = 0
                for gt in img_gts:
                    iou = calculate_iou(pred_item['bbox'], gt['bbox'])
                    if iou > best_match_iou:
                        best_match_iou = iou
                if best_match_iou > 0.5:
                    matched += 1
                total_iou += best_match_iou
            
            avg_iou = total_iou / len(img_preds) if len(img_preds) > 0 else 0
            match_rate = matched / len(img_preds) if len(img_preds) > 0 else 0
            
            print(f"\n   Average IoU across all predictions: {avg_iou:.4f}")
            print(f"   Match rate (IoU > 0.5): {match_rate:.2%} ({matched}/{len(img_preds)})")
            
            if best_iou < 0.5:
                print(f"   ⚠️  WARNING: Best IoU < 0.5 indicates coordinate mismatch!")
                
                # Check if one is normalized and other is not
                pred_max = max(pred['bbox'])
                gt_max = max(best_gt['bbox'])
                
                if pred_max > 1.0 and gt_max <= 1.0:
                    print(f"\n   ❌ ISSUE FOUND: Predictions are in pixels, GT is normalized!")
                elif pred_max <= 1.0 and gt_max > 1.0:
                    print(f"\n   ❌ ISSUE FOUND: Predictions are normalized, GT is in pixels!")
                else:
                    print(f"\n   ❌ ISSUE: Coordinates are in different reference frames!")
            else:
                print(f"   ✅ IoU looks good!")
                
            if avg_iou < 0.3:
                print(f"\n   ❌ CRITICAL: Average IoU is very low!")
                print(f"   This explains why COCO mAP is near zero.")
                print(f"   Predictions and GT are in different coordinate spaces!")
        
        # Visualize predictions vs GT
        first_img = gt_data['images'][0]
        img_path = Path(image_dir) / first_img['file_name']
        
        if img_path.exists():
            print(f"\n📊 Creating prediction comparison...")
            
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Original
            axes[0].imshow(img_array)
            axes[0].set_title('Original Image', fontweight='bold')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(img_array)
            axes[1].set_title(f'Ground Truth ({len(img_gts)} boxes)', fontweight='bold')
            for gt in img_gts:
                x, y, w, h = gt['bbox']
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2, edgecolor='lime', facecolor='none'
                )
                axes[1].add_patch(rect)
            axes[1].axis('off')
            
            # Predictions
            axes[2].imshow(img_array)
            axes[2].set_title(f'Predictions ({len(img_preds)} boxes)', fontweight='bold')
            for pred in img_preds:
                x, y, w, h = pred['bbox']
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                axes[2].add_patch(rect)
            axes[2].axis('off')
            
            plt.tight_layout()
            output_path = 'prediction_comparison.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Saved to: {output_path}")
            plt.close()

def main():
    # Paths
    val_json = './yolo/prepared_data/coco/val.json'
    val_images = './yolo/prepared_data/coco/val'
    yolo_pred = './comparison_results/yolo_predictions.json'
    rcnn_pred = './comparison_results/r_cnn_predictions.json'
    
    # Check ground truth format
    print("\n🔍 Checking Ground Truth Annotations...")
    gt_data = check_coco_format(val_json, val_images)
    
    # Check predictions if they exist
    if Path(yolo_pred).exists():
        print("\n🔍 Checking YOLO Predictions...")
        check_predictions(yolo_pred, val_json, val_images)
    
    if Path(rcnn_pred).exists():
        print("\n🔍 Checking R-CNN Predictions...")
        check_predictions(rcnn_pred, val_json, val_images)
    
    print("\n" + "="*70)
    print("✅ DIAGNOSTIC COMPLETE")
    print("="*70)
    print("\nCheck the generated images:")
    print("  - coco_diagnostic.png - Ground truth visualization")
    print("  - prediction_comparison.png - Predictions vs GT")
    print("\nLook for coordinate mismatches between predictions and GT!")

if __name__ == "__main__":
    main()