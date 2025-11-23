"""
Faster R-CNN Training Script - Production Version with Comprehensive History Tracking
Training script with robust error handling, validation, and external metric saving
FIXED: Category ID mismatch causing zero AP scores
FIXED: Missing category info in COCO dataset causing evaluation to fail
ADDED: Complete training history tracking and visualization
"""

import os
import sys
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import warnings
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import copy 
warnings.filterwarnings('ignore')

# Check PyTorch version for compatibility
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
print(f"PyTorch version: {torch.__version__}")

class COCODataset(torch.utils.data.Dataset):
    """
    Custom dataset for loading COCO format annotations.
    Handles transforms and validates all data during initialization.
    """
    def __init__(self, json_file, img_dir, transforms=None):
        from pycocotools.coco import COCO
        
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        
        # Load COCO annotations
        print(f"Loading annotations from {json_file}...")
        try:
            # Load and fix the COCO JSON if needed
            with open(json_file, 'r') as f:
                coco_data = json.load(f)
            
            # FIXED: Ensure categories field exists and is properly formatted
            if 'categories' not in coco_data or len(coco_data['categories']) == 0:
                print("   ⚠️  Warning: Missing or empty 'categories' field in COCO JSON")
                print("   ✅ Adding default category: car (id=1)")
                coco_data['categories'] = [
                    {
                        'id': 1,
                        'name': 'car',
                        'supercategory': 'vehicle'
                    }
                ]
                # Save the fixed version
                fixed_json = str(json_file).replace('.json', '_fixed.json')
                with open(fixed_json, 'w') as f:
                    json.dump(coco_data, f)
                print(f"   ✅ Saved fixed JSON to: {fixed_json}")
                json_file = fixed_json
            
            # FIXED: Ensure info field exists
            if 'info' not in coco_data:
                print("   ⚠️  Warning: Missing 'info' field in COCO JSON")
                coco_data['info'] = {
                    'description': 'Car Detection Dataset',
                    'version': '1.0',
                    'year': 2024,
                    'contributor': 'Auto-generated',
                    'date_created': datetime.now().isoformat()
                }
            
            # FIXED: Ensure licenses field exists
            if 'licenses' not in coco_data:
                coco_data['licenses'] = []
            
            self.coco = COCO()
            self.coco.dataset = coco_data
            self.coco.createIndex()
            
        except Exception as e:
            print(f"Error: Failed to load COCO file: {e}")
            raise
        
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # Validate dataset
        self._validate_dataset()
        
        print(f"Loaded {len(self.ids)} images from {json_file}")
        
        # Print category info
        print(f"   Categories: {self.coco.loadCats(self.coco.getCatIds())}")
    
    def _validate_dataset(self):
        """Validate dataset integrity"""
        total_boxes = 0
        images_without_boxes = 0
        images_with_invalid_boxes = 0
        
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            if len(anns) == 0:
                images_without_boxes += 1
                continue
            
            # Check for invalid boxes
            valid_boxes = 0
            for ann in anns:
                x, y, w, h = ann['bbox']
                if w > 0 and h > 0:
                    valid_boxes += 1
            
            if valid_boxes == 0:
                images_with_invalid_boxes += 1
            
            total_boxes += valid_boxes
        
        if images_without_boxes > 0:
            print(f"   Warning: {images_without_boxes} images have no annotations")
        
        if images_with_invalid_boxes > 0:
            print(f"   Warning: {images_with_invalid_boxes} images have only invalid boxes")
        
        valid_images = len(self.ids) - images_without_boxes - images_with_invalid_boxes
        
        print(f"   Valid images: {valid_images}")
        print(f"   Total annotations: {total_boxes}")
        print(f"   Avg boxes/image: {total_boxes/len(self.ids):.2f}")
        
        if valid_images == 0:
            raise ValueError("No valid images found in dataset!")
    
    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_anns = self.coco.loadAnns(ann_ids)
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / img_info['file_name']
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            # Return a dummy sample
            img = Image.new('RGB', (640, 480))
            coco_anns = []
        
        # Parse annotations
        boxes, labels, areas = [], [], []
        for ann in coco_anns:
            x, y, w, h = ann['bbox']
            
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
            
            # COCO bbox is [x, y, width, height]
            # PyTorch needs [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            
            # FIXED: Faster R-CNN expects labels starting from 1 (0 is background)
            # Since category_id in dataset is already 1, we can use it directly
            labels.append(ann['category_id'])
            areas.append(ann['area'])
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
        else:
            # Empty annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            areas = torch.zeros(0, dtype=torch.float32)
        
        image_id = torch.tensor([img_id])
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64)
        }
        
        # Apply transforms
        if self.transforms:
            img = self.transforms(img)
        
        return img, target
    
    def __len__(self):
        return len(self.ids)


def get_transform(train=False):
    """
    Create image transforms for the model.
    Simple transforms that work with detection models.
    Only normalizes the image - augmentations handled by model.
    """
    transforms = []
    
    # Convert PIL to Tensor
    transforms.append(torchvision.transforms.ToTensor())
    
    # Normalize using ImageNet stats
    transforms.append(torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
    
    return torchvision.transforms.Compose(transforms)


def get_model(num_classes, anchor_sizes=None, pretrained=True):
    """
    Create Faster R-CNN model with custom anchors
    
    Args:
        num_classes: Number of classes (including background)
        anchor_sizes: Tuple of anchor sizes, e.g., ((8,), (16,), (32,), (64,), (128,))
        pretrained: Use pretrained backbone
    """
    print("\n" + "="*70)
    print("Creating Model")
    print("="*70)
    
    # Load pretrained ResNet-50 FPN backbone
    if pretrained:
        try:
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
            print("Loaded pretrained weights (FasterRCNN_ResNet50_FPN_V2)")
        except AttributeError:
            # Fallback for older torchvision versions
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
            print("Loaded pretrained weights (FasterRCNN_ResNet50_FPN)")
    else:
        try:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
        except AttributeError:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        print("Initialized model from scratch (NOT recommended)")
    
    # Custom anchor generator if provided
    if anchor_sizes is not None:
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        model.rpn.anchor_generator = anchor_generator
        print(f"Custom RPN anchors: {anchor_sizes}")
        print(f"  Aspect ratios: {aspect_ratios[0]}")
    else:
        print(f"Using default anchors")
    
    # Replace box predictor head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print(f"Replaced box predictor for {num_classes} classes")
    
    return model


def collate_fn(batch):
    """
    Custom collate function for batching
    Handles variable-sized images and annotations
    """
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None):
    """
    Train for one epoch with proper error handling and detailed loss tracking
    """
    model.train()
    
    total_loss = 0.0
    loss_dict_cumulative = {}
    num_batches = 0
    num_skipped = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]")
    
    for images, targets in pbar:
        try:
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Filter out images with no boxes to prevent training errors
            valid_idx = [i for i, t in enumerate(targets) if len(t['boxes']) > 0]
            if len(valid_idx) == 0:
                num_skipped += 1
                continue
            
            images = [images[i] for i in valid_idx]
            targets = [targets[i] for i in valid_idx]
            
            # Forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                
                # Check for NaN
                if torch.isnan(losses) or torch.isinf(losses):
                    print(f"\nWarning: NaN/Inf loss detected, skipping batch")
                    num_skipped += 1
                    continue
                
                # Backward pass with gradient scaling
                optimizer.zero_grad()
                scaler.scale(losses).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Check for NaN
                if torch.isnan(losses) or torch.isinf(losses):
                    print(f"\nWarning: NaN/Inf loss detected, skipping batch")
                    num_skipped += 1
                    continue
                
                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                optimizer.step()
            
            # Accumulate losses (detach from graph)
            total_loss += losses.item()
            for k, v in loss_dict.items():
                if k not in loss_dict_cumulative:
                    loss_dict_cumulative[k] = 0.0
                loss_dict_cumulative[k] += v.item()
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses.item():.4f}",
                'avg': f"{total_loss/num_batches:.4f}",
                'skip': num_skipped
            })
            
        except RuntimeError as e:
            print(f"\nWarning: Runtime error in batch: {e}")
            num_skipped += 1
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    # Calculate average for each loss component
    avg_loss_dict = {}
    if num_batches > 0:
        print(f"\n  Loss breakdown:")
        for k, v in loss_dict_cumulative.items():
            avg_val = v / num_batches
            avg_loss_dict[k] = avg_val
            print(f"    {k}: {avg_val:.4f}")
    
    if num_skipped > 0:
        print(f"  Warning: Skipped {num_skipped} batches due to errors")
    
    return avg_loss, avg_loss_dict


@torch.no_grad()
def evaluate(model, data_loader, device, coco_gt):
    """
    Evaluate using official COCO metrics with error handling
    FIXED: Proper category ID mapping to prevent zero AP scores
    FIXED: Ensure COCO ground truth has proper structure with categories
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    
    model.eval()
    
    # FIXED: Ensure the ground truth COCO object has proper structure
    if not hasattr(coco_gt, 'dataset') or coco_gt.dataset is None:
        print("⚠️  ERROR: COCO ground truth dataset is not properly initialized")
        return 0.0, 0.0, 0.0, None
    
    # FIXED: Verify categories exist
    if 'categories' not in coco_gt.dataset or len(coco_gt.dataset['categories']) == 0:
        print("⚠️  ERROR: No categories found in ground truth!")
        print("   Adding default category...")
        coco_gt.dataset['categories'] = [
            {
                'id': 1,
                'name': 'car',
                'supercategory': 'vehicle'
            }
        ]
        coco_gt.createIndex()
    
    print(f"\n📋 Ground Truth Info:")
    print(f"   Images: {len(coco_gt.dataset.get('images', []))}")
    print(f"   Annotations: {len(coco_gt.dataset.get('annotations', []))}")
    print(f"   Categories: {coco_gt.dataset.get('categories', [])}")
    
    coco_results = []
    num_errors = 0
    num_predictions = 0
    num_background_filtered = 0
    
    pbar = tqdm(data_loader, desc="Evaluating")
    
    for images, targets in pbar:
        try:
            images = [img.to(device) for img in images]
            
            outputs = model(images)
            
            # Convert to COCO format
            for target, output in zip(targets, outputs):
                image_id = target['image_id'].item()
                
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                # Filter by score threshold
                score_threshold = 0.05  # Low threshold for evaluation
                mask = scores > score_threshold
                
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]
                
                for box, score, label in zip(boxes, scores, labels):
                    # FIXED: Skip background predictions (label 0)
                    # Faster R-CNN outputs: 0=background, 1=car, etc.
                    if label == 0:
                        num_background_filtered += 1
                        continue
                    
                    # Convert [x1, y1, x2, y2] to [x, y, w, h]
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Skip invalid predictions
                    if w <= 0 or h <= 0:
                        continue
                    
                    # FIXED: Always use category_id=1 for car
                    # This matches the category_id in the ground truth annotations
                    coco_results.append({
                        'image_id': int(image_id),
                        'category_id': 1,  # Always 1 for car (matches ground truth)
                        'bbox': [float(x1), float(y1), float(w), float(h)],
                        'score': float(score)
                    })
                    num_predictions += 1
        
        except Exception as e:
            num_errors += 1
            pbar.set_postfix({'errors': num_errors})
            continue
    
    # Debug information
    print(f"\n📊 Evaluation Summary:")
    print(f"   Total predictions: {num_predictions}")
    print(f"   Background filtered: {num_background_filtered}")
    print(f"   Errors: {num_errors}")
    
    # Evaluate with COCO API
    if len(coco_results) == 0:
        print("\n⚠️  WARNING: No predictions made!")
        print("   Possible reasons:")
        print("     - Score threshold too high")
        print("     - Model not learning (check training loss)")
        print("     - All predictions are background")
        return 0.0, 0.0, 0.0, None
    
    # Additional debug info
    print(f"   Sample prediction: {coco_results[0]}")
    category_ids_pred = set(r['category_id'] for r in coco_results)
    print(f"   Category IDs in predictions: {category_ids_pred}")
    
    # Get ground truth category IDs for comparison
    gt_category_ids = set(cat['id'] for cat in coco_gt.dataset['categories'])
    print(f"   Category IDs in ground truth: {gt_category_ids}")
    
    if category_ids_pred != gt_category_ids:
        print(f"   ⚠️  WARNING: Category ID mismatch!")
        print(f"      Predictions: {category_ids_pred}")
        print(f"      Ground truth: {gt_category_ids}")
    
    try:
        # Create detection results COCO object
        coco_dt = coco_gt.loadRes(coco_results)
        
        # Run COCO evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract ALL metrics from COCO evaluation
        # stats[0:12] contains all standard COCO metrics
        metrics = {
            'map_50_95': float(coco_eval.stats[0]),  # AP @ IoU=0.50:0.95
            'ap50': float(coco_eval.stats[1]),        # AP @ IoU=0.50
            'ap75': float(coco_eval.stats[2]),        # AP @ IoU=0.75
            'ap_small': float(coco_eval.stats[3]),    # AP for small objects
            'ap_medium': float(coco_eval.stats[4]),   # AP for medium objects
            'ap_large': float(coco_eval.stats[5]),    # AP for large objects
            'ar_1': float(coco_eval.stats[6]),        # AR given 1 detection
            'ar_10': float(coco_eval.stats[7]),       # AR given 10 detections
            'ar_100': float(coco_eval.stats[8]),      # AR given 100 detections
            'ar_small': float(coco_eval.stats[9]),    # AR for small objects
            'ar_medium': float(coco_eval.stats[10]),  # AR for medium objects
            'ar_large': float(coco_eval.stats[11]),   # AR for large objects
        }
        
        return metrics['map_50_95'], metrics['ap50'], metrics['ap75'], metrics
        
    except Exception as e:
        print(f"\n⚠️  WARNING: COCO evaluation failed: {e}")
        print(f"    Generated {len(coco_results)} predictions")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, 0.0, None


def plot_training_curves(history, output_dir):
    """
    Generate comprehensive training visualization plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # 1. Training Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    # 2. Validation mAP
    ax = axes[0, 1]
    ax.plot(epochs, history['val_map'], 'g-o', linewidth=2, markersize=4, label='mAP@0.5:0.95')
    ax.plot(epochs, history['val_ap50'], 'r-s', linewidth=2, markersize=4, label='AP@0.50')
    ax.plot(epochs, history['val_ap75'], 'm-^', linewidth=2, markersize=4, label='AP@0.75')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AP Score')
    ax.set_title('Validation AP Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Learning Rate
    ax = axes[0, 2]
    ax.plot(epochs, history['lr'], 'orange', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 4. Loss Components (if available)
    ax = axes[1, 0]
    if 'loss_classifier' in history and len(history['loss_classifier']) > 0:
        ax.plot(epochs, history['loss_classifier'], label='Classifier', linewidth=2)
        ax.plot(epochs, history['loss_box_reg'], label='Box Reg', linewidth=2)
        ax.plot(epochs, history['loss_objectness'], label='Objectness', linewidth=2)
        ax.plot(epochs, history['loss_rpn_box_reg'], label='RPN Box Reg', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Loss components not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Loss Components')
    
    # 5. AP by Object Size (if available)
    ax = axes[1, 1]
    if 'val_ap_small' in history and len(history['val_ap_small']) > 0:
        ax.plot(epochs, history['val_ap_small'], label='Small', linewidth=2, marker='o')
        ax.plot(epochs, history['val_ap_medium'], label='Medium', linewidth=2, marker='s')
        ax.plot(epochs, history['val_ap_large'], label='Large', linewidth=2, marker='^')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AP Score')
        ax.set_title('AP by Object Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Size-specific AP not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('AP by Object Size')
    
    # 6. Average Recall (if available)
    ax = axes[1, 2]
    if 'val_ar_100' in history and len(history['val_ar_100']) > 0:
        ax.plot(epochs, history['val_ar_1'], label='AR@1', linewidth=2, marker='o')
        ax.plot(epochs, history['val_ar_10'], label='AR@10', linewidth=2, marker='s')
        ax.plot(epochs, history['val_ar_100'], label='AR@100', linewidth=2, marker='^')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AR Score')
        ax.set_title('Average Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'AR metrics not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Average Recall')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved to {output_dir / 'training_curves.png'}")
    
    # Create individual high-resolution plots
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # AP plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_map'], 'g-o', linewidth=2, markersize=6, label='mAP@0.5:0.95')
    plt.plot(epochs, history['val_ap50'], 'r-s', linewidth=2, markersize=6, label='AP@0.50')
    plt.plot(epochs, history['val_ap75'], 'm-^', linewidth=2, markersize=6, label='AP@0.75')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('AP Score', fontsize=12)
    plt.title('Validation AP Metrics', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'ap_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Individual plots saved: loss_curve.png, ap_curve.png")


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, output_dir, filename):
    """Save checkpoint with all necessary information"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'torch_version': torch.__version__,
        'torchvision_version': torchvision.__version__,
    }
    
    save_path = output_dir / filename
    torch.save(checkpoint, save_path)
    return save_path


def save_metrics_csv(history, output_dir):
    """
    Save training history as CSV for easy analysis in spreadsheet software
    """
    output_dir = Path(output_dir)
    
    # Create DataFrame from history
    df = pd.DataFrame(history)
    df.insert(0, 'epoch', range(1, len(df) + 1))
    
    # Save to CSV
    csv_path = output_dir / 'training_metrics.csv'
    df.to_csv(csv_path, index=False, float_format='%.6f')
    
    print(f"✅ Metrics saved to CSV: {csv_path}")
    
    # Also save a summary statistics file
    summary_path = output_dir / 'training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Epochs: {len(df)}\n\n")
        
        f.write("BEST METRICS:\n")
        f.write("-"*70 + "\n")
        
        # Find best epoch for each metric
        for col in df.columns:
            if col == 'epoch':
                continue
            
            if 'loss' in col:
                best_idx = df[col].idxmin()
                best_val = df[col].min()
                direction = "min"
            else:
                best_idx = df[col].idxmax()
                best_val = df[col].max()
                direction = "max"
            
            best_epoch = df.loc[best_idx, 'epoch']
            f.write(f"{col:25s}: {best_val:.6f} (epoch {best_epoch}, {direction})\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("FINAL EPOCH METRICS:\n")
        f.write("-"*70 + "\n")
        
        final_row = df.iloc[-1]
        for col in df.columns:
            if col != 'epoch':
                f.write(f"{col:25s}: {final_row[col]:.6f}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✅ Training summary saved: {summary_path}")


def main(args):
    print("="*70)
    print("Faster R-CNN Training - Production Version with Full History Tracking")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for organization
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    checkpoints_dir = output_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Save arguments and training metadata
    metadata = {
        'args': vars(args),
        'start_time': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        'torchvision_version': torchvision.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': str(device),
    }
    
    if torch.cuda.is_available():
        metadata['gpu_name'] = torch.cuda.get_device_name(0)
        metadata['cuda_version'] = torch.version.cuda
    
    with open(output_dir / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Load datasets
    print("\nLoading Datasets")
    print("="*70)
    
    data_path = Path(args.data_dir)
    
    # Verify data exists
    if not (data_path / 'train.json').exists():
        print(f"Error: Training data not found at {data_path / 'train.json'}")
        print("   Please run prepare_dataset.py first!")
        return
    
    if not (data_path / 'val.json').exists():
        print(f"Error: Validation data not found at {data_path / 'val.json'}")
        print("   Please run prepare_dataset.py first!")
        return
    
    try:
        train_dataset = COCODataset(
            json_file=data_path / 'train.json',
            img_dir=data_path / 'train',
            transforms=get_transform(train=True)
        )
        
        val_dataset = COCODataset(
            json_file=data_path / 'val.json',
            img_dir=data_path / 'val',
            transforms=get_transform(train=False)
        )
    except Exception as e:
        print(f"Error: Failed to load datasets: {e}")
        return
    
    # Data loaders with persistent_workers compatibility check
    use_persistent_workers = args.workers > 0 and TORCH_VERSION >= (1, 7)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=use_persistent_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=use_persistent_workers
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Workers: {args.workers}")
    print(f"  Persistent workers: {use_persistent_workers}")
    print(f"  Batch size: {args.batch_size}")
    
    # Create model
    num_classes = 2  # Background + car (Faster R-CNN handles background automatically)
    
    # Parse anchor sizes if provided
    anchor_sizes = None
    if args.anchor_sizes:
        try:
            anchor_sizes = tuple(tuple([int(s)]) for s in args.anchor_sizes.split(','))
            print(f"\nUsing custom anchor sizes: {anchor_sizes}")
        except Exception as e:
            print(f"Warning: Failed to parse anchor sizes: {e}")
            print(f"   Using default anchors instead")
    
    model = get_model(
        num_classes=num_classes,
        anchor_sizes=anchor_sizes,
        pretrained=not args.no_pretrained
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
        print(f"\nUsing SGD optimizer")
    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        print(f"\nUsing AdamW optimizer")
    
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    
    # Learning rate scheduler
    if args.scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
        print(f"Using StepLR scheduler")
        print(f"  Step size: {args.lr_step_size}")
        print(f"  Gamma: {args.lr_gamma}")
    elif args.scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
        print(f"Using CosineAnnealingLR scheduler")
    else:  # multistep
        milestones = [int(args.epochs * 0.6), int(args.epochs * 0.8)]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.1
        )
        print(f"Using MultiStepLR scheduler")
        print(f"  Milestones: {milestones}")
    
    # Mixed precision training with proper compatibility check
    scaler = None
    if args.amp and torch.cuda.is_available():
        if hasattr(torch.cuda.amp, 'GradScaler'):
            scaler = torch.cuda.amp.GradScaler()
            print("Using mixed precision training (AMP)")
        else:
            print("Warning: AMP not available in this PyTorch version")
    
    # Training loop
    print("\n" + "="*70)
    print("Training")
    print("="*70)
    
    best_map = 0.0
    best_ap50 = 0.0
    best_ap75 = 0.0
    epochs_no_improve = 0
    
    # Initialize comprehensive history tracking
    history = {
        'train_loss': [],
        'val_map': [],
        'val_ap50': [],
        'val_ap75': [],
        'lr': [],
        # Loss components
        'loss_classifier': [],
        'loss_box_reg': [],
        'loss_objectness': [],
        'loss_rpn_box_reg': [],
        # Additional COCO metrics
        'val_ap_small': [],
        'val_ap_medium': [],
        'val_ap_large': [],
        'val_ar_1': [],
        'val_ar_10': [],
        'val_ar_100': [],
        'val_ar_small': [],
        'val_ar_medium': [],
        'val_ar_large': [],
    }
    
    try:
        for epoch in range(1, args.epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*70}")
            
            # Train
            train_loss, loss_components = train_one_epoch(
                model, optimizer, train_loader, device, epoch, scaler
            )
            
            # Update learning rate
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Evaluate
            print(f"\nEvaluating Epoch {epoch}")
            val_map, ap50, ap75, detailed_metrics = evaluate(
                model, val_loader, device, val_dataset.coco
            )
            
            # Save to history - basic metrics
            history['train_loss'].append(float(train_loss))
            history['val_map'].append(float(val_map))
            history['val_ap50'].append(float(ap50))
            history['val_ap75'].append(float(ap75))
            history['lr'].append(float(current_lr))
            
            # Save loss components
            for key in ['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']:
                history[key].append(float(loss_components.get(key, 0.0)))
            
            # Save detailed COCO metrics if available
            if detailed_metrics:
                for key in ['ap_small', 'ap_medium', 'ap_large', 
                           'ar_1', 'ar_10', 'ar_100',
                           'ar_small', 'ar_medium', 'ar_large']:
                    history[f'val_{key}'].append(detailed_metrics.get(key, 0.0))
            else:
                # Append zeros if evaluation failed
                for key in ['ap_small', 'ap_medium', 'ap_large', 
                           'ar_1', 'ar_10', 'ar_100',
                           'ar_small', 'ar_medium', 'ar_large']:
                    history[f'val_{key}'].append(0.0)
            
            # Print summary
            print(f"\n{'='*70}")
            print(f"Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val mAP@[.5:.95]: {val_map:.4f}")
            print(f"  Val AP@0.50: {ap50:.4f}")
            print(f"  Val AP@0.75: {ap75:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"{'='*70}")
            
            # Save metrics after every epoch
            metrics = {
                'train_loss': train_loss,
                'val_map': val_map,
                'val_ap50': ap50,
                'val_ap75': ap75,
                'history': history
            }
            
            # Save history files after every epoch
            with open(output_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)
            
            save_metrics_csv(history, output_dir)
            
            # Generate and save plots after every epoch
            if epoch % args.plot_frequency == 0 or epoch == 1:
                try:
                    plot_training_curves(history, plots_dir)
                except Exception as e:
                    print(f"Warning: Failed to generate plots: {e}")
            
            # Save latest checkpoint
            save_checkpoint(model, optimizer, lr_scheduler, epoch, metrics, 
                          checkpoints_dir, 'latest.pt')
            
            # Save best checkpoint
            if ap50 > best_ap50:
                best_map = val_map
                best_ap50 = ap50
                best_ap75 = ap75
                epochs_no_improve = 0
                
                save_path = save_checkpoint(model, optimizer, lr_scheduler, epoch, 
                                           metrics, checkpoints_dir, 'best.pt')
                print(f"✅ New best model saved!")
                print(f"   mAP@[.5:.95]: {best_map:.4f}")
                print(f"   AP@0.50: {best_ap50:.4f}")
                print(f"   AP@0.75: {best_ap75:.4f}")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s)")
                print(f"Best AP@0.50: {best_ap50:.4f}")
            
            # Early stopping
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered after {args.patience} epochs without improvement")
                break
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                save_checkpoint(model, optimizer, lr_scheduler, epoch, metrics,
                              checkpoints_dir, f'checkpoint_epoch_{epoch}.pt')
    
    except KeyboardInterrupt:
        print("\nWarning: Training interrupted by user")
    
    except Exception as e:
        print(f"\nError: Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save final history and plots
        print("\n" + "="*70)
        print("Saving final results...")
        print("="*70)
        
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        save_metrics_csv(history, output_dir)
        
        try:
            plot_training_curves(history, plots_dir)
        except Exception as e:
            print(f"Warning: Failed to generate final plots: {e}")
        
        # Update metadata with completion info
        metadata['end_time'] = datetime.now().isoformat()
        metadata['total_epochs'] = len(history['train_loss'])
        metadata['best_map'] = float(best_map)
        metadata['best_ap50'] = float(best_ap50)
        metadata['best_ap75'] = float(best_ap75)
        
        with open(output_dir / 'training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"Best Results:")
        print(f"  mAP@[.5:.95]: {best_map:.4f}")
        print(f"  AP@0.50: {best_ap50:.4f}")
        print(f"  AP@0.75: {best_ap75:.4f}")
        print(f"\nOutputs saved to: {output_dir.absolute()}")
        print(f"  📁 Checkpoints: {checkpoints_dir.absolute()}")
        print(f"  📊 Plots: {plots_dir.absolute()}")
        print(f"  📈 Metrics: training_metrics.csv")
        print(f"  📝 History: history.json")
        print(f"  📋 Summary: training_summary.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Faster R-CNN Training - Production Version with Full History Tracking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing COCO format data')
    parser.add_argument('--output-dir', type=str, default='./runs/rcnn',
                        help='Directory to save outputs')
    
    # Model
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Do not use pretrained backbone (NOT recommended)')
    parser.add_argument('--anchor-sizes', type=str, default=None,
                        help='Comma-separated anchor sizes (e.g., "8,16,32,64,128")')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=6,
                        help='Batch size per GPU')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='sgd', 
                        choices=['sgd', 'adamw'],
                        help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='step', 
                        choices=['step', 'cosine', 'multistep'],
                        help='Learning rate scheduler')
    parser.add_argument('--lr-step-size', type=int, default=15,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler')
    
    # Other
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision training')
    parser.add_argument('--plot-frequency', type=int, default=1,
                        help='Generate plots every N epochs')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch_size < 1:
        print("Error: Batch size must be at least 1")
        sys.exit(1)
    
    if args.lr <= 0:
        print("Error: Learning rate must be positive")
        sys.exit(1)
    
    main(args)