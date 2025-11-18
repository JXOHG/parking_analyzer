"""
Faster R-CNN Evaluation Script
Evaluate a trained model on validation or test data
"""

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import argparse

# ============================================================================
# Dataset Class
# ============================================================================

class COCODataset(torch.utils.data.Dataset):
    """Custom dataset for loading COCO format annotations"""
    def __init__(self, json_file, img_dir, transforms=None):
        from pycocotools.coco import COCO
        
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        
        print(f"Loading annotations from {json_file}...")
        self.coco = COCO(str(json_file))
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        print(f"Loaded {len(self.ids)} images")
    
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
            img = Image.new('RGB', (640, 480))
            coco_anns = []
        
        # Parse annotations
        boxes, labels, areas = [], [], []
        for ann in coco_anns:
            x, y, w, h = ann['bbox']
            
            if w <= 0 or h <= 0:
                continue
            
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
        else:
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
        
        if self.transforms:
            img = self.transforms(img)
        
        return img, target
    
    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    """Custom collate function for batching"""
    return tuple(zip(*batch))


# ============================================================================
# Model Creation
# ============================================================================

def get_model(num_classes, anchor_sizes=None):
    """Create Faster R-CNN model"""
    
    # Load pretrained backbone
    try:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
    except AttributeError:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    
    # Custom anchors if provided
    if anchor_sizes is not None:
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        model.rpn.anchor_generator = anchor_generator
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def get_transform():
    """Get image transforms"""
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())
    transforms.append(torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
    return torchvision.transforms.Compose(transforms)


# ============================================================================
# Evaluation Function
# ============================================================================

@torch.no_grad()
def evaluate(model, data_loader, device, coco_gt, score_threshold=0.05):
    """Evaluate using COCO metrics"""
    from pycocotools.cocoeval import COCOeval
    
    model.eval()
    
    coco_results = []
    
    print(f"\nRunning inference on {len(data_loader)} batches...")
    for images, targets in tqdm(data_loader, desc="Evaluating"):
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
                mask = scores > score_threshold
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]
                
                for box, score, label in zip(boxes, scores, labels):
                    # Convert [x1, y1, x2, y2] to [x, y, w, h]
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    
                    if w <= 0 or h <= 0:
                        continue
                    
                    coco_results.append({
                        'image_id': int(image_id),
                        'category_id': int(label),
                        'bbox': [float(x1), float(y1), float(w), float(h)],
                        'score': float(score)
                    })
        
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue
    
    # Evaluate with COCO API
    if len(coco_results) == 0:
        print("\n⚠️ WARNING: No predictions made!")
        return None, 0, 0, 0
    
    print(f"\nGenerated {len(coco_results)} predictions")
    
    # Score statistics
    scores_only = [r['score'] for r in coco_results]
    print(f"Score range: {min(scores_only):.4f} - {max(scores_only):.4f}")
    print(f"Mean score: {np.mean(scores_only):.4f}")
    print(f"Predictions > 0.5: {sum(1 for s in scores_only if s > 0.5)}")
    
    try:
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        map_50_95 = coco_eval.stats[0]
        ap50 = coco_eval.stats[1]
        ap75 = coco_eval.stats[2]
        
        return coco_eval, map_50_95, ap50, ap75
        
    except Exception as e:
        print(f"\n⚠️ COCO evaluation failed: {e}")
        return None, 0.0, 0.0, 0.0


# ============================================================================
# Visualization Function
# ============================================================================

def visualize_predictions(model, dataset, device, output_dir, num_samples=5, score_threshold=0.5):
    """Visualize predictions vs ground truth"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    print(f"\nGenerating visualizations for {num_samples} samples...")
    
    for idx in range(min(num_samples, len(dataset))):
        # Get sample (without transforms for visualization)
        img_id = dataset.ids[idx]
        img_info = dataset.coco.loadImgs(img_id)[0]
        img_path = dataset.img_dir / img_info['file_name']
        
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        # Get ground truth
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
        anns = dataset.coco.loadAnns(ann_ids)
        
        gt_boxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x+w, y+h])
        
        # Get predictions
        img_tensor = get_transform()(img)
        
        with torch.no_grad():
            prediction = model([img_tensor.to(device)])[0]
        
        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()
        pred_labels = prediction['labels'].cpu().numpy()
        
        # Filter by score
        mask = pred_scores > score_threshold
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Ground truth
        axes[0].imshow(img_np)
        axes[0].set_title(f'Ground Truth ({len(gt_boxes)} boxes)', fontsize=16)
        axes[0].axis('off')
        
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                    edgecolor='lime', facecolor='none')
            axes[0].add_patch(rect)
        
        # Predictions
        axes[1].imshow(img_np)
        axes[1].set_title(f'Predictions ({len(pred_boxes)} boxes, score>{score_threshold})', fontsize=16)
        axes[1].axis('off')
        
        for box, score in zip(pred_boxes, pred_scores):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                    edgecolor='red', facecolor='none')
            axes[1].add_patch(rect)
            
            # Add score label
            axes[1].text(x1, y1-5, f'{score:.2f}', 
                        color='red', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7, pad=2))
        
        plt.tight_layout()
        save_path = output_dir / f'eval_sample_{idx+1}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path.name}")
    
    print(f"\n✓ Visualizations saved to {output_dir}")


# ============================================================================
# Main Evaluation Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate Faster R-CNN Model')
    
    # Paths
    parser.add_argument('--checkpoint', type=str, default='./runs/rcnn/best.pt',
                       help='Path to checkpoint file')
    parser.add_argument('--data-dir', type=str, default='../yolo/prepared_data/coco',
                       help='Path to COCO dataset directory')
    parser.add_argument('--output-dir', type=str, default='./runs/rcnn/evaluation',
                       help='Directory to save evaluation results')
    
    # Model
    parser.add_argument('--num-classes', type=int, default=2,
                       help='Number of classes (including background)')
    
    # Evaluation
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--score-threshold', type=float, default=0.05,
                       help='Score threshold for predictions')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization images')
    parser.add_argument('--num-vis', type=int, default=10,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Setup
    print("="*70)
    print("Faster R-CNN Model Evaluation")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"  Train Loss: {metrics.get('train_loss', 'N/A')}")
        print(f"  Val AP@0.50: {metrics.get('val_ap50', 'N/A')}")
    
    # Load dataset
    print(f"\nLoading validation dataset from {args.data_dir}...")
    
    data_path = Path(args.data_dir)
    val_json = data_path / 'val.json'
    val_img_dir = data_path / 'val'
    
    if not val_json.exists():
        print(f"Error: Validation JSON not found at {val_json}")
        return
    
    val_dataset = COCODataset(
        json_file=val_json,
        img_dir=val_img_dir,
        transforms=get_transform()
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Create model
    print(f"\nCreating model...")
    model = get_model(num_classes=args.num_classes)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"✓ Model loaded successfully")
    
    # Evaluate
    print("\n" + "="*70)
    print("Running Evaluation")
    print("="*70)
    
    coco_eval, map_50_95, ap50, ap75 = evaluate(
        model, val_loader, device, val_dataset.coco, 
        score_threshold=args.score_threshold
    )
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"mAP@[.5:.95]: {map_50_95:.4f}")
    print(f"AP@0.50:      {ap50:.4f}")
    print(f"AP@0.75:      {ap75:.4f}")
    print("="*70)
    
    # Save results
    results = {
        'checkpoint': str(checkpoint_path),
        'map_50_95': float(map_50_95),
        'ap50': float(ap50),
        'ap75': float(ap75),
        'num_images': len(val_dataset),
        'score_threshold': args.score_threshold
    }
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    
    # Visualize if requested
    if args.visualize:
        print("\n" + "="*70)
        print("Generating Visualizations")
        print("="*70)
        
        visualize_predictions(
            model, val_dataset, device, 
            output_dir / 'visualizations',
            num_samples=args.num_vis,
            score_threshold=0.5
        )
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()