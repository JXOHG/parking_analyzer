"""
Faster R-CNN Training Script - PRODUCTION VERSION
All critical issues fixed with robust error handling
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
warnings.filterwarnings('ignore')

# Check PyTorch version for compatibility
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
print(f"PyTorch version: {torch.__version__}")

class COCODataset(torch.utils.data.Dataset):
    """
    FIXED: Properly handles transforms and validates all data
    """
    def __init__(self, json_file, img_dir, transforms=None):
        from pycocotools.coco import COCO
        
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        
        # Load COCO annotations
        print(f"📂 Loading annotations from {json_file}...")
        try:
            self.coco = COCO(str(json_file))
        except Exception as e:
            print(f"❌ Failed to load COCO file: {e}")
            raise
        
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # Validate dataset
        self._validate_dataset()
        
        print(f"✓ Loaded {len(self.ids)} images from {json_file}")
    
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
            print(f"   ⚠️  {images_without_boxes} images have no annotations")
        
        if images_with_invalid_boxes > 0:
            print(f"   ⚠️  {images_with_invalid_boxes} images have only invalid boxes")
        
        valid_images = len(self.ids) - images_without_boxes - images_with_invalid_boxes
        
        print(f"   ✓ Valid images: {valid_images}")
        print(f"   ✓ Total annotations: {total_boxes}")
        print(f"   ✓ Avg boxes/image: {total_boxes/len(self.ids):.2f}")
        
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
            print(f"⚠️  Failed to load image {img_path}: {e}")
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
    FIXED: Simple transforms that work with detection models
    Only normalize the image - augmentations handled by model
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
    print("🏗️  Creating Model")
    print("="*70)
    
    # Load pretrained ResNet-50 FPN backbone
    if pretrained:
        try:
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
            print("✓ Loaded pretrained weights (FasterRCNN_ResNet50_FPN_V2)")
        except AttributeError:
            # Fallback for older torchvision versions
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
            print("✓ Loaded pretrained weights (FasterRCNN_ResNet50_FPN)")
    else:
        try:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
        except AttributeError:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        print("✓ Initialized model from scratch (NOT recommended)")
    
    # Custom anchor generator if provided
    if anchor_sizes is not None:
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        model.rpn.anchor_generator = anchor_generator
        print(f"✓ Custom RPN anchors: {anchor_sizes}")
        print(f"  Aspect ratios: {aspect_ratios[0]}")
    else:
        print(f"✓ Using default anchors")
    
    # Replace box predictor head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print(f"✓ Replaced box predictor for {num_classes} classes")
    
    return model


def collate_fn(batch):
    """
    Custom collate function for batching
    Handles variable-sized images and annotations
    """
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None):
    """
    Train for one epoch with proper error handling
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
            
            # Filter out images with no boxes (CRITICAL FIX)
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
                    print(f"\n⚠️  NaN/Inf loss detected, skipping batch")
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
                    print(f"\n⚠️  NaN/Inf loss detected, skipping batch")
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
            print(f"\n⚠️  Runtime error in batch: {e}")
            num_skipped += 1
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    # Print detailed loss breakdown
    if num_batches > 0:
        print(f"\n  Loss breakdown:")
        for k, v in loss_dict_cumulative.items():
            print(f"    {k}: {v/num_batches:.4f}")
    
    if num_skipped > 0:
        print(f"  ⚠️  Skipped {num_skipped} batches due to errors")
    
    return avg_loss


@torch.no_grad()
def evaluate(model, data_loader, device, coco_gt):
    """
    Evaluate using official COCO metrics with error handling
    """
    from pycocotools.cocoeval import COCOeval
    
    model.eval()
    
    coco_results = []
    num_errors = 0
    
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
                    # Convert [x1, y1, x2, y2] to [x, y, w, h]
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Skip invalid predictions
                    if w <= 0 or h <= 0:
                        continue
                    
                    coco_results.append({
                        'image_id': int(image_id),
                        'category_id': int(label),
                        'bbox': [float(x1), float(y1), float(w), float(h)],
                        'score': float(score)
                    })
        
        except Exception as e:
            num_errors += 1
            pbar.set_postfix({'errors': num_errors})
            continue
    
    # Evaluate with COCO API
    if len(coco_results) == 0:
        print("\n⚠️  No predictions made!")
        return 0.0, 0.0, 0.0
    
    try:
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract key metrics
        map_50_95 = coco_eval.stats[0]  # AP @ IoU=0.50:0.95
        ap50 = coco_eval.stats[1]       # AP @ IoU=0.50
        ap75 = coco_eval.stats[2]       # AP @ IoU=0.75
        
        return map_50_95, ap50, ap75
        
    except Exception as e:
        print(f"\n⚠️  COCO evaluation failed: {e}")
        print(f"    Generated {len(coco_results)} predictions")
        return 0.0, 0.0, 0.0


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


def main(args):
    print("="*70)
    print("🚀 Faster R-CNN Training - PRODUCTION VERSION")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load datasets
    print("\n📚 Loading Datasets")
    print("="*70)
    
    data_path = Path(args.data_dir)
    
    # Verify data exists
    if not (data_path / 'train.json').exists():
        print(f"❌ Training data not found at {data_path / 'train.json'}")
        print("   Please run prepare_dataset.py first!")
        return
    
    if not (data_path / 'val.json').exists():
        print(f"❌ Validation data not found at {data_path / 'val.json'}")
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
        print(f"❌ Failed to load datasets: {e}")
        return
    
    # Data loaders (FIXED: persistent_workers compatibility)
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
    
    print(f"\n✓ DataLoaders created:")
    print(f"  Workers: {args.workers}")
    print(f"  Persistent workers: {use_persistent_workers}")
    print(f"  Batch size: {args.batch_size}")
    
    # Create model
    num_classes = 2  # Background + car
    
    # Parse anchor sizes if provided
    anchor_sizes = None
    if args.anchor_sizes:
        try:
            anchor_sizes = tuple(tuple([int(s)]) for s in args.anchor_sizes.split(','))
            print(f"\n🎯 Using custom anchor sizes: {anchor_sizes}")
        except Exception as e:
            print(f"⚠️  Failed to parse anchor sizes: {e}")
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
    print(f"\n📊 Model Parameters:")
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
        print(f"\n✓ Using SGD optimizer")
    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        print(f"\n✓ Using AdamW optimizer")
    
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    
    # Learning rate scheduler
    if args.scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
        print(f"✓ Using StepLR scheduler")
        print(f"  Step size: {args.lr_step_size}")
        print(f"  Gamma: {args.lr_gamma}")
    elif args.scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
        print(f"✓ Using CosineAnnealingLR scheduler")
    else:  # multistep
        milestones = [int(args.epochs * 0.6), int(args.epochs * 0.8)]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.1
        )
        print(f"✓ Using MultiStepLR scheduler")
        print(f"  Milestones: {milestones}")
    
    # Mixed precision training (FIXED: proper compatibility check)
    scaler = None
    if args.amp and torch.cuda.is_available():
        if hasattr(torch.cuda.amp, 'GradScaler'):
            scaler = torch.cuda.amp.GradScaler()
            print("✓ Using mixed precision training (AMP)")
        else:
            print("⚠️  AMP not available in this PyTorch version")
    
    # Training loop
    print("\n" + "="*70)
    print("🏋️  Training")
    print("="*70)
    
    best_map = 0.0
    best_ap50 = 0.0
    best_ap75 = 0.0
    epochs_no_improve = 0
    
    history = {
        'train_loss': [],
        'val_map': [],
        'val_ap50': [],
        'val_ap75': [],
        'lr': []
    }
    
    try:
        for epoch in range(1, args.epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*70}")
            
            # Train
            train_loss = train_one_epoch(
                model, optimizer, train_loader, device, epoch, scaler
            )
            
            # Update learning rate
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Evaluate
            print(f"\n📊 Evaluating Epoch {epoch}")
            val_map, ap50, ap75 = evaluate(model, val_loader, device, val_dataset.coco)
            
            # Save history
            history['train_loss'].append(float(train_loss))
            history['val_map'].append(float(val_map))
            history['val_ap50'].append(float(ap50))
            history['val_ap75'].append(float(ap75))
            history['lr'].append(float(current_lr))
            
            # Print summary
            print(f"\n{'='*70}")
            print(f"Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val mAP@[.5:.95]: {val_map:.4f}")
            print(f"  Val AP@0.50: {ap50:.4f}")
            print(f"  Val AP@0.75: {ap75:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"{'='*70}")
            
            # Save latest checkpoint
            metrics = {
                'train_loss': train_loss,
                'val_map': val_map,
                'val_ap50': ap50,
                'val_ap75': ap75,
                'history': history
            }
            
            save_checkpoint(model, optimizer, lr_scheduler, epoch, metrics, 
                          output_dir, 'latest.pt')
            
            # Save best checkpoint
            if ap50 > best_ap50:
                best_map = val_map
                best_ap50 = ap50
                best_ap75 = ap75
                epochs_no_improve = 0
                
                save_path = save_checkpoint(model, optimizer, lr_scheduler, epoch, 
                                           metrics, output_dir, 'best.pt')
                print(f"🎉 New best model saved!")
                print(f"   mAP@[.5:.95]: {best_map:.4f}")
                print(f"   AP@0.50: {best_ap50:.4f}")
                print(f"   AP@0.75: {best_ap75:.4f}")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s)")
                print(f"Best AP@0.50: {best_ap50:.4f}")
            
            # Early stopping
            if epochs_no_improve >= args.patience:
                print(f"\n⏹️  Early stopping triggered after {args.patience} epochs without improvement")
                break
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                save_checkpoint(model, optimizer, lr_scheduler, epoch, metrics,
                              output_dir, f'checkpoint_epoch_{epoch}.pt')
    
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save final history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print("\n" + "="*70)
        print("✅ Training Complete!")
        print("="*70)
        print(f"Best Results:")
        print(f"  mAP@[.5:.95]: {best_map:.4f}")
        print(f"  AP@0.50: {best_ap50:.4f}")
        print(f"  AP@0.75: {best_ap75:.4f}")
        print(f"\nModels saved to: {output_dir.absolute()}")
        print(f"  - best.pt (best AP@0.50)")
        print(f"  - latest.pt (most recent)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Faster R-CNN Training - Production Version',
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
    parser.add_argument('--batch-size', type=int, default=4,
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
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch_size < 1:
        print("❌ Batch size must be at least 1")
        sys.exit(1)
    
    if args.lr <= 0:
        print("❌ Learning rate must be positive")
        sys.exit(1)
    
    main(args)