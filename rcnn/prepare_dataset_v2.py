"""
CARPK Parking Dataset Preparation - PRODUCTION VERSION (FIXED)
Robust handling of HuggingFace dataset with comprehensive error checking
FIXED: Added required COCO JSON fields (info, licenses)
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

class CARPKDataPreparation:
    def __init__(self, output_dir='./data', train_split=0.7, val_split=0.2):
        """
        Args:
            train_split: 70% training
            val_split: 20% validation (10% will be test)
        """
        self.output_dir = Path(output_dir)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = 1.0 - train_split - val_split
        
        # Statistics tracking
        self.stats = {
            'train': {'images': 0, 'boxes': 0, 'tiny_boxes': 0, 'invalid_boxes': 0},
            'val': {'images': 0, 'boxes': 0, 'tiny_boxes': 0, 'invalid_boxes': 0},
            'test': {'images': 0, 'boxes': 0, 'tiny_boxes': 0, 'invalid_boxes': 0}
        }
        
        self.box_sizes = []  # Track box sizes for anchor analysis
        self.bbox_format = None  # Will be detected automatically
        
    def create_directories(self):
        """Create directory structure"""
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
        print(f" Created directory structure at {self.output_dir}")
    
    def detect_bbox_format(self, bbox, img_width, img_height):
        """
        Auto-detect bounding box format by examining values
        Returns: 'xyxy' or 'xywh'
        """
        if self.bbox_format is not None:
            return self.bbox_format
        
        x1, y1, x2_or_w, y2_or_h = bbox
        
        # If third/fourth values are larger than image dimensions, likely xywh
        # If x2 < x1 or y2 < y1, definitely xywh format
        if x2_or_w < x1 or y2_or_h < y1:
            self.bbox_format = 'xywh'
        elif x2_or_w > img_width or y2_or_h > img_height:
            self.bbox_format = 'xywh'
        else:
            # Assume xyxy if coordinates are within bounds
            self.bbox_format = 'xyxy'
        
        print(f" Detected bounding box format: {self.bbox_format}")
        return self.bbox_format
    
    def normalize_bbox(self, bbox, img_width, img_height):
        """
        Convert bbox to [x1, y1, x2, y2] format regardless of input format
        """
        fmt = self.detect_bbox_format(bbox, img_width, img_height)
        
        if fmt == 'xywh':
            x, y, w, h = bbox
            return [x, y, x + w, y + h]
        else:
            return list(bbox)
    
    def get_split_name(self, index, total_samples):
        """Determine split for a sample"""
        if index < int(total_samples * self.train_split):
            return 'train'
        elif index < int(total_samples * (self.train_split + self.val_split)):
            return 'val'
        else:
            return 'test'
    
    def validate_and_clean_bbox(self, bbox, img_width, img_height, min_size=3):
        """
        Validate and clean bounding box
        
        Returns:
            tuple: (is_valid, cleaned_bbox, reason)
        """
        # Normalize to xyxy format
        try:
            bbox = self.normalize_bbox(bbox, img_width, img_height)
        except Exception as e:
            return False, None, f"normalization_error: {e}"
        
        x1, y1, x2, y2 = bbox
        
        # Check if coordinates are in correct order
        if x2 <= x1 or y2 <= y1:
            return False, None, "invalid_coordinates"
        
        # Clip to image boundaries
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        # Check again after clipping
        if x2 <= x1 or y2 <= y1:
            return False, None, "invalid_after_clipping"
        
        # Calculate dimensions
        width = x2 - x1
        height = y2 - y1
        
        # Check minimum size (reject very tiny boxes that models can't learn)
        if width < min_size or height < min_size:
            return False, None, "too_small"
        
        # Store box size for analysis
        self.box_sizes.append((width, height))
        
        return True, [float(x1), float(y1), float(x2), float(y2)], "valid"
    
    def load_and_normalize_image(self, image):
        """
        Load image from various formats and convert to RGB numpy array
        """
        try:
            # Case 1: String path
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
                return np.array(image)
            
            # Case 2: PIL Image
            elif isinstance(image, Image.Image):
                image = image.convert('RGB')
                return np.array(image)
            
            # Case 3: Numpy array
            elif isinstance(image, np.ndarray):
                # Ensure RGB
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.shape[2] == 4:
                    image = image[:, :, :3]
                return image
            
            # Case 4: Bytes (from HuggingFace datasets)
            elif isinstance(image, bytes):
                from io import BytesIO
                image = Image.open(BytesIO(image)).convert('RGB')
                return np.array(image)
            
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
                
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    
    def process_sample(self, sample, sample_idx, split, annotation_id):
        """
        Process a single sample with extensive validation
        
        Returns:
            tuple: (image_info, annotations, new_annotation_id) or (None, None, annotation_id) if failed
        """
        try:
            # Get and normalize image
            image_np = self.load_and_normalize_image(sample['image'])
            img_height, img_width = image_np.shape[:2]
            
            # Generate filename
            img_filename = f"{split}_{sample_idx:06d}.jpg"
            img_path = self.output_dir / split / img_filename
            
            # Save image
            Image.fromarray(image_np).save(img_path, quality=95)
            
            # Process annotations
            bboxes = sample.get('bboxes', sample.get('bbox', []))
            if not isinstance(bboxes, list):
                bboxes = list(bboxes)
            
            # Get labels (default to 0 if not present)
            labels = sample.get('labels', sample.get('label', [0] * len(bboxes)))
            if not isinstance(labels, list):
                labels = list(labels)
            
            # Ensure labels list matches bboxes length
            if len(labels) < len(bboxes):
                labels.extend([0] * (len(bboxes) - len(labels)))
            
            annotations = []
            invalid_count = 0
            tiny_count = 0
            
            for bbox, label in zip(bboxes, labels):
                # Skip empty bboxes
                if bbox is None or len(bbox) != 4:
                    invalid_count += 1
                    continue
                
                is_valid, cleaned_bbox, reason = self.validate_and_clean_bbox(
                    bbox, img_width, img_height
                )
                
                if not is_valid:
                    if reason == "too_small":
                        tiny_count += 1
                    else:
                        invalid_count += 1
                    continue
                
                x1, y1, x2, y2 = cleaned_bbox
                width = x2 - x1
                height = y2 - y1
                
                # COCO format: [x, y, width, height]
                coco_bbox = [x1, y1, width, height]
                
                ann = {
                    'id': annotation_id,
                    'image_id': sample_idx,
                    'category_id': 1,  # Car
                    'bbox': coco_bbox,
                    'area': float(width * height),
                    'iscrowd': 0
                }
                annotations.append(ann)
                annotation_id += 1
            
            # Skip images with no valid annotations
            if len(annotations) == 0:
                print(f"  Skipping image {sample_idx}: no valid annotations")
                os.remove(img_path)  # Remove saved image
                return None, None, annotation_id
            
            # Update statistics
            self.stats[split]['images'] += 1
            self.stats[split]['boxes'] += len(annotations)
            self.stats[split]['tiny_boxes'] += tiny_count
            self.stats[split]['invalid_boxes'] += invalid_count
            
            image_info = {
                'id': sample_idx,
                'file_name': img_filename,
                'width': img_width,
                'height': img_height
            }
            
            return image_info, annotations, annotation_id
            
        except Exception as e:
            print(f"  Error processing sample {sample_idx}: {e}")
            return None, None, annotation_id
    
    def create_coco_json(self, split, images, annotations):
        """
        Create COCO format JSON with all required fields
        FIXED: Added 'info' and 'licenses' fields required by pycocotools
        """
        coco_format = {
            'info': {
                'description': 'CARPK Parking Dataset',
                'url': 'https://huggingface.co/datasets/backseollgi/parking_dataset',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'CARPK Dataset Preparation Script',
                'date_created': datetime.now().strftime('%Y/%m/%d')
            },
            'licenses': [
                {
                    'id': 1,
                    'name': 'Unknown',
                    'url': ''
                }
            ],
            'images': images,
            'annotations': annotations,
            'categories': [
                {
                    'id': 1,
                    'name': 'car',
                    'supercategory': 'vehicle'
                }
            ]
        }
        
        json_path = self.output_dir / f'{split}.json'
        with open(json_path, 'w') as f:
            json.dump(coco_format, f, indent=2)
        
        print(f" Created {json_path}")
        print(f"  - Images: {len(images)}")
        print(f"  - Annotations: {len(annotations)}")
    
    def visualize_samples(self, split, num_samples=3):
        """Visualize samples to verify data quality"""
        json_path = self.output_dir / f'{split}.json'
        
        if not json_path.exists():
            print(f"  Cannot visualize {split}: JSON file not found")
            return
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if len(data['images']) == 0:
            print(f"  No images in {split} split to visualize")
            return
        
        print(f"\n Visualizing {num_samples} samples from {split}...")
        
        num_samples = min(num_samples, len(data['images']))
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
        if num_samples == 1:
            axes = [axes]
        
        for idx in range(num_samples):
            img_info = data['images'][idx]
            img_path = self.output_dir / split / img_info['file_name']
            
            if not img_path.exists():
                print(f"  Image not found: {img_path}")
                continue
            
            img = Image.open(img_path)
            
            # Get annotations for this image
            img_anns = [ann for ann in data['annotations'] if ann['image_id'] == img_info['id']]
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"{split} - {len(img_anns)} boxes")
            axes[idx].axis('off')
            
            # Draw boxes
            for ann in img_anns:
                x, y, w, h = ann['bbox']
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                axes[idx].add_patch(rect)
        
        plt.tight_layout()
        save_path = self.output_dir / f'{split}_samples.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved visualization to {save_path}")
        plt.close()
    
    def analyze_box_sizes(self):
        """Analyze bounding box sizes to recommend anchor sizes"""
        if not self.box_sizes:
            print("  No box sizes to analyze")
            return
        
        widths = [w for w, h in self.box_sizes]
        heights = [h for w, h in self.box_sizes]
        areas = [w * h for w, h in self.box_sizes]
        
        print("\n" + "="*70)
        print(" Bounding Box Size Analysis")
        print("="*70)
        
        print(f"\nWidth Statistics:")
        print(f"  Min: {np.min(widths):.1f}px")
        print(f"  Max: {np.max(widths):.1f}px")
        print(f"  Mean: {np.mean(widths):.1f}px")
        print(f"  Median: {np.median(widths):.1f}px")
        print(f"  25th percentile: {np.percentile(widths, 25):.1f}px")
        print(f"  75th percentile: {np.percentile(widths, 75):.1f}px")
        
        print(f"\nHeight Statistics:")
        print(f"  Min: {np.min(heights):.1f}px")
        print(f"  Max: {np.max(heights):.1f}px")
        print(f"  Mean: {np.mean(heights):.1f}px")
        print(f"  Median: {np.median(heights):.1f}px")
        print(f"  25th percentile: {np.percentile(heights, 25):.1f}px")
        print(f"  75th percentile: {np.percentile(heights, 75):.1f}px")
        
        print(f"\nArea Statistics:")
        print(f"  Min: {np.min(areas):.1f}px²")
        print(f"  Max: {np.max(areas):.1f}px²")
        print(f"  Mean: {np.mean(areas):.1f}px²")
        print(f"  Median: {np.median(areas):.1f}px²")
        
        # Recommend anchor sizes
        print("\n Recommended Anchor Sizes:")
        sqrt_areas = np.sqrt(areas)
        percentiles = [10, 30, 50, 70, 90]
        anchor_sizes = [int(np.percentile(sqrt_areas, p)) for p in percentiles]
        anchor_str = ','.join(map(str, anchor_sizes))
        print(f"  {tuple(anchor_sizes)}")
        print(f"  (Based on {percentiles}th percentiles of sqrt(area))")
        print(f"\n Use this flag when training:")
        print(f"  --anchor-sizes \"{anchor_str}\"")
    
    def prepare_dataset(self, max_samples=None, test_first=True):
        """Main preparation function"""
        print("="*70)
        print("CARPK Dataset Preparation - PRODUCTION VERSION (FIXED)")
        print("="*70)
        
        # Create directories
        self.create_directories()
        
        # Load dataset
        print("\n Loading dataset from HuggingFace...")
        try:
            dataset = load_dataset("backseollgi/parking_dataset", "carpk", streaming=False)
        except Exception as e:
            print(f" Failed to load dataset: {e}")
            print(" Make sure you have internet connection and the dataset exists")
            return
        
        print(f" Dataset loaded successfully")
        print(f" Available splits: {list(dataset.keys())}")
        
        # Get samples from the dataset
        if 'train' in dataset:
            samples = list(dataset['train'])
        elif len(dataset.keys()) > 0:
            key = list(dataset.keys())[0]
            print(f"  'train' split not found, using '{key}' instead")
            samples = list(dataset[key])
        else:
            print(" Dataset has unexpected structure!")
            return
        
        total_samples = len(samples)
        print(f" Found {total_samples} samples")
        
        # Test first sample if requested
        if test_first:
            print("\n Inspecting first sample...")
            sample = samples[0]
            print(f"  Keys: {list(sample.keys())}")
            print(f"  Image type: {type(sample['image'])}")
            
            # Try to get bboxes with different keys
            if 'bboxes' in sample:
                print(f"  Number of bboxes: {len(sample['bboxes'])}")
                if len(sample['bboxes']) > 0:
                    print(f"  First bbox: {sample['bboxes'][0]}")
            elif 'bbox' in sample:
                print(f"  Number of bboxes: {len(sample['bbox'])}")
                if len(sample['bbox']) > 0:
                    print(f"  First bbox: {sample['bbox'][0]}")
            else:
                print("    Warning: No 'bboxes' or 'bbox' field found!")
        
        if max_samples:
            total_samples = min(total_samples, max_samples)
            samples = samples[:total_samples]
            print(f" Limited to {total_samples} samples for testing")
        
        # Check dataset size
        if total_samples < 1000:
            print(f"\n  WARNING: Only {total_samples} samples available!")
            print("   Faster R-CNN performs best with 2000+ images")
            print("   Consider using a pretrained model and strong augmentation")
        
        # Process samples
        print("\n Processing samples...")
        
        train_images, train_annotations = [], []
        val_images, val_annotations = [], []
        test_images, test_annotations = [], []
        
        train_ann_id = 1
        val_ann_id = 1
        test_ann_id = 1
        
        for idx in tqdm(range(total_samples), desc="Processing"):
            sample = samples[idx]
            split = self.get_split_name(idx, total_samples)
            
            if split == 'train':
                img_info, anns, train_ann_id = self.process_sample(
                    sample, idx, split, train_ann_id
                )
                if img_info is not None:
                    train_images.append(img_info)
                    train_annotations.extend(anns)
                    
            elif split == 'val':
                img_info, anns, val_ann_id = self.process_sample(
                    sample, idx, split, val_ann_id
                )
                if img_info is not None:
                    val_images.append(img_info)
                    val_annotations.extend(anns)
                    
            else:
                img_info, anns, test_ann_id = self.process_sample(
                    sample, idx, split, test_ann_id
                )
                if img_info is not None:
                    test_images.append(img_info)
                    test_annotations.extend(anns)
        
        # Validate we have data
        if len(train_images) == 0:
            print("\n ERROR: No training images were processed successfully!")
            print("   Please check the dataset format and try again")
            return
        
        # Create COCO JSONs
        print("\n Creating COCO format files...")
        self.create_coco_json('train', train_images, train_annotations)
        
        if len(val_images) > 0:
            self.create_coco_json('val', val_images, val_annotations)
        else:
            print("  No validation images - skipping val.json")
        
        if len(test_images) > 0:
            self.create_coco_json('test', test_images, test_annotations)
        else:
            print("  No test images - skipping test.json")
        
        # Analyze and visualize
        self.analyze_box_sizes()
        self.print_statistics()
        
        # Visualize samples
        for split in ['train', 'val']:
            if self.stats[split]['images'] > 0:
                self.visualize_samples(split, num_samples=3)
        
        print("\n" + "="*70)
        print(" Dataset Preparation Complete!")
        print("="*70)
        print(f"\nData saved to: {self.output_dir.absolute()}")
        print("\n Next steps:")
        print("  1. Review the visualization images to verify data quality")
        print("  2. Use the recommended anchor sizes when training")
        print("  3. Start training with: python train_rcnn.py")
    
    def print_statistics(self):
        """Print comprehensive statistics"""
        print("\n" + "="*70)
        print(" Dataset Statistics")
        print("="*70)
        
        total_images = sum(s['images'] for s in self.stats.values())
        total_boxes = sum(s['boxes'] for s in self.stats.values())
        
        if total_images == 0:
            print("\n  No images were processed successfully!")
            return
        
        for split in ['train', 'val', 'test']:
            stats = self.stats[split]
            imgs = stats['images']
            
            if imgs == 0:
                continue
            
            boxes = stats['boxes']
            tiny = stats['tiny_boxes']
            invalid = stats['invalid_boxes']
            avg_boxes = boxes / imgs if imgs > 0 else 0
            
            print(f"\n{split.upper()}:")
            print(f"  Images: {imgs} ({imgs/total_images*100:.1f}%)")
            print(f"  Valid boxes: {boxes}")
            print(f"  Avg boxes/image: {avg_boxes:.2f}")
            print(f"  Rejected (too small): {tiny}")
            print(f"  Rejected (invalid): {invalid}")
        
        print(f"\nTOTAL:")
        print(f"  Images: {total_images}")
        print(f"  Valid boxes: {total_boxes}")
        print(f"  Avg boxes/image: {total_boxes/total_images:.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare CARPK Dataset for Faster R-CNN (FIXED)')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Output directory for processed data')
    parser.add_argument('--train-split', type=float, default=0.7,
                        help='Training split ratio (default: 0.7)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    parser.add_argument('--no-test-first', action='store_true',
                        help='Skip testing first sample inspection')
    
    args = parser.parse_args()
    
    preparer = CARPKDataPreparation( 
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split
    )
    
    preparer.prepare_dataset(
        max_samples=args.max_samples,
        test_first=not args.no_test_first
    )