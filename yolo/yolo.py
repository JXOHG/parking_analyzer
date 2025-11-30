"""
YOLOv9 Training Script for CARPK Parking Dataset
Trains YOLOv9 model on prepared parking space data
"""

import os
import torch
import yaml
from pathlib import Path
import subprocess
import sys

class YOLOv9Trainer:
    def __init__(self, data_yaml_path='./prepared_data/yolo/data.yaml', 
                 output_dir='./runs/yolov9'):
        """
        Initialize YOLOv9 trainer
        
        Args:
            data_yaml_path: Path to data.yaml file
            output_dir: Where to save training outputs
        """
        self.data_yaml_path = Path(data_yaml_path)
        self.output_dir = Path(output_dir)
        self.yolov9_dir = Path('./yolov9')
        
        # Check if data.yaml exists
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"Data YAML not found at {self.data_yaml_path}")
        
        print(f"Found data.yaml at {self.data_yaml_path}")
    
    def setup_yolov9(self):
        """Clone and setup YOLOv9 repository"""
        if self.yolov9_dir.exists():
            print(f"YOLOv9 directory already exists at {self.yolov9_dir}")
            
            # Verify critical files exist
            train_script = self.yolov9_dir / 'train.py'
            if not train_script.exists():
                print(f" train.py not found in {self.yolov9_dir}")
                print("   Deleting and re-cloning...")
                import shutil
                shutil.rmtree(self.yolov9_dir)
                self.setup_yolov9()  # Recursive call to re-clone
                return
        else:
            print("Cloning YOLOv9 repository...")
            subprocess.run([
                'git', 'clone', 
                'https://github.com/WongKinYiu/yolov9.git',
                str(self.yolov9_dir)
            ], check=True)
            print("✓ YOLOv9 cloned successfully")
        
        # Verify structure
        required_files = [
            'train.py',
            'models/detect/yolov9-c.yaml',
            'data/hyps/hyp.scratch-high.yaml'
        ]
        
        print("\n🔍 Verifying YOLOv9 structure...")
        all_exist = True
        for file in required_files:
            file_path = self.yolov9_dir / file
            if file_path.exists():
                print(f" {file} exists")
            else:
                print(f"  {file} not found")
                all_exist = False
        
        if not all_exist:
            raise FileNotFoundError("YOLOv9 repository structure is incomplete. Try deleting 'yolov9' folder and run again.")
        
        # Fix PyTorch 2.6 compatibility issues
        print("\nPatching YOLOv9 for compatibility...")
        self._patch_torch_load()
        self._patch_loss_tal()
        self._patch_plots_pillow()
        
        # Install requirements
        print("\nInstalling YOLOv9 requirements...")
        requirements_file = self.yolov9_dir / 'requirements.txt'
        if requirements_file.exists():
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 
                str(requirements_file)
            ], check=True)
            print("✓ Requirements installed")
    
    def _patch_torch_load(self):
        """Patch train.py to fix PyTorch 2.6 weights_only issue"""
        train_file = self.yolov9_dir / 'train.py'
        
        with open(train_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already patched
        if 'weights_only=False' in content:
            print("  torch.load already patched")
            return
        
        # Replace torch.load calls
        original = "torch.load(weights, map_location='cpu')"
        patched = "torch.load(weights, map_location='cpu', weights_only=False)"
        
        if original in content:
            content = content.replace(original, patched)
            
            with open(train_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("  Patched train.py for PyTorch 2.6")
        else:
            print("Could not find torch.load to patch (already updated?)")
    
    def _patch_loss_tal(self):
        """Patch loss_tal.py to fix the list.view() AttributeError"""
        loss_file = self.yolov9_dir / 'utils' / 'loss_tal.py'
        
        if not loss_file.exists():
            print(" loss_tal.py not found, skipping patch")
            return
        
        
        with open(loss_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already patched
        if '# PATCHED v2: YOLOv9 AuxLoss compatibility' in content:
            print(" loss_tal.py already patched (v2)")
            return
        
        # Remove old patch if present
        if '# PATCHED: Fixed list/tensor compatibility' in content:
            print("   Removing old patch...")
            lines = content.split('\n')
            new_lines = []
            skip_next = 0
            for i, line in enumerate(lines):
                if skip_next > 0:
                    skip_next -= 1
                    continue
                if '# PATCHED: Fixed list/tensor compatibility' in line:
                    skip_next = 2  # Skip the next 2 lines (the if statement and conversion)
                    continue
                new_lines.append(line)
            content = '\n'.join(new_lines)
        
        # The issue is that YOLOv9's AuxLoss returns tuples, not lists
        # We need to handle the case where feats is a tuple of (main_output, aux_output)
        # The main_output is what we want, aux_output is for auxiliary loss
        
        import re
        
        # Find the __call__ method and add a check at the beginning
        # Look for the line with pred_distri, pred_scores
        pattern = r'(\s+)(pred_distri, pred_scores = torch\.cat\(\[xi\.view\(feats\[0\]\.shape\[0\], self\.no, -1\) for xi in feats\], 2\)\.split\()'
        
        # Check if feats is a tuple/list with nested structure from AuxLoss
        replacement = r'''\1# PATCHED v2: YOLOv9 AuxLoss compatibility
\1# Handle case where model returns (main_preds, aux_preds) tuple
\1if isinstance(feats, (tuple, list)) and len(feats) > 0:
\1    # Check if first element is also a tuple/list (aux loss structure)
\1    if isinstance(feats[0], (tuple, list)):
\1        feats = feats[0]  # Use main predictions, ignore aux
\1pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split('''
        
        new_content = re.sub(pattern, replacement, content, count=1)
        
        if new_content != content:
            with open(loss_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("  Patched loss_tal.py for YOLOv9 AuxLoss compatibility")
        else:
            # Manual line-by-line approach
            print("  Applying manual patch to loss_tal.py...")
            
            lines = content.split('\n')
            patched = False
            
            for i, line in enumerate(lines):
                if 'pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0]' in line:
                    # Check we haven't already patched
                    if i > 0 and 'PATCHED' in lines[i-1]:
                        print(" Already patched (manual check)")
                        return
                    
                    indent = len(line) - len(line.lstrip())
                    patch_lines = [
                        ' ' * indent + '# PATCHED v2: YOLOv9 AuxLoss compatibility',
                        ' ' * indent + '# Handle case where model returns (main_preds, aux_preds) tuple',
                        ' ' * indent + 'if isinstance(feats, (tuple, list)) and len(feats) > 0:',
                        ' ' * indent + '    # Check if first element is also a tuple/list (aux loss structure)',
                        ' ' * indent + '    if isinstance(feats[0], (tuple, list)):',
                        ' ' * indent + '        feats = feats[0]  # Use main predictions, ignore aux'
                    ]
                    lines = lines[:i] + patch_lines + lines[i:]
                    patched = True
                    break
            
            if patched:
                with open(loss_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                print("   Manually patched loss_tal.py")
            else:
                print("  Could not automatically patch loss_tal.py")
                print("     Manual fix needed in yolov9/utils/loss_tal.py")
                print("     Before the 'pred_distri, pred_scores = torch.cat' line, add:")
                print("")
                print("     if isinstance(feats, (tuple, list)) and len(feats) > 0:")
                print("         if isinstance(feats[0], (tuple, list)):")
                print("             feats = feats[0]")
    
    def _patch_plots_pillow(self):
        """Patch plots.py to fix Pillow 10+ getsize() deprecation"""
        plots_file = self.yolov9_dir / 'utils' / 'plots.py'
        
        if not plots_file.exists():
            print("    plots.py not found, skipping patch")
            return
        
        with open(plots_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Check if already patched
        if any('# PATCHED: Pillow' in line for line in lines):
            print("  plots.py already patched")
            return
        
        # Find and replace the getsize line
        patched = False
        for i, line in enumerate(lines):
            if 'self.font.getsize(label)' in line and 'w, h =' in line:
                # Get the indentation
                indent = len(line) - len(line.lstrip())
                spaces = ' ' * indent
                
                # Replace with getbbox version
                lines[i] = f'{spaces}# PATCHED: Pillow 10+ compatibility\n'
                lines.insert(i+1, f'{spaces}bbox = self.font.getbbox(label)\n')
                lines.insert(i+2, f'{spaces}w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]  # text width, height\n')
                patched = True
                break
        
        if patched:
            with open(plots_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print("   Patched plots.py for Pillow 10+ compatibility")
        else:
            print("    Could not find getsize() to patch in plots.py")
    
    def download_pretrained_weights(self, model_size='yolov9-c'):
        """
        Download pretrained weights
        
        Args:
            model_size: 'yolov9-c' (compact), 'yolov9-e' (extended), or 'yolov9-m' (medium)
        """
        weights_dir = self.yolov9_dir / 'weights'
        weights_dir.mkdir(exist_ok=True)
        
        weights_urls = {
            'yolov9-c': 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt',
            'yolov9-e': 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt',
            'yolov9-m': 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m.pt',
        }
        
        weights_file = weights_dir / f'{model_size}.pt'
        
        if weights_file.exists():
            print(f" Weights already exist at {weights_file}")
            return weights_file
        
        print(f"\n Downloading {model_size} pretrained weights...")
        import urllib.request
        urllib.request.urlretrieve(weights_urls[model_size], weights_file)
        print(f" Downloaded to {weights_file}")
        
        return weights_file
    
    def train(self, 
              model_size='yolov9-c',
              epochs=100,
              batch_size=8,
              img_size=640,
              device='0',
              workers=8,
              project='runs/train',
              name='carpk_yolov9',
              freeze_layers=None,
              transfer_learning_mode='finetune',
              patience=10):
        """
        Train YOLOv9 model
        
        Args:
            model_size: Model size ('yolov9-c', 'yolov9-e', 'yolov9-m')
            epochs: Number of training epochs
            batch_size: Batch size (reduce if GPU memory issues)
            img_size: Input image size
            device: GPU device ID ('0', '1', etc. or 'cpu')
            workers: Number of data loading workers
            project: Project directory
            name: Experiment name
            freeze_layers: Number of layers to freeze (None, 10, 20, etc.)
            transfer_learning_mode: 
                'finetune' - Retrain all layers (default, best results)
                'freeze_backbone' - Only train detection head (faster, less accurate)
                'scratch' - Train from scratch (slowest, no pretrained weights)
            patience: Early stopping patience (epochs with no improvement)
        """
        print("\n" + "=" * 70)
        print("Starting YOLOv9 Training")
        print("=" * 70)
        
        # Handle transfer learning mode
        if transfer_learning_mode == 'scratch':
            print("\n  Training from scratch (no pretrained weights)")
            weights_file = ''  # Empty string = train from scratch
        else:
            # Get pretrained weights
            weights_file = self.download_pretrained_weights(model_size)
            
            if transfer_learning_mode == 'finetune':
                print("\n  Fine-tuning mode: Retraining ALL layers (recommended)")
            elif transfer_learning_mode == 'freeze_backbone':
                print("\n  Freeze mode: Only training detection head")
                freeze_layers = 10  # Freeze first 10 layers
        
        # Build command with absolute paths BEFORE changing directory
        original_dir = Path(os.getcwd())
        
        # Get absolute paths while we're still in the project directory
        data_yaml_abs = self.data_yaml_path.absolute()
        project_abs = original_dir / project
        
        if weights_file:
            weights_abs = weights_file.absolute()
        
        # Now change to YOLOv9 directory
        os.chdir(self.yolov9_dir)
        
        # Build command with correct paths
        cmd = [
            sys.executable,
            'train.py',  # Relative path since we're in yolov9 dir
            '--data', str(data_yaml_abs),
            '--cfg', f'models/detect/{model_size}.yaml',
            '--hyp', 'data/hyps/hyp.scratch-high.yaml',
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--img-size', str(img_size),
            '--device', str(device),
            '--workers', str(workers),
            '--project', str(project_abs),
            '--name', name,
            '--exist-ok'
        ]
        
        # Add weights if not training from scratch
        if weights_file:
            cmd.extend(['--weights', str(weights_abs)])
        
        # Add freeze layers if specified
        if freeze_layers is not None:
            cmd.extend(['--freeze', str(freeze_layers)])
        
        print(f"\n Training command:")
        print(' '.join(cmd))
        print(f"\n Working directory: {os.getcwd()}")
        print(f" Original directory: {original_dir}")
        print(f" Data YAML: {data_yaml_abs}")
        print("=" * 70)
        
        try:
            # Run training (already in yolov9 directory)
            subprocess.run(cmd, check=True)
            print("\n Training completed successfully!")
            print(f"Results saved to: {project_abs / name}")
        except subprocess.CalledProcessError as e:
            print(f"\n Training failed with error: {e}")
            print("\nTroubleshooting:")
            print(f"1. Check if data.yaml exists: {data_yaml_abs}")
            print("2. Check if yolov9/train.py exists")
            print("3. Try reducing batch_size if GPU memory error")
            print("4. Check if loss_tal.py was properly patched")
            raise
        finally:
            # Return to original directory
            os.chdir(original_dir)
    
    def print_training_tips(self):
        """Print helpful training tips"""
        print("\n" + "=" * 70)
        print("Training Tips & Transfer Learning Modes")
        print("=" * 70)
        print("""
 TRANSFER LEARNING MODES:

1. Fine-tuning (DEFAULT - RECOMMENDED):
   - Starts with pretrained COCO weights
   - Retrains ALL layers
   - Best accuracy for your parking dataset
   - Takes: ~2-4 hours on GPU
   
2. Freeze Backbone:
   - Starts with pretrained COCO weights
   - Only trains detection head (last layers)
   - Faster but less accurate
   - Takes: ~1-2 hours on GPU
   
3. From Scratch:
   - No pretrained weights
   - Trains everything from random initialization
   - Usually worse results (need more data)
   - Takes: ~3-5 hours on GPU

 RECOMMENDATION: Use Fine-tuning (default)
   Cars in parking lots are similar to COCO cars, so pretrained
   weights help significantly!

  GPU Memory Issues?
   - Reduce batch_size (try 8, 4, or 2)
   - Reduce img_size (try 512 or 416)

 Model Sizes:
   - yolov9-c: Compact, faster, less accurate
   - yolov9-m: Medium, balanced
   - yolov9-e: Extended, slower, more accurate

  Training Time Estimates (on modern GPU):
   - 100 epochs: ~2-4 hours
   - 200 epochs: ~4-8 hours

 Monitor Training:
   - Check runs/train/carpk_yolov9/
   - View results.png for loss curves
   - View val_batch*_pred.jpg for predictions

 Early Stopping:
   - Training automatically saves best weights
   - You can stop early with Ctrl+C
   - Best model saved as: best.pt
        """)


def main():
    """Main training function"""
    # Check CUDA availability
    print("🔍 Checking system...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory_gb:.2f} GB")
        
        # Memory recommendations
        if gpu_memory_gb < 8:
            print(f"\n  WARNING: Your GPU has only {gpu_memory_gb:.1f} GB memory")
            print("   Recommended settings for this GPU:")
            print("   - batch_size: 2-4 (currently using 4)")
            print("   - img_size: 640 or lower")
            print("   - Consider using 'yolov9-c' (most memory efficient)")
    else:
        print("  No GPU detected. Training will be VERY slow on CPU.")
        print("   Consider using Google Colab or a cloud GPU.")
    
    # Initialize trainer
    trainer = YOLOv9Trainer(
        data_yaml_path='./prepared_data/yolo/data.yaml',
        output_dir='./runs/yolov9'
    )
    
    # Setup YOLOv9
    trainer.setup_yolov9()
    
    # Print training tips
    trainer.print_training_tips()
    
    # Training configuration
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)
    
    config = {
        'model_size': 'yolov9-c',           # Options: yolov9-c, yolov9-m, yolov9-e
        'epochs': 100,                       # Training epochs
        'batch_size': 4,                     # REDUCED from 8 due to GPU memory
        'img_size': 640,                     # Input image size
        'device': '0',                       # GPU device (use 'cpu' if no GPU)
        'workers': 0,                        # Data loading workers (0 = main process only)
        'project': 'runs/train',             # Output directory
        'name': 'carpk_yolov9',             # Experiment name
        'transfer_learning_mode': 'finetune' # Options: 'finetune', 'freeze_backbone', 'scratch'
    }
    
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n🔍 Transfer Learning Mode: " + config['transfer_learning_mode'])
    if config['transfer_learning_mode'] == 'finetune':
        print("   → Retraining ALL layers with COCO pretrained weights (RECOMMENDED)")
    elif config['transfer_learning_mode'] == 'freeze_backbone':
        print("   → Only training detection head (faster, less accurate)")
    else:
        print("   → Training from scratch (no pretrained weights)")

    
    # Ask for confirmation
    print("\n" + "=" * 70)
    response = input("Start training with these settings? (y/n): ")
    
    if response.lower() == 'y':
        # Start training
        trainer.train(**config)
        
        print("\n" + "=" * 70)
        print(" Training Complete!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Check results in: runs/train/carpk_yolov9/")
        print("2. Best model saved as: runs/train/carpk_yolov9/weights/best.pt")
        print("3. Run evaluation script to get metrics")
    else:
        print("\n⏸  Training cancelled. Modify config dict above and run again.")


if __name__ == "__main__":
    main()