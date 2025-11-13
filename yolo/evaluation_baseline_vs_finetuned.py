"""
YOLOv9 Baseline vs Fine-tuned Comparison Script
Evaluates pretrained COCO model against your fine-tuned parking model
"""

import os
import sys
import torch
import yaml
import subprocess
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

class YOLOv9Evaluator:
    def __init__(self, data_yaml_path='./prepared_data/yolo/data.yaml'):
        """Initialize evaluator"""
        self.data_yaml_path = Path(data_yaml_path)
        self.yolov9_dir = Path('./yolo/yolov9')
        
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"Data YAML not found at {self.data_yaml_path}")
        
        if not self.yolov9_dir.exists():
            raise FileNotFoundError(f"YOLOv9 directory not found. Run training script first to set it up.")
        
        print(f"✓ Found data.yaml at {self.data_yaml_path}")
        print(f"✓ Found YOLOv9 at {self.yolov9_dir}")
    
    def download_pretrained_weights(self, model_size='yolov9-c'):
        """Download pretrained COCO weights as baseline"""
        weights_dir = self.yolov9_dir / 'weights'
        weights_dir.mkdir(exist_ok=True)
        
        weights_urls = {
            'yolov9-c': 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt',
            'yolov9-e': 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt',
            'yolov9-m': 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m.pt',
        }
        
        weights_file = weights_dir / f'{model_size}.pt'
        
        if weights_file.exists():
            print(f"✓ Pretrained weights exist: {weights_file}")
            return weights_file
        
        print(f"\n📥 Downloading {model_size} pretrained weights...")
        import urllib.request
        urllib.request.urlretrieve(weights_urls[model_size], weights_file)
        print(f"✓ Downloaded to {weights_file}")
        
        return weights_file
    
    def find_finetuned_weights(self, search_path='./yolo/runs/train/carpk_yolov9/weights'):
        """Find the fine-tuned model weights"""
        search_path = Path(search_path)
        
        # Look for best.pt in training runs
        possible_paths = [
            search_path / 'carpk_yolov9' / 'weights' / 'best.pt',
            search_path / 'carpk_yolov9' / 'best.pt',
        ]
        
        # Also search recursively
        if search_path.exists():
            found_weights = list(search_path.rglob('best.pt'))
            possible_paths.extend(found_weights)
        
        for path in possible_paths:
            if path.exists():
                print(f"✓ Found fine-tuned weights: {path}")
                return path
        
        # Not found
        print(f"\n⚠️  WARNING: Could not find fine-tuned weights!")
        print(f"   Searched in: {search_path}")
        print(f"   Looking for: best.pt")
        print(f"\n   Please provide the path manually or train the model first.")
        
        return None
    
    def run_validation(self, weights_path, output_name, img_size=640, batch_size=16, device='0'):
        """
        Run YOLOv9 validation/testing
        
        Args:
            weights_path: Path to model weights (.pt file)
            output_name: Name for this evaluation run
            img_size: Image size for evaluation
            batch_size: Batch size
            device: Device to use ('0' for GPU, 'cpu' for CPU)
        
        Returns:
            Dictionary with metrics
        """
        print(f"\n{'='*70}")
        print(f"Evaluating: {output_name}")
        print(f"Weights: {weights_path}")
        print(f"{'='*70}")
        
        # Change to YOLOv9 directory
        original_dir = Path(os.getcwd())
        os.chdir(self.yolov9_dir)
        
        try:
            # Get absolute paths
            data_yaml_abs = self.data_yaml_path.absolute()
            weights_abs = Path(weights_path).absolute()
            project_abs = original_dir / 'runs' / 'evaluate'
            
            # Build validation command
            cmd = [
                sys.executable,
                'val.py',
                '--data', str(data_yaml_abs),
                '--weights', str(weights_abs),
                '--img-size', str(img_size),
                '--batch-size', str(batch_size),
                '--device', str(device),
                '--project', str(project_abs),
                '--name', output_name,
                '--exist-ok',
                '--save-txt',
                '--save-conf',
                '--verbose'
            ]
            
            print(f"\n🔍 Running validation...")
            print(f"Command: {' '.join(cmd)}")
            
            # Run validation
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            print("✓ Validation complete!")
            
            # Parse results
            results_path = project_abs / output_name
            metrics = self._parse_results(results_path, result.stdout)
            
            return metrics
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Validation failed!")
            print(f"Error: {e}")
            print(f"Output: {e.stdout}")
            print(f"Error output: {e.stderr}")
            return None
        finally:
            # Return to original directory
            os.chdir(original_dir)
    
    def _parse_results(self, results_path, stdout_text):
        """Parse validation results from output"""
        metrics = {
            'results_path': str(results_path),
            'timestamp': datetime.now().isoformat()
        }
        
        # Try to parse from stdout
        lines = stdout_text.split('\n')
        for line in lines:
            # Look for metric lines
            if 'Precision' in line or 'P' in line and 'R' in line:
                print(f"📊 {line.strip()}")
            
            # Parse mAP values
            if 'all' in line.lower():
                parts = line.split()
                try:
                    # Typical format: "all  <values>  mAP@0.5  mAP@0.5:0.95"
                    for i, part in enumerate(parts):
                        if part.replace('.', '').isdigit():
                            if 'mAP@0.5' not in metrics:
                                # Try to extract metrics (format varies)
                                # Usually: class, images, labels, P, R, mAP@0.5, mAP@0.5:0.95
                                if len(parts) >= 6:
                                    metrics['precision'] = float(parts[-4]) if parts[-4].replace('.', '').isdigit() else None
                                    metrics['recall'] = float(parts[-3]) if parts[-3].replace('.', '').isdigit() else None
                                    metrics['mAP@0.5'] = float(parts[-2]) if parts[-2].replace('.', '').isdigit() else None
                                    metrics['mAP@0.5:0.95'] = float(parts[-1]) if parts[-1].replace('.', '').isdigit() else None
                except:
                    pass
        
        # Try to read from results.txt if it exists
        results_file = Path(results_path) / 'results.txt'
        if results_file.exists():
            print(f"✓ Found results file: {results_file}")
        
        # Try to read from results.csv if it exists (newer YOLOv9 versions)
        csv_file = Path(results_path) / 'results.csv'
        if csv_file.exists():
            print(f"✓ Found CSV results: {csv_file}")
            try:
                df = pd.read_csv(csv_file)
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    if 'metrics/precision' in df.columns:
                        metrics['precision'] = last_row['metrics/precision']
                    if 'metrics/recall' in df.columns:
                        metrics['recall'] = last_row['metrics/recall']
                    if 'metrics/mAP_0.5' in df.columns:
                        metrics['mAP@0.5'] = last_row['metrics/mAP_0.5']
                    if 'metrics/mAP_0.5:0.95' in df.columns:
                        metrics['mAP@0.5:0.95'] = last_row['metrics/mAP_0.5:0.95']
            except Exception as e:
                print(f"⚠️  Could not parse CSV: {e}")
        
        return metrics
    
    def compare_models(self, baseline_metrics, finetuned_metrics, output_dir='comparison_results'):
        """Generate comparison report and visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print("MODEL COMPARISON REPORT")
        print(f"{'='*70}")
        
        # Print comparison table
        print(f"\n{'Metric':<20} {'Baseline':<15} {'Fine-tuned':<15} {'Improvement':<15}")
        print("-" * 70)
        
        comparison_data = []
        
        for metric in ['precision', 'recall', 'mAP@0.5', 'mAP@0.5:0.95']:
            baseline_val = baseline_metrics.get(metric, 0)
            finetuned_val = finetuned_metrics.get(metric, 0)
            
            if baseline_val and finetuned_val:
                improvement = ((finetuned_val - baseline_val) / baseline_val) * 100
                print(f"{metric:<20} {baseline_val:<15.4f} {finetuned_val:<15.4f} {improvement:+.2f}%")
                
                comparison_data.append({
                    'metric': metric,
                    'baseline': baseline_val,
                    'finetuned': finetuned_val,
                    'improvement_percent': improvement
                })
            else:
                print(f"{metric:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        
        # Create visualizations
        if comparison_data:
            self._create_comparison_plots(comparison_data, output_dir)
        
        # Save detailed report
        report_path = output_dir / 'comparison_report.json'
        report = {
            'timestamp': datetime.now().isoformat(),
            'baseline': baseline_metrics,
            'finetuned': finetuned_metrics,
            'comparison': comparison_data
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Detailed report saved to: {report_path}")
        print(f"✓ Visualizations saved to: {output_dir}")
        
        return comparison_data
    
    def _create_comparison_plots(self, comparison_data, output_dir):
        """Create comparison visualizations"""
        
        # Extract data
        metrics = [d['metric'] for d in comparison_data]
        baseline_vals = [d['baseline'] for d in comparison_data]
        finetuned_vals = [d['finetuned'] for d in comparison_data]
        improvements = [d['improvement_percent'] for d in comparison_data]
        
        # Plot 1: Side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        x = range(len(metrics))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], baseline_vals, width, label='Baseline (Pretrained)', alpha=0.8, color='skyblue')
        ax1.bar([i + width/2 for i in x], finetuned_vals, width, label='Fine-tuned', alpha=0.8, color='orange')
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Baseline vs Fine-tuned Model Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        # Plot 2: Improvement percentages
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        ax2.barh(metrics, improvements, color=colors, alpha=0.7)
        ax2.set_xlabel('Improvement (%)')
        ax2.set_title('Performance Improvement from Fine-tuning')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (metric, imp) in enumerate(zip(metrics, improvements)):
            ax2.text(imp, i, f'{imp:+.1f}%', va='center', ha='left' if imp > 0 else 'right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {output_dir / 'model_comparison.png'}")
        plt.close()
        
        # Plot 3: Radar chart
        try:
            from math import pi
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            angles = [n / len(metrics) * 2 * pi for n in range(len(metrics))]
            angles += angles[:1]
            
            baseline_vals_plot = baseline_vals + [baseline_vals[0]]
            finetuned_vals_plot = finetuned_vals + [finetuned_vals[0]]
            
            ax.plot(angles, baseline_vals_plot, 'o-', linewidth=2, label='Baseline', color='skyblue')
            ax.fill(angles, baseline_vals_plot, alpha=0.25, color='skyblue')
            
            ax.plot(angles, finetuned_vals_plot, 'o-', linewidth=2, label='Fine-tuned', color='orange')
            ax.fill(angles, finetuned_vals_plot, alpha=0.25, color='orange')
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title('Performance Radar Chart', size=16, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'radar_comparison.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved radar chart: {output_dir / 'radar_comparison.png'}")
            plt.close()
        except Exception as e:
            print(f"⚠️  Could not create radar chart: {e}")


def main():
    """Main evaluation function"""
    print("=" * 70)
    print("YOLOv9 BASELINE vs FINE-TUNED COMPARISON")
    print("=" * 70)
    
    # Check CUDA
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"\n🔍 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize evaluator
    evaluator = YOLOv9Evaluator(data_yaml_path='./yolo/prepared_data/yolo/data.yaml')
    
    # Configuration
    model_size = 'yolov9-c'
    img_size = 640
    batch_size = 16
    
    print(f"\n📋 Evaluation Configuration:")
    print(f"   Model: {model_size}")
    print(f"   Image Size: {img_size}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Device: {device}")
    
    # Step 1: Get baseline pretrained weights
    print(f"\n{'='*70}")
    print("STEP 1: Baseline Model (Pretrained COCO)")
    print(f"{'='*70}")
    baseline_weights = evaluator.download_pretrained_weights(model_size)
    
    # Step 2: Find fine-tuned weights
    print(f"\n{'='*70}")
    print("STEP 2: Fine-tuned Model")
    print(f"{'='*70}")
    finetuned_weights = evaluator.find_finetuned_weights()
    
    if finetuned_weights is None:
        print("\n❌ Cannot proceed without fine-tuned weights!")
        print("   Please train the model first using yolo.py")
        print("   Or provide the path manually.")
        
        # Ask user for path
        response = input("\nEnter path to fine-tuned weights (or 'skip' to only test baseline): ")
        if response.lower() != 'skip':
            finetuned_weights = Path(response)
            if not finetuned_weights.exists():
                print(f"❌ File not found: {finetuned_weights}")
                return
    
    # Step 3: Evaluate baseline
    print(f"\n{'='*70}")
    print("STEP 3: Evaluating Baseline Model")
    print(f"{'='*70}")
    
    baseline_metrics = evaluator.run_validation(
        weights_path=baseline_weights,
        output_name='baseline_pretrained',
        img_size=img_size,
        batch_size=batch_size,
        device=device
    )
    
    # Step 4: Evaluate fine-tuned
    finetuned_metrics = None
    if finetuned_weights:
        print(f"\n{'='*70}")
        print("STEP 4: Evaluating Fine-tuned Model")
        print(f"{'='*70}")
        
        finetuned_metrics = evaluator.run_validation(
            weights_path=finetuned_weights,
            output_name='finetuned_model',
            img_size=img_size,
            batch_size=batch_size,
            device=device
        )
    
    # Step 5: Compare and generate report
    if baseline_metrics and finetuned_metrics:
        print(f"\n{'='*70}")
        print("STEP 5: Generating Comparison Report")
        print(f"{'='*70}")
        
        comparison = evaluator.compare_models(baseline_metrics, finetuned_metrics)
        
        print(f"\n{'='*70}")
        print("✅ EVALUATION COMPLETE!")
        print(f"{'='*70}")
        print(f"\n📁 Results saved to:")
        print(f"   - runs/evaluate/baseline_pretrained/")
        print(f"   - runs/evaluate/finetuned_model/")
        print(f"   - comparison_results/")
        
        print(f"\n📊 View comparison plots:")
        print(f"   - comparison_results/model_comparison.png")
        print(f"   - comparison_results/radar_comparison.png")
        print(f"   - comparison_results/comparison_report.json")
    else:
        print("\n⚠️  Could not complete full comparison")
        if baseline_metrics:
            print("✓ Baseline evaluation completed")
        if finetuned_metrics:
            print("✓ Fine-tuned evaluation completed")


if __name__ == "__main__":
    main()