import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('./runs/train/carpk_yolov9/results.csv')

# Clean column names (remove leading/trailing whitespace)
df.columns = df.columns.str.strip()

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('YOLOv9 Training Learning Curves - Parking Dataset', fontsize=16, fontweight='bold')

# 1. Training and Validation Loss
ax1 = axes[0, 0]
ax1.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', linewidth=2, color='#ef4444')
ax1.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', linewidth=2, color='#f59e0b')
ax1.plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss', linewidth=2, color='#eab308')
ax1.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linewidth=2, color='#3b82f6', linestyle='--')
ax1.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', linewidth=2, color='#8b5cf6', linestyle='--')
ax1.plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss', linewidth=2, color='#ec4899', linestyle='--')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. mAP Metrics
ax2 = axes[0, 1]
ax2.plot(df['epoch'], df['metrics/mAP_0.5'], label='mAP@0.5', linewidth=2.5, color='#10b981')
ax2.plot(df['epoch'], df['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95', linewidth=2.5, color='#059669')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('mAP', fontsize=11)
ax2.set_title('Mean Average Precision (mAP)', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.65, 1.0])

# 3. Precision and Recall
ax3 = axes[1, 0]
ax3.plot(df['epoch'], df['metrics/precision'], label='Precision', linewidth=2.5, color='#6366f1')
ax3.plot(df['epoch'], df['metrics/recall'], label='Recall', linewidth=2.5, color='#8b5cf6')
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('Score', fontsize=11)
ax3.set_title('Precision and Recall', fontsize=12, fontweight='bold')
ax3.legend(loc='lower right', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0.8, 1.0])

# 4. Summary Statistics Box
ax4 = axes[1, 1]
ax4.axis('off')

# Calculate statistics
final_map50 = df['metrics/mAP_0.5'].iloc[-1]
final_map50_95 = df['metrics/mAP_0.5:0.95'].iloc[-1]
final_precision = df['metrics/precision'].iloc[-1]
final_recall = df['metrics/recall'].iloc[-1]
best_map50_95 = df['metrics/mAP_0.5:0.95'].max()
best_epoch = df['metrics/mAP_0.5:0.95'].idxmax()

# Create summary text
summary_text = f"""
Training Summary
{'='*40}

Total Epochs: {len(df)}

Final Metrics (Epoch {df['epoch'].iloc[-1]}):
  • mAP@0.5:           {final_map50:.4f} ({final_map50*100:.2f}%)
  • mAP@0.5:0.95:      {final_map50_95:.4f} ({final_map50_95*100:.2f}%)
  • Precision:         {final_precision:.4f} ({final_precision*100:.2f}%)
  • Recall:            {final_recall:.4f} ({final_recall*100:.2f}%)

Best Performance:
  • Best mAP@0.5:0.95: {best_map50_95:.4f} ({best_map50_95*100:.2f}%)
  • Achieved at Epoch: {best_epoch}

Loss Reduction:
  • Train Box Loss:    {df['train/box_loss'].iloc[0]:.4f} → {df['train/box_loss'].iloc[-1]:.4f}
  • Val Box Loss:      {df['val/box_loss'].iloc[0]:.4f} → {df['val/box_loss'].iloc[-1]:.4f}
"""

ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('yolov9_training_curves.png', dpi=300, bbox_inches='tight')
print(" Plot saved as 'yolov9_training_curves.png'")

# Display the plot
plt.show()

# Print additional statistics
print("\n" + "="*50)
print("YOLOv9 Training Results Summary")
print("="*50)
print(f"Final mAP@0.5:0.95:      {final_map50_95*100:.2f}%")
print(f"Best mAP@0.5:0.95:       {best_map50_95*100:.2f}% (Epoch {best_epoch})")
print(f"Final Precision:         {final_precision*100:.2f}%")
print(f"Final Recall:            {final_recall*100:.2f}%")
print(f"Training Convergence:    {'Good' if df['train/box_loss'].iloc[-1] < df['train/box_loss'].iloc[0]/2 else 'Moderate'}")
print("="*50)