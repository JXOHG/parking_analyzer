"""
Comprehensive Model Comparison Script (v9 - Final V1 Compatibility with Threshold Fix)

Key Fixes:
1.  **CRITICAL FIX**: Re-added `model.roi_heads.score_thresh` and `nms_thresh` to the R-CNN loading function. This prevents the model from outputting thousands of low-confidence predictions that were destroying the AP score.
2.  Continues to use `strict=False` to ensure the old V1 model checkpoint loads correctly.
"""

import os
import sys
import torch
import torchvision
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import gc
import collections
import matplotlib.pyplot as plt

# --- Manual Metric Calculation Functions (Unchanged) ---
def calculate_iou(box1, box2):
    x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1]); x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area, box2_area = (box1[2] - box1[0]) * (box1[3] - box1[1]), (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0.0

def calculate_manual_metrics(ground_truths, predictions, iou_threshold=0.5):
    print(f"\n--- Calculating manual metrics at IoU threshold {iou_threshold} ---")
    gt_matched = collections.defaultdict(lambda: collections.defaultdict(bool))
    predictions.sort(key=lambda x: x['score'], reverse=True)
    tp = np.zeros(len(predictions))
    total_gt_boxes = sum(len(boxes) for boxes in ground_truths.values())
    if total_gt_boxes == 0: return 0.0, 0.0, 0.0, 0.0
    for i, pred in enumerate(tqdm(predictions, desc="Matching predictions")):
        img_id, gt_boxes = pred['image_id'], ground_truths.get(pred['image_id'], [])
        best_iou, best_gt_idx = 0, -1
        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred['bbox'], gt_box)
            if iou > best_iou: best_iou, best_gt_idx = iou, j
        if best_iou >= iou_threshold and not gt_matched[img_id][best_gt_idx]:
            tp[i], gt_matched[img_id][best_gt_idx] = 1, True
    tp_cumsum = np.cumsum(tp); recalls = tp_cumsum / total_gt_boxes if total_gt_boxes > 0 else np.zeros_like(tp_cumsum); precisions = tp_cumsum / (np.arange(1, len(predictions) + 1) + 1e-10)
    ap = sum(np.max(precisions[recalls >= t]) if np.sum(recalls >= t) > 0 else 0 for t in np.arange(0., 1.1, 0.1)) / 11.
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10); best_f1 = np.max(f1_scores) if len(f1_scores) > 0 else 0.0
    print(f"   Manual AP@{iou_threshold}: {ap:.4f}"); print(f"   Best F1-Score: {best_f1:.4f}")
    return ap, best_f1, precisions, recalls

class ModelComparator:
    def __init__(self, rcnn_weights, yolo_weights, test_data, test_images, output_dir):
        self.rcnn_weights, self.yolo_weights, self.test_data, self.test_images, self.output_dir = Path(rcnn_weights), Path(yolo_weights), Path(test_data), Path(test_images), Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True); self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f"🖥️  Device: {self.device}")
        with open(self.test_data, 'r') as f: self.gt_data = json.load(f)
        self.categories = {cat['id']: cat['name'] for cat in self.gt_data['categories']}
        self.rcnn_label_to_coco_id = {i + 1: cat['id'] for i, cat in enumerate(self.gt_data['categories'])}; self.yolo_class_index_to_coco_id = {i: cat['id'] for i, cat in enumerate(self.gt_data['categories'])}
        self.rcnn_model, self.yolo_model, self.results = None, None, {'rcnn': {}, 'yolo': {}}

    def load_rcnn_model(self, num_classes=2, conf_threshold=0.25, nms_threshold=0.45):
        print("\n" + "="*70 + "\nLoading Faster R-CNN Model (V1 Architecture for Compatibility)\n" + "="*70)
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
        print("   Loading checkpoint with weights_only=False.")
        checkpoint = torch.load(self.rcnn_weights, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # --- THE DEFINITIVE FIX ---
        # Set the score threshold to filter out low-confidence predictions
        model.roi_heads.score_thresh = conf_threshold
        model.roi_heads.nms_thresh = nms_threshold
        # --- END OF FIX ---

        model.to(self.device).eval()
        self.rcnn_model = model
        print(f"✅ Model loaded successfully. Score threshold set to {conf_threshold}.")
        self.results['rcnn']['model_size_mb'] = self.rcnn_weights.stat().st_size / (1024*1024)
        self.results['rcnn']['params_millions'] = sum(p.numel() for p in model.parameters()) / 1e6

    # The rest of the script is unchanged and is omitted for brevity.
    def load_yolo_model(self, conf_threshold=0.25, nms_threshold=0.45):
        print("\n" + "="*70 + "\nLoading YOLOv9 Model\n" + "="*70); yolo_dir = next((p for p in [Path('./yolo/yolov9'), Path('./yolov9')] if p.exists()), None); sys.path.insert(0, str(yolo_dir.resolve()))
        from models.common import DetectMultiBackend; model = DetectMultiBackend(str(self.yolo_weights), device=self.device, dnn=False, data=None, fp16=False)
        model.conf, model.iou = conf_threshold, nms_threshold; self.yolo_model = model; print("✅ Model loaded.")
        self.results['yolo']['model_size_mb'] = self.yolo_weights.stat().st_size / (1024*1024); self.results['yolo']['params_millions'] = sum(p.numel() for p in model.model.parameters()) / 1e6 if hasattr(model, 'model') else 0
    def evaluate_accuracy(self, model_name, max_images=None):
        print(f"\n🎯 Evaluating {model_name} accuracy..."); image_infos = self.gt_data['images'];
        if max_images and max_images < len(image_infos): image_infos = image_infos[:max_images]
        print(f"   Evaluating on {len(image_infos)} images.")
        manual_predictions, manual_ground_truths = [], collections.defaultdict(list)
        for ann in self.gt_data['annotations']:
            if any(img['id'] == ann['image_id'] for img in image_infos): x, y, w, h = ann['bbox']; manual_ground_truths[ann['image_id']].append([x, y, x+w, y+h])
        for img_info in tqdm(image_infos, desc=f"{model_name} inference"):
            img_path = self.test_images / img_info['file_name']
            if not img_path.exists(): continue
            img_tensor, scale_info = self.preprocess_image(img_path, model_name)
            boxes, scores, _ = self.get_predictions(img_tensor, scale_info, model_name)
            for box, score in zip(boxes, scores): manual_predictions.append({'image_id': img_info['id'], 'bbox': box.tolist(), 'score': score})
        ap, f1, precisions, recalls = calculate_manual_metrics(manual_ground_truths, manual_predictions)
        return {'Manual_AP50': ap, 'Best_F1_Score': f1, 'precisions': precisions, 'recalls': recalls}
    def measure_performance(self, model_name):
        print(f"\n⚡ Measuring {model_name} performance..."); model = self.rcnn_model if model_name == 'R-CNN' else self.yolo_model
        dummy_input, _ = self.preprocess_image(next(self.test_images.glob('*.*')), model_name)
        if self.device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
        warmup, iterations, times = 10, 50, []
        for i in range(warmup + iterations):
            inference_input = [dummy_input.to(self.device)] if model_name == 'R-CNN' else dummy_input
            if i == warmup and self.device.type == 'cuda': torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad(): _ = model(inference_input)
            if self.device.type == 'cuda': torch.cuda.synchronize()
            if i >= warmup: times.append(time.time() - start_time)
        latency, fps = np.mean(times) * 1000, 1 / np.mean(times)
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024*1024) if self.device.type == 'cuda' else 0
        print(f"   Latency: {latency:.2f} ms/image, Throughput: {fps:.2f} FPS, Peak GPU Memory: {gpu_mem_mb:.2f} MB")
        return {'latency_ms': latency, 'fps': fps, 'gpu_mem_mb': gpu_mem_mb}
    def run_full_comparison(self, max_images=None):
        self.load_rcnn_model(); self.load_yolo_model(); self.results['rcnn']['performance'] = self.measure_performance('R-CNN'); self.results['yolo']['performance'] = self.measure_performance('YOLO'); self.results['rcnn']['accuracy'] = self.evaluate_accuracy('R-CNN', max_images); self.results['yolo']['accuracy'] = self.evaluate_accuracy('YOLO', max_images); self.generate_report_and_plots()
    def generate_report_and_plots(self):
        rcnn, yolo = self.results['rcnn'], self.results['yolo']; report = ["="*70, "MODEL COMPARISON REPORT", "="*70]; report.extend(["\nACCURACY METRICS", "-"*70, f"{'Metric':<20} {'R-CNN':>12} {'YOLO':>12}", "-"*46, f"{'Manual AP@0.50':<20} {rcnn['accuracy'].get('Manual_AP50', 0):>12.4f} {yolo['accuracy'].get('Manual_AP50', 0):>12.4f}", f"{'Best F1-Score':<20} {rcnn['accuracy'].get('Best_F1_Score', 0):>12.4f} {yolo['accuracy'].get('Best_F1_Score', 0):>12.4f}"]); report.extend(["\nPERFORMANCE & RESOURCE METRICS", "-"*70, f"{'Metric':<20} {'R-CNN':>12} {'YOLO':>12}", "-"*46, f"{'Latency (ms)':<20} {rcnn['performance']['latency_ms']:>12.2f} {yolo['performance']['latency_ms']:>12.2f}", f"{'Throughput (FPS)':<20} {rcnn['performance']['fps']:>12.2f} {yolo['performance']['fps']:>12.2f}", f"{'GPU Memory (MB)':<20} {rcnn['performance']['gpu_mem_mb']:>12.2f} {yolo['performance']['gpu_mem_mb']:>12.2f}", f"{'Parameters (M)':<20} {rcnn['params_millions']:>12.2f} {yolo['params_millions']:>12.2f}", f"{'Model Size (MB)':<20} {rcnn['model_size_mb']:>12.2f} {yolo['model_size_mb']:>12.2f}"]); report_text = "\n".join(report); print("\n" + report_text); (self.output_dir / 'comparison_report.txt').write_text(report_text)
        plt.figure(figsize=(10, 7)); plt.plot(rcnn['accuracy']['recalls'], rcnn['accuracy']['precisions'], label=f"Faster R-CNN (AP={rcnn['accuracy']['Manual_AP50']:.3f})"); plt.plot(yolo['accuracy']['recalls'], yolo['accuracy']['precisions'], label=f"YOLOv9 (AP={yolo['accuracy']['Manual_AP50']:.3f})"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.legend(); plt.grid(True); plt.xlim([0, 1]); plt.ylim([0, 1.05]); pr_curve_path = self.output_dir / 'precision_recall_curve.png'; plt.savefig(pr_curve_path); print(f"\n📊 Precision-Recall curve saved to {pr_curve_path}")
    def preprocess_image(self, image_path, model_name):
        img = Image.open(image_path).convert('RGB');
        if model_name == 'R-CNN':
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]); return transform(img), None
        else:
            img_np = np.array(img); h0, w0 = img_np.shape[:2]; r = 640 / max(h0, w0)
            if r != 1: img_np = np.array(img.resize((int(w0 * r), int(h0 * r)), Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR))
            h, w = img_np.shape[:2]; dw, dh = (640 - w) / 2, (640 - h) / 2; top, bottom, left, right = int(round(dh - 0.1)), int(round(dh + 0.1)), int(round(dw - 0.1)), int(round(dw + 0.1))
            img_padded = np.pad(img_np, ((top, bottom), (left, right), (0, 0)), 'constant', constant_values=114); img_tensor = torch.from_numpy(img_padded.transpose(2, 0, 1)).to(self.device).float() / 255.0
            if img_tensor.ndimension() == 3: img_tensor = img_tensor.unsqueeze(0)
            return img_tensor, {'ratio': r, 'pad_left': left, 'pad_top': top, 'original_size': (w0, h0)}
    @torch.no_grad()
    def get_predictions(self, img_tensor, scale_info, model_name):
        if model_name == 'R-CNN':
            img_tensor = img_tensor.to(self.device); output = self.rcnn_model([img_tensor])[0]; boxes, scores, labels = output['boxes'], output['scores'], output['labels']
            return boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()
        else:
            from utils.general import non_max_suppression; pred = self.yolo_model(img_tensor, augment=False, visualize=False)
            if isinstance(pred, (list, tuple)): pred = pred[0]
            det = non_max_suppression(pred, self.yolo_model.conf, self.yolo_model.iou, agnostic=False, max_det=300)[0]
            if det is None: return np.zeros((0,4)), np.zeros(0), np.zeros(0)
            w0, h0 = scale_info['original_size']; det[:, :4] -= torch.tensor([scale_info['pad_left'], scale_info['pad_top'], scale_info['pad_left'], scale_info['pad_top']], device=det.device); det[:, :4] /= scale_info['ratio']; det[:, [0, 2]] = det[:, [0, 2]].clamp(0, w0); det[:, [1, 3]] = det[:, [1, 3]].clamp(0, h0)
            boxes, scores, yolo_classes = det[:, :4].cpu().numpy(), det[:, 4].cpu().numpy(), det[:, 5].int().cpu().numpy()
            coco_labels = np.array([self.yolo_class_index_to_coco_id.get(cls, -1) for cls in yolo_classes]); valid = coco_labels != -1
            return boxes[valid], scores[valid], coco_labels[valid]

def main():
    print("\n" + "="*70 + "\nMODEL COMPARISON TOOL\n" + "="*70)
    config = {'rcnn_weights': './rcnn/runs/faster_rcnn/best.pt', 'yolo_weights': './yolo/runs/train/carpk_yolov9/weights/best.pt', 'test_data': './yolo/prepared_data/coco/val.json', 'test_images': Path('./yolo/prepared_data/coco/val'), 'output_dir': Path('./comparison_results')}
    if not all(Path(p).exists() for p in list(config.values())[:-1]): print("\n❌ Error: A required file or directory was not found."); return
    choice = input("\nEvaluation Options:\n1. Quick (e.g., 20 images)\n2. Full (all images)\nSelect (1/2): ").strip()
    max_images = 20 if choice == '1' else None; ModelComparator(**config).run_full_comparison(max_images=max_images)

if __name__ == "__main__":
    main()