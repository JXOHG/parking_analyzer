"""
R-CNN Prediction Visualizer (v2 - Import Fix)

This script loads your trained Faster R-CNN model, runs it on a random
image from the validation set, and saves the output with bounding boxes drawn on it.
"""
import torch
import torchvision
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import argparse

def visualize_predictions(checkpoint_path, data_dir, output_path, conf_threshold=0.3):
    print("="*70)
    print("R-CNN Prediction Visualizer")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # 1. Define the V1 model architecture with the CORRECT import path
    print("🏗️  Creating V1 model structure...")
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    # --- THIS IS THE FIX ---
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    # --- END OF FIX ---
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    num_classes = 2 # 1 class (car) + 1 background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 2. Load the checkpoint
    print(f"💾 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("✓ Checkpoint loaded successfully.")

    # 3. Set the score threshold
    model.roi_heads.score_thresh = conf_threshold
    model.to(device).eval()
    print(f"⚙️  Confidence threshold set to: {conf_threshold}")

    # 4. Pick a random validation image
    val_images_dir = Path(data_dir) / 'val'
    image_paths = list(val_images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {val_images_dir}")
    
    random_img_path = random.choice(image_paths)
    print(f"🖼️  Selected random image for prediction: {random_img_path.name}")
    img = Image.open(random_img_path).convert("RGB")

    # 5. Preprocess the image
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).to(device)

    # 6. Run prediction
    print("🧠 Running model inference...")
    with torch.no_grad():
        prediction = model([img_tensor])
    
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    
    print(f"👍 Found {len(boxes)} predictions with score > {conf_threshold}")

    # 7. Draw boxes on the image
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        text = f"{score:.2f}"
        text_bbox = draw.textbbox((x1, y1 - 15), text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x1, y1 - 15), text, font=font, fill="white")

    # 8. Save the output
    img.save(output_path)
    print(f"\n✅ Prediction saved to: {output_path}")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize R-CNN Predictions")
    parser.add_argument('--checkpoint', type=str, default='./rcnn/runs/faster_rcnn/best.pt', help='Path to the R-CNN model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./rcnn/prepared_data/coco', help='Path to the COCO data directory')
    parser.add_argument('--output', type=str, default='rcnn_prediction.jpg', help='Path to save the output image')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for displaying boxes')
    args = parser.parse_args()
    visualize_predictions(args.checkpoint, args.data_dir, args.output, args.threshold)