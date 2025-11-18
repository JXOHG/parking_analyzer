import argparse
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def draw_yolo_boxes(image_path, label_path):
    """
    Draws YOLO-formatted bounding boxes on an image.

    Args:
        image_path (Path): Path to the image file.
        label_path (Path): Path to the YOLO .txt label file.

    Returns:
        PIL.Image: Image with bounding boxes drawn, or original image if no label file.
    """
    # Open the image
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    # Check if the label file exists
    if not label_path.exists():
        print(f"  - WARNING: Label file not found at {label_path}")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except IOError:
            font = ImageFont.load_default()
        draw.text((10, 10), "NO LABEL FILE FOUND", fill="red", font=font)
        return img

    print(f"  - Found label file: {label_path}")
    draw = ImageDraw.Draw(img)

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            # YOLO format: class_id, x_center, y_center, width, height (all normalized)
            _, x_center, y_center, width, height = map(float, parts)

            # De-normalize coordinates
            abs_x_center = x_center * img_w
            abs_y_center = y_center * img_h
            abs_width = width * img_w
            abs_height = height * img_h

            # Calculate corner coordinates
            x1 = abs_x_center - (abs_width / 2)
            y1 = abs_y_center - (abs_height / 2)
            x2 = abs_x_center + (abs_width / 2)
            y2 = abs_y_center + (abs_height / 2)

            # Draw the rectangle
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
            
    return img

def main(args):
    """
    Main function to find and visualize ground truth labels.
    """
    dataset_dir = Path(args.dataset_dir)
    num_samples = args.num_samples

    # Define paths to test images and labels
    images_dir = dataset_dir / 'test' / 'images'
    labels_dir = dataset_dir / 'test' / 'labels'

    print(f"Searching for images in: {images_dir}")
    print(f"Searching for labels in: {labels_dir}\n")

    # Get a list of all image files
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

    if not image_files:
        print(f"FATAL ERROR: No images found in {images_dir}.")
        print("Please check the 'dataset_dir' path.")
        return

    # Select random samples
    sample_images = random.sample(image_files, min(num_samples, len(image_files)))

    for image_path in sample_images:
        print(f"Processing image: {image_path.name}")
        
        # Construct the corresponding label path
        label_path = labels_dir / f"{image_path.stem}.txt"

        # Draw the boxes
        img_with_boxes = draw_yolo_boxes(image_path, label_path)

        # Display the image
        plt.figure(figsize=(15, 10))
        plt.imshow(img_with_boxes)
        plt.title(f"Ground Truth for: {image_path.name}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO Ground Truth Labels.")
    parser.add_argument(
        '--dataset-dir', 
        type=str, 
        default='./prepared_data/yolo',
        help="Path to the root of your prepared YOLO dataset directory."
    )
    parser.add_argument(
        '--num-samples', 
        type=int, 
        default=3,
        help="Number of random samples to visualize."
    )
    args = parser.parse_args()
    main(args)