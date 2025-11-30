# Parking Analyzer — CSE407 Modern Machine Learning

This repository contains the coursework project for CSE407: Modern Machine Learning, supervised by Dr. F. Rivest. The README is structured to facilitate ease of grading and quick navigation to the most important artifacts and notebooks.

## Project Overview

Goal: Analyze parking occupancy and detection using modern object detection architectures, focusing on Faster R-CNN and YOLOv9. The project includes:
- Dataset preparation and COCO-format preprocessing
- Model training pipelines for YOLOv9 and Faster R-CNN
- Evaluation and visualization on test splits
- Side-by-side and pretrained vs. finetuned comparisons

## Quick Links to Important Files

### General
- [dataset_perparation.ipynb](./dataset_perparation.ipynb)  
  Imports the dataset, applies train/val/test split, generates corresponding COCO splits, and performs required preprocessing.

- [model_comparison_rcnn_vs_yolo.ipynb](./model_comparison_rcnn_vs_yolo.ipynb)  
  Provides a side-by-side comparison of finetuned Faster R-CNN vs. YOLOv9 models.

### YOLO
- [yolo_training.ipynb](./yolo/yolo_training.ipynb)  
  Training script to fine-tune YOLOv9.

- [evaluate_yolov9.ipynb](./yolo/evaluate_yolov9.ipynb)  
  Evaluates YOLOv9 on the test dataset with visualizations.

- [yolov9_model_comparison_complete_Version3.ipynb](./yolo/yolov9_model_comparison_complete_Version3.ipynb)  
  Compares the performance of finetuned and pretrained YOLOv9.

### Faster R-CNN
- [train_rcnn_Version3.py](./rcnn/train_rcnn_Version3.py)  
  Training script to fine-tune Faster R-CNN on the parking dataset.  
  Note: This script is provided as a standalone Python file rather than a Jupyter Notebook due to native issues in Jupyter with multiple workers and multithreading.

- [evaluate_rcnn.ipynb](./rcnn/evaluate_rcnn.ipynb)  
  Evaluation script for the finetuned Faster R-CNN.

- [model_comparison_finetuned_pretrained.ipynb](./rcnn/model_comparison_finetuned_pretrained.ipynb)  
  Compares the performance of finetuned and pretrained Faster R-CNN.

## How to Reproduce

1. Environment Setup
   - Python 3.10+ recommended
   - Install dependencies (PyTorch, torchvision, ultralytics or YOLOv9-specific deps, COCO tools):
     - `pip install -r requirements.txt` (if available)
     - Otherwise, ensure:
       - PyTorch with CUDA (if GPU is available)
       - torchvision
       - opencv-python
       - numpy, pandas
       - matplotlib, seaborn
       - pycocotools
       - any YOLOv9 training framework dependencies

2. Dataset Preparation
   - Open and run `dataset_perparation.ipynb`.
   - Confirm output directories for train/val/test splits and generated COCO JSON files.

3. Training
   - YOLOv9: Run `./yolo/yolo_training.ipynb` end-to-end.
   - Faster R-CNN: Run `python ./rcnn/train_rcnn_Version3.py`.
     - If using Jupyter, prefer running from terminal due to multiprocessing/threading constraints.

4. Evaluation and Visualization
   - YOLOv9: Open `./yolo/evaluate_yolov9.ipynb`.
   - Faster R-CNN: Open `./rcnn/evaluate_rcnn.ipynb`.

5. Model Comparisons
   - Finetuned vs. Pretrained: Use `./yolo/yolov9_model_comparison_complete_Version3.ipynb` and `./rcnn/model_comparison_finetuned_pretrained.ipynb`.
   - RCNN vs. YOLOv9: Use `./model_comparison_rcnn_vs_yolo.ipynb`.

## Notes for Grading

- The dataset preparation notebook creates standardized COCO-format splits to ensure consistent evaluation across models.
- The Faster R-CNN training script is a `.py` file due to known Jupyter limitations with multi-worker data loaders and threading; please run from a terminal/IDE.
- Each evaluation notebook includes visualizations and metrics on the test set to streamline assessment.
- Some large files (including models) have been pushed via Github LFS (Large File Storage). I have been having some issues with its bandwidth limit, if this is the case let me know.

## Repository Structure (key folders/files)
- `/yolo` — YOLOv9 training, evaluation, and comparison notebooks
- `/rcnn` — Faster R-CNN training script, evaluation, and comparison notebooks
- `dataset_perparation.ipynb` — Dataset import, splitting, COCO conversion, preprocessing
- `model_comparison_rcnn_vs_yolo.ipynb` — Cross-architecture comparison

## Acknowledgements
- Course: CSE407 — Modern Machine Learning
- Supervisor: Dr. F. Rivest
