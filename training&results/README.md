# Training & Results

> **⚠️ IMPORTANT NOTICE:**
> The scripts and data in this directory (`training&results/`) are **NOT REQUIRED** to run the main tracking web application. They are provided solely for reference and transparency regarding how the models were originally trained, tuned, and evaluated.

This directory contains standalone scripts and results used by the original author for:
- Downloading and formatting datasets (VisDrone, HIT-UAV)
- Training object detection models (YOLO26s, RT-DETR-l)
- Hyperparameter tuning
- Model quantization and ONNX export
- Evaluation and benchmarking (see `.csv` files for metrics)

### Directory Structure
- **`Data/`**: Dataset download and remapping scripts (VisDrone, HIT-UAV).
- **`training/`**: Training, tuning, and quantization scripts.
- **`models/`**: Trained model weights (`.pt`) and exported ONNX models.
- **`evaluation_report_metrics.csv`**: PyTorch model evaluation results across data fractions.
- **`onnx_evaluation_metrics.csv`**: ONNX model evaluation results (FP16 & FP32).

### Dependencies
**You do NOT need to install the dependencies for these scripts** to use the tactical surveillance application. The main application is built to run entirely on lightweight ONNX Runtime execution without heavy PyTorch frameworks.

If you choose to run these scripts for your own research or re-training purposes, you will need to set up a separate environment with PyTorch and the Ultralytics library.
