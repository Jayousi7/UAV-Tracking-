# Training & Evaluation Scripts

> **⚠️ IMPORTANT NOTICE:**
> The scripts in this directory (`tracker_train/`) are **NOT REQUIRED** to run the main tracking web application. They are provided solely for reference and transparency regarding how the models were originally trained and evaluated.

This directory contains standalone scripts used by the original author for:
- Downloading and formatting datasets (VisDrone, HIT-UAV)
- Training object detection models (YOLO, RT-DETR)
- Model quantization
- Evaluation and benchmarking

### Dependencies
**You do NOT need to install the dependencies for these scripts** to use the tactical surveillance application. The main application is built to run entirely on lightweight ONNX Runtime execution without heavy PyTorch frameworks.

If you choose to run these scripts for your own research or re-training purposes, you will need to set up a separate environment with PyTorch and the Ultralytics library.
