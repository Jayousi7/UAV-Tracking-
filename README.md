# Tactical Aerial Surveillance & Tracking System

This project is a deep learning-based automated surveillance pipeline designed to process UAV/Drone footage. It performs simultaneous object detection and Multi-Object Tracking (MOT) in real-time, focusing on tactically relevant targets: `Person` and `Vehicle`. 

## 1. Installation & Environment Setup
This project is engineered for production and uses ONNX Runtime to eliminate heavy framework dependencies (like PyTorch).

### Prerequisites
- Python 3.9 or higher
- **NVIDIA GPU (Recommended):** For real-time GPU-accelerated inference, you must install the following **before** setting up the Python environment:
  - **[CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)** — Download and install from NVIDIA's official page.
  - **[cuDNN 9.x for CUDA 12](https://developer.nvidia.com/cudnn-downloads)** — Download and install from NVIDIA's official page (requires a free NVIDIA developer account).
  - Ensure both `cuda` and `cudnn` `bin` directories are added to your system `PATH`.

> **Note:** The system will still run on CPU without CUDA/cuDNN installed, but inference will be significantly slower.

### Setup Steps
1. Open a terminal in the project directory and create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate the environment:
   - **Windows:** `.venv\Scripts\activate`
   - **Mac/Linux:** `source .venv/bin/activate`
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## 2. How to Run the System Step-by-Step
1. **Download Models:** Download the trained ONNX models from [Google Drive](https://drive.google.com/drive/folders/1x7lq_nVe8BOSsaC-iIKJOtTVQO0WVc8O?usp=sharing) and extract them into the `onnx_models/` folder. Ensure the following files are directly inside that folder:
   - `yolo_RGB_FP32.onnx` — YOLO26s detection model (RGB)
   - `yolo_IR_FP32.onnx` — YOLO26s detection model (Infrared)
   - `rtdtr_RGB_FP32.onnx` — RT-DETR-l detection model (RGB)
   - `rtdtr_IR_FP32.onnx` — RT-DETR-l detection model (Infrared)
   - `osnet.onnx` + `osnet.onnx.data` — OSNet Re-Identification model (used by BoT-SORT)

2. Run the FastAPI server:
   ```bash
   python -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```
3. The terminal will indicate that the server has started.
4. Open a web browser and navigate to: `http://localhost:8000`

## 3. How to Use the Interface

### Input Format
The system accepts standard video files (e.g., `.mp4`, `.avi`) containing drone or aerial footage.

### Expected Outputs
The system outputs a real-time tracking stream to the web browser HUD, overlaying cyan bounding boxes, numerical target IDs, and tracking trajectories over the targets.

### Description of Controls
* **SELECT VIDEO:** Click to select and upload your drone footage. The system will automatically begin processing and streaming the video.
* **CAMERA TYPE:** Toggle between `RGB` and `INFRARED` to switch between the corresponding model variants.
* **TRACKING ENGINE:** Choose between `OC-SORT` (faster, relies on motion and IoU) or `BoT-SORT` (highly accurate, uses Deep Learning visual Re-Identification via OSNet to recover lost targets).
* **MODEL:** Swap between `YOLO26-Small` (fast, lightweight) and `RT-DETR-Large` (higher accuracy, heavier compute).
* **PROCESSING RATE:** Select `30 FPS` for fluid playback with the YOLO model, or `15 FPS (Tactical)` for the heavier RT-DETR model to maintain real-time synchronization.
* **CONFIDENCE:** Adjust the detection confidence threshold using the slider.
* **DEPLOY:** Applies your newly selected Model/Tracker/FPS configuration instantly without restarting the server.
* **PAUSE / RESUME:** Pauses or resumes the analysis stream.
* **RESTART:** Resets the video to the beginning.
* **📷 CAPTURE:** Takes a screenshot of the current frame with overlays.

### Interactive Target Lock
Click on any tracked bounding box in the video stream. The system will highlight the target in tactical green, record its movements, and maintain a persistent lock on that object even if it is temporarily occluded.

## 4. Important Notes & Limitations
* **Hardware Acceleration:** The system defaults to CPU Execution for maximum compatibility. If an NVIDIA GPU is present with **CUDA 12.x** and **cuDNN 9.x** properly installed, ONNX Runtime will automatically use `CUDAExecutionProvider` for massive speedups.
* **Processing Load Limitation:** The `RT-DETR-Large` model is computationally expensive. If the video playback stutters or falls behind real-time, switch the Processing Rate to `15 FPS (Tactical)`.
* **Domain Generalization:** The models were fine-tuned specifically on UAV aerial datasets (VisDrone & HIT-UAV). Performance may drop if run on standard ground-level camera footage.

## 5. Project Structure & File Descriptions

### Core Architecture
* **`app.py`**: The FastAPI backend server. It manages WebSocket connections, loads the detection engine, calculates processing framerates, and handles the continuous video streaming loop.
* **`inference.py`**: The core Deep Learning Inference Engine. It acts as the bridge between the raw ONNX models (YOLO / RT-DETR) and the tracking algorithms. It handles bounding box coordinate transformations and object classification.
* **`main.py`**: A lightweight CLI entry point used purely for local debugging of the detection pipeline without launching the web server.
* **`requirements.txt`**: Contains all necessary Python dependencies (OpenCV, ONNX Runtime, FastAPI, etc.).

### Multi-Object Trackers (`trackers/`)
* **`basetrack.py`**: The foundational state machine logic shared across all tracking algorithms. Defines whether an object is 'Tracked', 'Lost', or 'Removed'.
* **`ocsort/`**: Contains the codebase for **OC-SORT** (Observation-Centric SORT). Uses momentum and Intersection-over-Union (IoU) matrices for fast, lightweight tracking.
* **`botsort/`**: Contains the codebase for **BoT-SORT**. A highly robust tracker utilizing Deep Learning visual Re-Identification (`osnet.onnx`) and Global Motion Compensation (GMC) to recover targets after severe occlusions.

### Frontend Interface (`static/`)
* **`index.html`**: The structural layout of the tactical military HUD and control panels.
* **`style.css`**: Defines the visual design system, including the dark-mode aesthetic, cyan highlights, and responsive layouts.
* **`app.js`**: The client-side logic. It maintains the WebSocket connection to the server, dynamically renders the video frames onto an HTML Canvas, draws the bounding boxes, and handles the interactive "Target Lock" clicking mechanism.

### Training & Results (`training&results/`)
Contains standalone scripts used for model training, hyperparameter tuning, dataset preparation, and evaluation. These are **not required** to run the main application. See the README inside that directory for details.

## 6. Citations & Credits

### Tracking Algorithms
This project implements modified versions of the following original open-source works:
[1] N. Aharon, R. Or-El, and T. Hassner, "BoT-SORT: Robust Associations Multi-Object Tracking," arXiv preprint arXiv:2206.14651, 2022. [Online]. Available: https://github.com/NirAharon/BoT-SORT
[2] J. Cao, J. Pang, X. Weng, R. Khirodkar, and K. Kitani, "Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking," arXiv preprint arXiv:2203.14360, 2022. [Online]. Available: https://github.com/noahcao/OC_SORT

### Re-Identification Model
[3] K. Zhou, Y. Yang, A. Cavallaro, and T. Xiang, "Omni-Scale Feature Learning for Person Re-Identification," in ICCV, 2019. [Online]. Available: https://github.com/KaiyangZhou/deep-person-reid

### Datasets
If you use the datasets associated with this project, please credit the original authors:
[4] VisDrone Team, "VisDrone-Dataset2023," Kaggle. [Online]. Available: https://www.kaggle.com/datasets/kushagrapandya/visdrone-dataset
[5] C. Yiit, "Aerial-Traffic-Images," Kaggle. [Online]. Available: https://www.kaggle.com/datasets/cihangiryiit/aerial-traffic-images
[6] M. Mandal, "Merged Aerial Traffic & VisDrone Dataset," Kaggle. 
[7] J. Suo, T. Wang, X. Zhang, H. Chen, W. Zhou, and W. Shi, "HIT-UAV: A high-altitude infrared thermal dataset for Unmanned Aerial Vehicle-based object detection," Scientific Data, vol. 10, no. 1, p. 467, 2023. [Online]. Available: https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset

License Details: See `license.md` in the respective dataset repositories.
