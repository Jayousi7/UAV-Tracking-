# Tactical Aerial Surveillance & Tracking System

This project is a deep learning-based automated surveillance pipeline designed to process UAV/Drone footage. It performs simultaneous object detection and Multi-Object Tracking (MOT) in real-time, focusing on tactically relevant targets: `Person` and `Vehicle`. 

## 1. Installation & Environment Setup
This project is engineered for production and uses ONNX Runtime to eliminate heavy framework dependencies (like PyTorch).

### Prerequisites
- Python 3.9 or higher

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
1. **Download Models:** Download the trained ONNX models from `(https://drive.google.com/drive/folders/1x7lq_nVe8BOSsaC-iIKJOtTVQO0WVc8O?usp=sharing)` and extract them into the `onnx_models/` folder. Ensure `yolo26n_RGB.onnx`, `yolo26s_RGB.onnx`, `DT_RGB.onnx`, and `reid_mobilenetv3.onnx` are directly inside that folder.

### Option A: Run Natively (Python)
2. Run the FastAPI server:
   ```bash
   python -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```
3. The terminal will indicate that the server has started.
4. Open a web browser and navigate to: `http://localhost:8000`

### Option B: Run via Docker (Containerized)
2. Ensure Docker and Docker Compose are installed on your system.
3. Build and launch the container in the background:
   ```bash
   docker-compose up -d --build
   ```
4. Open a web browser and navigate to: `http://localhost:8000`

## 3. How to Use the Interface

### Input Format
The system accepts standard video files (e.g., `.mp4`, `.avi`) containing drone or aerial footage.

### Expected Outputs
The system outputs a real-time tracking stream to the web browser HUD, overlaying cyan bounding boxes, numerical target IDs, and tracking trajectories over the targets.

### Description of Controls
* **UPLOAD VIDEO:** Click to select and upload your drone footage. The system will automatically begin processing and streaming the video.
* **TRACKING ENGINE:** Choose between `OC-SORT` (faster, relies on motion and IoU) or `BoT-SORT` (highly accurate, uses Deep Learning visual Re-Identification to recover lost targets).
* **MODEL:** Swap between `YOLO26-Nano`, `YOLO26-Small`, and `DT-Large` (RT-DETR).
* **PROCESSING RATE:** Select `30 FPS` for fluid playback with lightweight YOLO models, or `15 FPS (Tactical)` for heavier models (like DT-Large) to maintain real-time synchronization.
* **DEPLOY:** Applies your newly selected Model/Tracker/FPS configuration instantly without restarting the server.
* **PAUSE / PLAY:** Pauses the analysis stream.

### Interactive Target Lock
Click on any tracked bounding box in the video stream. The system will highlight the target in tactical Cyan, record its movements, and maintain a persistent lock on that object even if it is temporarily occluded.

## 4. Important Notes & Limitations
* **Hardware Acceleration:** The system defaults to CPU Execution for maximum compatibility. If an NVIDIA GPU is present and CUDA is configured, ONNX Runtime will automatically attempt to use `CUDAExecutionProvider` for massive speedups.
* **Processing Load Limitation:** The `DT-Large` model is computationally expensive. If the video playback stutters or falls behind real-time, switch the Processing Rate to `15 FPS (Tactical)`.
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
* **`botsort/`**: Contains the codebase for **BoT-SORT**. A highly robust tracker utilizing Deep Learning visual Re-Identification (`reid_mobilenetv3.onnx`) and Global Motion Compensation (GMC) to recover targets after severe occlusions.

### Frontend Interface (`static/`)
* **`index.html`**: The structural layout of the tactical military HUD and control panels.
* **`style.css`**: Defines the visual design system, including the dark-mode aesthetic, cyan highlights, and responsive layouts.
* **`app.js`**: The client-side logic. It maintains the WebSocket connection to the server, dynamically renders the video frames onto an HTML Canvas, draws the bounding boxes, and handles the interactive "Target Lock" clicking mechanism.

### Containerization
* **`Dockerfile`**: Defines the optimized Linux-based container environment, installing core OS dependencies required for video processing (OpenCV).
* **`docker-compose.yml`**: Allows for one-click deployment of the entire system into an isolated container.
