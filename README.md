# Tactical Aerial Surveillance & Tracking System

This project is a deep learning-based automated surveillance pipeline designed to process UAV/Drone footage. It performs simultaneous object detection and Multi-Object Tracking (MOT) in real-time, focusing on tactically relevant targets: `Person` and `Vehicle`. 

## 1. Installation & Environment Setup
This project is engineered for production and uses ONNX Runtime to eliminate heavy framework dependencies (like PyTorch).

### Prerequisites
- **Python 3.10 – 3.12** (tested on 3.12)
- **NVIDIA GPU + CUDA Toolkit 12.x (Recommended):** For real-time GPU-accelerated inference, install the [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) from NVIDIA's official page and make sure the CUDA `bin` directory is added to your system `PATH` (the installer does this by default).
  > You do **not** need to install cuDNN separately. The required cuDNN 9.x and cuBLAS 12.x libraries are automatically installed as Python packages via `pip install -r requirements.txt`.

> **Note:** The system will still run on CPU without CUDA installed, but inference will be significantly slower.

### Setup Steps
1. Open a terminal in the project directory and create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate the environment:
   - **Windows (PowerShell):** `.venv\Scripts\activate`
   - **Windows (CMD):** `.venv\Scripts\activate.bat`
   - **Mac/Linux:** `source .venv/bin/activate`
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   > **Windows Note:** If `lap` fails to install, you may need to install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first. Select "Desktop development with C++" during installation, then retry `pip install -r requirements.txt`.

## 2. How to Run the System

### Step 1 — Download Models
Download the trained ONNX models from [Google Drive](https://drive.google.com/drive/folders/1x7lq_nVe8BOSsaC-iIKJOtTVQO0WVc8O?usp=sharing) and place them into the `onnx_models/` folder at the project root. Create the folder if it does not exist.

The folder must contain the following **6 files**:

| File | Description |
|---|---|
| `yolo_RGB_FP32.onnx` | YOLO26s detection model — RGB (visible light) |
| `yolo_IR_FP32.onnx` | YOLO26s detection model — Infrared (thermal) |
| `rtdtr_RGB_FP32.onnx` | RT-DETR-l detection model — RGB |
| `rtdtr_IR_FP32.onnx` | RT-DETR-l detection model — Infrared |
| `reid_mobilenetv3.onnx` | MobileNetV3-Small Re-Identification model (used by BoT-SORT) |

### Step 2 — Start the Server
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

On startup, the terminal will print the active execution device. Look for:
```
Model initialized. Device: GPU (CUDA) | Precision: tensor(float) | Tracker: OC-SORT
```
If it shows `Device: CPU` instead of `GPU (CUDA)`, see the Troubleshooting section below.

### Step 3 — Open the Interface
Open a web browser and navigate to: **http://localhost:8000**

## 3. How to Use the Interface

### Input Format
The system accepts standard video files (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) containing drone or aerial footage.

### Description of Controls
| Control | Description |
|---|---|
| **SELECT VIDEO** | Upload a drone footage video file. Processing begins automatically. |
| **CAMERA TYPE** | Toggle between `RGB` and `INFRARED` to switch detection model variants. |
| **TRACKING ENGINE** | `OC-SORT` (faster, motion-based) or `BoT-SORT` (highly accurate, uses deep learning Re-ID via MobileNet to recover lost targets). |
| **MODEL** | `YOLO26-Small` (fast, lightweight) or `RT-DETR-Large` (higher accuracy, heavier compute). |
| **PROCESSING RATE** | `30 FPS` for fluid playback, or `15 FPS (Tactical)` for heavier models. |
| **CONFIDENCE** | Adjust the detection confidence threshold (0.05 – 0.90). |
| **DEPLOY** | Apply the current configuration without restarting the server. |
| **PAUSE / RESUME** | Pause or resume the analysis stream. |
| **RESTART** | Reset the video to the beginning. |
| **📷 CAPTURE** | Download a screenshot of the current frame with overlays. |

### Interactive Target Lock
Click on any tracked bounding box in the video stream. The selected target will be highlighted in **red** with corner brackets, and its movement trail will be recorded. Click again to deselect.

## 4. Important Notes & Limitations
* **GPU Acceleration:** If an NVIDIA GPU is present with CUDA 12.x installed, the system will automatically use `CUDAExecutionProvider`. The cuDNN and cuBLAS libraries are bundled via pip and do not require separate installation.
* **Processing Load:** The `RT-DETR-Large` model is computationally expensive. If the video stream stutters, switch the Processing Rate to `15 FPS (Tactical)`.
* **Domain Generalization:** The models were fine-tuned on UAV aerial datasets (VisDrone & HIT-UAV). Performance may degrade on standard ground-level camera footage.

## 5. Troubleshooting

| Problem | Solution |
|---|---|
| `Device: CPU` shown on startup (GPU not used) | Ensure CUDA Toolkit 12.x is installed and `nvcc --version` works in your terminal. Reinstall `onnxruntime-gpu` with `pip install --force-reinstall onnxruntime-gpu`. |
| `lap` fails to install | Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select "Desktop development with C++". |
| `ModuleNotFoundError: No module named 'xxx'` | Make sure you activated the virtual environment and ran `pip install -r requirements.txt`. |
| Server starts but browser shows nothing | Hard-refresh the browser with `Ctrl+Shift+R` to clear cached static files. |
| `Cannot read video file` error on upload | Ensure the video codec is supported by OpenCV (H.264 MP4 is recommended). |
| Port 8000 already in use | Kill the existing process: on Windows, run `netstat -ano \| findstr :8000` then `taskkill /PID <PID> /F`. |

## 6. Project Structure & File Descriptions

### Core Architecture
| File | Description |
|---|---|
| `app.py` | FastAPI backend server. Manages WebSocket connections, model loading, video streaming loop, and frame processing coordination. |
| `inference.py` | Deep Learning Inference Engine. Bridges ONNX models (YOLO / RT-DETR) with tracking algorithms. Handles preprocessing, postprocessing, and class assignment. |
| `requirements.txt` | All Python dependencies (OpenCV, ONNX Runtime, FastAPI, cuDNN, cuBLAS, etc.). |

### Multi-Object Trackers (`trackers/`)
| File | Description |
|---|---|
| `basetrack.py` | Shared state machine logic defining track states: Tracked, Lost, Removed. |
| `ocsort/` | **OC-SORT** — Observation-Centric SORT. Fast, lightweight tracking using motion prediction and IoU matrices. |
| `botsort/` | **BoT-SORT** — Robust tracker using deep visual Re-Identification (`reid_mobilenetv3.onnx`) and Global Motion Compensation. |

### Frontend Interface (`static/`)
| File | Description |
|---|---|
| `index.html` | Tactical HUD layout and control panel structure. |
| `style.css` | Visual design system — dark-mode aesthetic, color tokens, responsive layout. |
| `app.js` | Client-side logic — WebSocket management, canvas rendering, overlay drawing, target-lock interaction. |

### Training & Results (`training&results/`)
### Detailed Documentation
For a deep dive into the mathematical logic, function-level breakdowns, and architecture of each module, see the [Project Docs Folder](./project_docs/).

## 7. Citations & Credits

### Tracking Algorithms
This project implements modified versions of the following original open-source works:

[1] N. Aharon, R. Or-El, and T. Hassner, "BoT-SORT: Robust Associations Multi-Object Tracking," arXiv preprint arXiv:2206.14651, 2022. [Online]. Available: https://github.com/NirAharon/BoT-SORT

[2] J. Cao, J. Pang, X. Weng, R. Khirodkar, and K. Kitani, "Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking," arXiv preprint arXiv:2203.14360, 2022. [Online]. Available: https://github.com/noahcao/OC_SORT

### Re-Identification Model
[3] A. Howard et al., "Searching for MobileNetV3," in ICCV, 2019. (Pre-trained on Market-1501 for Re-ID).

### Datasets
If you use the datasets associated with this project, please credit the original authors:

[4] VisDrone Team, "VisDrone-Dataset2023," Kaggle. [Online]. Available: https://www.kaggle.com/datasets/kushagrapandya/visdrone-dataset

[5] C. Yiit, "Aerial-Traffic-Images," Kaggle. [Online]. Available: https://www.kaggle.com/datasets/cihangiryiit/aerial-traffic-images

[6] M. Mandal, "Merged Aerial Traffic & VisDrone Dataset," Kaggle.

[7] J. Suo, T. Wang, X. Zhang, H. Chen, W. Zhou, and W. Shi, "HIT-UAV: A high-altitude infrared thermal dataset for Unmanned Aerial Vehicle-based object detection," Scientific Data, vol. 10, no. 1, p. 467, 2023. [Online]. Available: https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset

License Details: See `license.md` in the respective dataset repositories.
