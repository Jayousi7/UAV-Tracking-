import cv2
import time
import os
import torch
from ultralytics import YOLO


def benchmark_fps(model_path, video_path, use_half, frames_to_test=200):
    precision_label = "FP16" if use_half else "FP32"
    model_name = os.path.basename(model_path)

    print(f"Loading {model_name} in {precision_label}")
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)


    #THE WARM-UP PHASE

    for _ in range(10):
        ret, frame = cap.read()
        if not ret: break
        # verbose=False stops it from spamming the console
        model.predict(frame, half=use_half, device=0, verbose=False)

    #THE BENCHMARK PHASE
    frames_processed = 0
    start_time = time.time()

    while frames_processed < frames_to_test:
        ret, frame = cap.read()
        if not ret:
            break

        model.predict(frame, half=use_half, device=0, verbose=False)
        frames_processed += 1

    end_time = time.time()
    cap.release()

    total_time = end_time - start_time
    fps = frames_processed / total_time if total_time > 0 else 0

    print(f"{model_name} ({precision_label}): Processed {frames_processed} frames at {fps:.2f} FPS")
    print("-" * 50)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    models_directory = r"models"

    sample_video = r"vid.mp4"

    # Grab the PyTorch models
    model_files = [f for f in os.listdir(models_directory) if f.endswith('.pt')]

    print(f"Testing {len(model_files)} models on {sample_video}\n")
    print("=" * 50)

    for model_file in model_files:
        full_path = os.path.join(models_directory, model_file)

        # Test standard FP32
        benchmark_fps(full_path, sample_video, use_half=False)

        # Test high-speed FP16
        benchmark_fps(full_path, sample_video, use_half=True)

