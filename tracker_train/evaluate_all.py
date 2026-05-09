import os
from ultralytics import YOLO
import torch
import gc


def evaluate_model(weights_path, dataset_yaml, run_name, use_half_precision):
    precision_label = "FP16 (High Speed)" if use_half_precision else "FP32 (Baseline)"

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {run_name}")
    print(f"Mode: {precision_label}")
    print(f"Dataset: {dataset_yaml}")
    print(f"{'=' * 60}")

    model = YOLO(weights_path)

    model.val(
        data=dataset_yaml,
        split='test',
        project='Final_Evaluation',
        name=run_name,
        device=0,
        half=use_half_precision
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    models_directory = r"models"

    RGB_DATASET = r"VisDrone/data.yaml"
    IR_DATASET = r"hit-uav/dataset.yaml"

    model_files = [f for f in os.listdir(models_directory) if f.endswith('.pt')]

    print(f"Found {len(model_files)} PyTorch models.")
    print("Starting Dual-Evaluation Pipeline (FP32 and FP16)...\n")

    for model_file in model_files:
        full_weights_path = os.path.join(models_directory, model_file)
        base_name, _ = os.path.splitext(model_file)

        # dataset
        if "IR" in base_name.upper():
            target_yaml = IR_DATASET
        else:
            target_yaml = RGB_DATASET

        run_name_fp32 = f"{base_name}_FP32"
        evaluate_model(full_weights_path, target_yaml, run_name_fp32, use_half_precision=False)

        run_name_fp16 = f"{base_name}_FP16"
        evaluate_model(full_weights_path, target_yaml, run_name_fp16, use_half_precision=True)
