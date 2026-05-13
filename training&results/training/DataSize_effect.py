import os
import torch
import yaml
from ultralytics import RTDETR, YOLO


def run_experiment(model_weight, data_path, tuned_yaml_path, fraction, dataset_name):
    """
    Runs a single data-fraction experiment using tuned hyperparameters.
    """
    if not os.path.exists(tuned_yaml_path):
        print(f"\n[ERROR] Cannot find tuned HPs at: {tuned_yaml_path}\nSkipping...\n")
        return

    with open(tuned_yaml_path, 'r') as f:
        tuned_params = yaml.safe_load(f)

    clean_model_name = os.path.basename(model_weight).split('.')[0]

    static_overrides = {
        'epochs': 30,
        'imgsz': 640,
        'batch': -1,
        'fraction': fraction,
        'project': f'experiment_{dataset_name}',
        'name': f"{clean_model_name}_frac_{fraction}",
        'workers': 2,
        'optimizer': 'AdamW',
        'patience': 20,
        'device': 0
    }

    final_params = {**tuned_params, **static_overrides}

    if 'rtdetr' in model_weight.lower():
        model = RTDETR(model_weight)
    else:
        model = YOLO(model_weight)

    print(f"\n{'=' * 70}")
    print(f"TRAINING: {clean_model_name} | Data: {dataset_name} | Fraction: {fraction * 100}%")
    print(f"{'=' * 70}")

    model.train(data=data_path, **final_params)

    del model
    torch.cuda.empty_cache()


def main():
    # DATASET PATHS
    VISDRONE_DATA = r'C:\Users\User\PythonProject\cv_proj\Data\VisDrone\data.yaml'
    HITUAV_DATA = r'C:\Users\User\PythonProject\cv_proj\Data\hit-uav\dataset.yaml'

    # MODEL WEIGHTS
    rtdetr = r'C:\Users\User\PythonProject\cv_proj\training\rtdetr-l.pt'
    yolo = r'C:\Users\User\PythonProject\cv_proj\training\yolo26s.pt'

    # TUNED HP YAML PATHS
    rtdetr_ir = r'C:\Users\User\PythonProject\cv_proj\training\runs\detect\tuning_ir\rtdetr_run\best_hyperparameters.yaml'
    yolo_ir = r'C:\Users\User\PythonProject\cv_proj\training/runs/detect/tuning_ir/yolo_run/best_hyperparameters.yaml'

    rtdetr_rgb = r'C:\Users\User\PythonProject\cv_proj\training/runs/detect/tuning_rgb/rtdetr_run/best_hyperparameters.yaml'
    yolo_rgb = r'C:\Users\User\PythonProject\cv_proj\training/runs/detect/tuning_rgb/yolo_run/best_hyperparameters.yaml'

    # EXECUTION QUEUE
    # Structure: (Model Path, Dataset Path, Tuned YAML Path, Dataset Label)
    tasks = [
        # # IR Tasks (Hit-UAV)
        # (rtdetr, HITUAV_DATA, rtdetr_ir, "IR"),
        # (yolo, HITUAV_DATA, yolo_ir, "IR"),

        # RGB Tasks (VisDrone)
        (rtdetr, VISDRONE_DATA, rtdetr_rgb, "RGB"),
        (yolo, VISDRONE_DATA, yolo_rgb, "RGB")
    ]

    fractions_to_test = [0.25, 0.50]
    # the 1 fraction was already tested on the tuning phase so no need to redo it

    for model_weight, data_path, tuned_yaml, dataset_label in tasks:
        for fraction in fractions_to_test:
            run_experiment(
                model_weight=model_weight,
                data_path=data_path,
                tuned_yaml_path=tuned_yaml,
                fraction=fraction,
                dataset_name=dataset_label
            )

    print("\nALL EXPERIMENTS COMPLETED SUCCESSFULLY.")


if __name__ == '__main__':
    main()