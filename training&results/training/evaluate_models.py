import os
import torch
import csv
from ultralytics import RTDETR, YOLO


def evaluate_single_model(model_name, weight_path, data_path, fraction, dataset_label):
    """
    Evaluates a single model on the test set and returns the core metrics.
    """
    if not os.path.exists(weight_path):
        print(f"\n[WARNING] Weights not found at: {weight_path}")
        return None

    print(f"\n{'=' * 70}")
    print(f"EVALUATING: {model_name} | Data: {dataset_label} | Fraction: {fraction}")
    print(f"{'=' * 70}")

    # Load Model
    if 'rtdetr' in model_name.lower():
        model = RTDETR(weight_path)
    else:
        model = YOLO(weight_path)

    # Run Validation specifically on the 'test' split
    # batch=16 is safe for your RTX 5070, workers=4 for fast dataloading
    metrics = model.val(data=data_path, split='test', batch=16, workers=4, device=0)

    # Extract metrics for the report
    results = {
        'Architecture': model_name,
        'Dataset': dataset_label,
        'Data_Fraction': fraction,
        'Precision': round(metrics.box.mp, 4),
        'Recall': round(metrics.box.mr, 4),
        'mAP_50': round(metrics.box.map50, 4),
        'mAP_50_95': round(metrics.box.map, 4),
        'Inference_Time_ms': round(metrics.speed['inference'], 2),
        'FPS': round(1000 / metrics.speed['inference'], 2) if metrics.speed['inference'] > 0 else 0
    }

    # Clear VRAM safely
    del model
    torch.cuda.empty_cache()

    return results


def main():
    BASE_DIR = r'C:\Users\User\PythonProject\cv_proj'
    VISDRONE_DATA = os.path.join(BASE_DIR, r'Data\VisDrone\data.yaml')
    HITUAV_DATA = os.path.join(BASE_DIR, r'Data\hit-uav\dataset.yaml')

    RUNS_DIR = os.path.join(BASE_DIR, r'training\runs\detect')


    evaluation_queue = [
        # --- HIT-UAV (IR) YOLO ---
        ('YOLO26s', os.path.join(RUNS_DIR, r'experiment_IR\yolo26s_frac_0.25\weights\best.pt'), HITUAV_DATA, '0.25',
         'IR'),
        ('YOLO26s', os.path.join(RUNS_DIR, r'experiment_IR\yolo26s_frac_0.5\weights\best.pt'), HITUAV_DATA, '0.50',
         'IR'),
        ('YOLO26s', os.path.join(RUNS_DIR, r'tuning_ir\yolo_run\weights\best.pt'), HITUAV_DATA, '1.00', 'IR'),

        # --- HIT-UAV (IR) RT-DETR ---
        ('RT-DETR-l', os.path.join(RUNS_DIR, r'experiment_IR\rtdetr-l_frac_0.25\weights\best.pt'), HITUAV_DATA, '0.25',
         'IR'),
        ('RT-DETR-l', os.path.join(RUNS_DIR, r'experiment_IR\rtdetr-l_frac_0.5\weights\best.pt'), HITUAV_DATA, '0.50',
         'IR'),
        ('RT-DETR-l', os.path.join(RUNS_DIR, r'tuning_ir\rtdetr_run\weights\best.pt'), HITUAV_DATA, '1.00', 'IR'),

        # --- VisDrone (RGB) YOLO ---
        ('YOLO26s', os.path.join(RUNS_DIR, r'experiment_RGB\yolo26s_frac_0.25\weights\best.pt'), VISDRONE_DATA, '0.25',
         'RGB'),
        ('YOLO26s', os.path.join(RUNS_DIR, r'experiment_RGB\yolo26s_frac_0.5\weights\best.pt'), VISDRONE_DATA, '0.50',
         'RGB'),
        ('YOLO26s', os.path.join(RUNS_DIR, r'tuning_rgb\yolo_run\weights\best.pt'), VISDRONE_DATA, '1.00', 'RGB'),

        # --- VisDrone (RGB) RT-DETR ---
        ('RT-DETR-l', os.path.join(RUNS_DIR, r'experiment_RGB\rtdetr-l_frac_0.25\weights\best.pt'), VISDRONE_DATA,
         '0.25', 'RGB'),
        ('RT-DETR-l', os.path.join(RUNS_DIR, r'experiment_RGB\rtdetr-l_frac_0.5\weights\best.pt'), VISDRONE_DATA,
         '0.50', 'RGB'),
        ('RT-DETR-l', os.path.join(RUNS_DIR, r'tuning_rgb\rtdetr_run\weights\best.pt'), VISDRONE_DATA, '1.00', 'RGB')
    ]

    all_results = []

    for name, weight_path, data_path, fraction, label in evaluation_queue:
        res = evaluate_single_model(name, weight_path, data_path, fraction, label)
        if res:
            all_results.append(res)

    csv_file = os.path.join(BASE_DIR, 'evaluation_report_metrics.csv')
    if all_results:
        keys = all_results[0].keys()
        with open(csv_file, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_results)
        print(f"\nAll evaluations complete! Metrics saved to: {csv_file}")
    else:
        print("\n No models were successfully evaluated")


if __name__ == '__main__':
    main()