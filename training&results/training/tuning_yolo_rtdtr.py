import torch
from ultralytics import YOLO, RTDETR

# ==============================================================================
# 1. DEFINE RESEARCH-BACKED SPACES
# ==============================================================================

BASE_AUG = {'degrees': (0.0, 15.0), 'translate': (0.1, 0.2), 'scale': (0.5, 0.9), 'mosaic': (1.0, 1.0)}
HSV_OFF = {'hsv_h': (0.0, 0.0), 'hsv_s': (0.0, 0.0), 'hsv_v': (0.0, 0.0)}
HSV_ON = {'hsv_h': (0.015, 0.015), 'hsv_s': (0.4, 0.7), 'hsv_v': (0.4, 0.7)}

YOLO_RGB_SPACE = {**BASE_AUG, **HSV_ON, 'lr0': (5e-4, 2e-3), 'weight_decay': (5e-4, 2e-3), 'box': (7.5, 15.0),
                  'cls': (0.5, 1.5)}

YOLO_IR_SPACE = {**BASE_AUG, **HSV_OFF, 'lr0': (1e-4, 1e-3), 'weight_decay': (1e-4, 1e-3), 'box': (10.0, 20.0),
                 'cls': (0.5, 1.0)}

RTDETR_RGB_SPACE = {**BASE_AUG, **HSV_ON, 'lr0': (1e-5, 2e-4), 'weight_decay': (1e-4, 5e-4),
                    'warmup_epochs': (3.0, 5.0), 'box': (15.0, 25.0), 'cls': (1.0, 2.5)}

RTDETR_IR_SPACE = {**BASE_AUG, **HSV_OFF, 'lr0': (5e-6, 1e-4), 'weight_decay': (5e-4, 1e-3),
                   'warmup_epochs': (4.0, 5.0), 'box': (20.0, 30.0), 'cls': (0.5, 1.5)}


# ==============================================================================
# 2. MODULAR TUNING ENGINE
# ==============================================================================

def run_tuning_session(model_type, model_weight, data_path, space, project_name, run_name):
    """
    Instantiates and tunes a model with unique directory saving.
    """
    print(f"\n{'=' * 60}\nSTARTING: {model_weight} | Project: {project_name} | Name: {run_name}\n{'=' * 60}")

    if 'rtdetr' in model_weight.lower():
        model = RTDETR(model_weight)
    else:
        model = YOLO(model_weight)

    model.tune(
        data=data_path,
        space=space,
        epochs=30,
        iterations=5,
        optimizer='AdamW',
        workers = 4,
        batch=-1,
        imgsz=640,
        project=project_name,
        name=run_name,
        plots=True,
        save=True,
        val=True,
        device=0
    )

    del model
    torch.cuda.empty_cache()


def main():
    VISDRONE_DATA = r'C:\Users\User\PythonProject\cv_proj\Data\VisDrone\data.yaml'
    HITUAV_DATA = r'C:\Users\User\PythonProject\cv_proj\Data\hit-uav\dataset.yaml'

    # TASK 1: YOLO on VisDrone
    run_tuning_session('YOLO', 'yolo26s.pt', VISDRONE_DATA, YOLO_RGB_SPACE,
                       project_name='tuning_rgb', run_name='yolo_run')

    # TASK 2: YOLO on Hit-UAV
    run_tuning_session('YOLO', 'yolo26s.pt', HITUAV_DATA, YOLO_IR_SPACE,
                       project_name='tuning_ir', run_name='yolo_run')

    # TASK 3: RT-DETR on VisDrone
    run_tuning_session('RTDETR', 'rtdetr-l.pt', VISDRONE_DATA, RTDETR_RGB_SPACE,
                       project_name='tuning_rgb', run_name='rtdetr_run')

    # TASK 4: RT-DETR on Hit-UAV
    run_tuning_session('RTDETR', 'rtdetr-l.pt', HITUAV_DATA, RTDETR_IR_SPACE,
                       project_name='tuning_ir', run_name='rtdetr_run')


if __name__ == '__main__':
    main()