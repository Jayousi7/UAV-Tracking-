from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('rtdetr-l.pt')

    results = model.train(
        data='hit-uav/dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        project='DT_results',
        name='DTIR_run1',

        # --- Infrared specific adjustments ---
        hsv_h=0.0,  # Turn off image HSV-Hue augmentation
        hsv_s=0.0,  # Turn off image HSV-Saturation augmentation
        hsv_v=0.0,  # Turn off image HSV-Value augmentation
    )