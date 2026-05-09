import os
from ultralytics import YOLO


def batch_export_onnx(models_dir):

    pt_files = [f for f in os.listdir(models_dir) if f.endswith(".pt") ]

    print(f"Found {len(pt_files)} PyTorch models")

    for pt_file in pt_files:
        full_path = os.path.join(models_dir, pt_file)
        print(f"\n{'-' * 50}")
        print(f"Processing: {pt_file}")
        print(f"{'-' * 50}")

        model = YOLO(full_path)

        half = True if pt_file.startswith("DT") else False
        model.export(
            format="onnx",
            device=0,
            half=half,
            workspace=4,
            simplify=True
        )



if __name__ == "__main__":

    models_directory = r"models"

    batch_export_onnx(models_directory)
