import os

def count_images(folder_path):
    """Returns the number of files in a directory, or 0 if the directory is missing."""
    if not os.path.exists(folder_path):
        return 0
    return len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

# Base paths
ir_data = r"C:\Users\User\PycharmProjects\tracker_prod\training&results\Data\hit-uav"
rgb_data = r"C:\Users\User\PycharmProjects\tracker_prod\training&results\Data\VisDrone"

# HIT-UAV Structure: hit-uav/images/[split]
hit_uav_paths = {
    "Train": os.path.join(ir_data, "images", "train"),
    "Validation": os.path.join(ir_data, "images", "val"),
    "Test": os.path.join(ir_data, "images", "test")
}

# VisDrone Structure: VisDrone/[split]/images
visdrone_paths = {
    "Train": os.path.join(rgb_data, "train", "images"),
    "Validation": os.path.join(rgb_data, "val", "images"),
    "Test": os.path.join(rgb_data, "test", "images")
}

print("--- HIT-UAV (Infrared) Dataset Image Counts ---")
total_hit = 0
for split, path in hit_uav_paths.items():
    count = count_images(path)
    total_hit += count
    print(f"{split}: {count} images")
print(f"Total HIT-UAV: {total_hit}\n")

print("--- VisDrone (RGB) Dataset Image Counts ---")
total_vis = 0
for split, path in visdrone_paths.items():
    count = count_images(path)
    total_vis += count
    print(f"{split}: {count} images")
print(f"Total VisDrone: {total_vis}")