import os
def process_annotations(directory, mapping):
    """Iterates through YOLO .txt files and remaps class IDs based on a dictionary."""
    if not os.path.exists(directory):
        print(f"Skipping missing directory: {directory}")
        return 0, 0

    modified_count = 0
    dropped_count = 0

    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(directory, filename)
        new_lines = []

        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                old_cls = int(parts[0])


                if old_cls in mapping:
                    parts[0] = str(mapping[old_cls])
                    new_lines.append(" ".join(parts) + "\n")
                else:
                    dropped_count += 1

        with open(file_path, 'w') as f:
            f.writelines(new_lines)

        modified_count += 1

    return modified_count, dropped_count


if __name__ == "__main__":

    merged_train_labels = r"C:\Users\User\PycharmProjects\tracker\VisDrone\train\labels"
    merged_val_labels = r"C:\Users\User\PycharmProjects\tracker\VisDrone\val\labels"
    visdrone_test_labels = r"C:\Users\User\PycharmProjects\tracker\VisDrone\test\labels"


    # Merged Dataset Mapping

    merged_map = {
        2: 0,  # human -> person
        0: 1,  # small-vehicle -> vehicle
        1: 1  # large-vehicle -> vehicle
    }

    # VisDrone Original Mapping (Standard 11-class format)
    # Target: 0: person, 1: vehicle
    visdrone_map = {
        0: 0,  # pedestrian -> person
        1: 0,  # people -> person
        2: 1,  # bicycle -> vehicle
        3: 1,  # car -> vehicle
        4: 1,  # van -> vehicle
        5: 1,  # truck -> vehicle
        6: 1,  # tricycle -> vehicle
        7: 1,  # awning-tricycle -> vehicle
        8: 1,  # bus -> vehicle
        9: 1  # motor -> vehicle
        # Note: Class 10 (others) is dropped because it's not in the map
    }


    for path in [merged_train_labels, merged_val_labels]:
        mod, drop = process_annotations(path, merged_map)
        print(f"Processed {os.path.basename(os.path.dirname(path))}: {mod} files modified, {drop} labels dropped.")

    # Process VisDrone Test Data
    mod, drop = process_annotations(visdrone_test_labels, visdrone_map)
    print(f"Processed VisDrone Test: {mod} files modified, {drop} labels dropped.")
