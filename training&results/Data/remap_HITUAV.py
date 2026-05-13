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

    train = r"hit-uav/labels/train"
    val = r"hit-uav/labels/val"
    test = r"hit-uav/labels/test"


    # Merged Dataset Mapping

    map = {
        0: 0,
        1: 1,
        2: 1,
        3: 1
    }



    print("Starting remapping process...")

    for path in [train, val,test]:
        mod, drop = process_annotations(path, map)
        print(f"Processed {os.path.basename(os.path.dirname(path))}: {mod} files modified, {drop} labels dropped.")
