import os
import json
from glob import glob
from PIL import Image

# === CONFIG ===
DATA_ROOT = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\glove_seg\data\yolo"
OUTPUT_DIR = os.path.join(DATA_ROOT, "coco_annotations")
IMAGE_SIZE = 640

SETS = {
    "train": {
        "img_dir": os.path.join(DATA_ROOT, "train", "images"),
        "label_dir": os.path.join(DATA_ROOT, "train", "labels"),
        "json_path": os.path.join(OUTPUT_DIR, "train_coco.json")
    },
    "val": {
        "img_dir": os.path.join(DATA_ROOT, "val", "images"),
        "label_dir": os.path.join(DATA_ROOT, "val", "labels"),
        "json_path": os.path.join(OUTPUT_DIR, "val_coco.json")
    }
}

def convert(split_name, config):
    print(f"\n🔁 Converting {split_name} set...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    category_set = {}
    annotation_id = 0
    image_id = 0

    def yolo_to_abs(coords):
        return [float(coords[i]) * IMAGE_SIZE if i % 2 == 0 else float(coords[i]) * IMAGE_SIZE for i in range(len(coords))]

    label_files = sorted(glob(os.path.join(config["label_dir"], "*.txt")))

    for label_path in label_files:
        filename = os.path.basename(label_path).replace(".txt", ".jpg")  # or .png
        img_path = os.path.join(config["img_dir"], filename)
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found for label: {filename}")
            continue

        # Register image
        coco_output["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE
        })

        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 7:
                    continue  # skip non-polygon entries
                class_id = int(parts[0])
                coords = yolo_to_abs(parts[1:])

                xs = coords[::2]
                ys = coords[1::2]
                x_min = min(xs)
                y_min = min(ys)
                width_box = max(xs) - x_min
                height_box = max(ys) - y_min
                area = width_box * height_box

                if class_id not in category_set:
                    category_set[class_id] = {
                        "id": class_id,
                        "name": f"class_{class_id}",
                        "supercategory": "glove"
                    }

                coco_output["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "segmentation": [coords],
                    "area": area,
                    "bbox": [x_min, y_min, width_box, height_box],
                    "iscrowd": 0
                })
                annotation_id += 1

        image_id += 1

    coco_output["categories"] = list(category_set.values())

    with open(config["json_path"], "w") as out_file:
        json.dump(coco_output, out_file, indent=4)
        print(f"✅ Saved {split_name} annotations to {config['json_path']}")


# === RUN BOTH ===
for split, cfg in SETS.items():
    convert(split, cfg)
