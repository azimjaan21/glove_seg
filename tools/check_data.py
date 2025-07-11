import json
import os
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# === CONFIG ===
coco_json_path = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\glove_seg\data\yolo\coco_annotations\train_coco.json"
images_dir = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\glove_seg\data\yolo\train\images"
num_images_to_show = 5  # how many samples to preview

# === LOAD COCO ===
coco = COCO(coco_json_path)
img_ids = coco.getImgIds()

for i, img_id in enumerate(img_ids[:num_images_to_show]):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(images_dir, img_info['file_name'])

    # Load image
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Failed to load image: {img_path}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Annotations
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # Plot
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)
    ax.set_title(img_info['file_name'])

    for ann in anns:
        if 'segmentation' in ann:
            for seg in ann['segmentation']:
                poly = Polygon(
                    [[seg[i], seg[i+1]] for i in range(0, len(seg), 2)],
                    facecolor='red', edgecolor='white', alpha=0.4
                )
                ax.add_patch(poly)
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

    plt.axis('off')
    plt.tight_layout()
    plt.show()
