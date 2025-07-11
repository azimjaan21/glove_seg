from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import mmcv
import os
import cv2
import matplotlib.pyplot as plt

# --- File paths ---
config_file = "configs/rtmpose-m_8xb256-420e_coco-256x192.py"
checkpoint_file = "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"
img_path = "demo.jpg"

# --- COCO skeleton pairs (0-based indexing) ---
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]

# --- Colors ---
SKELETON_COLOR = (0, 255, 255)  # Yellow
KEYPOINT_COLOR = (0, 255, 0)    # Green
WRIST_COLOR = (0, 0, 255)       # Red

# --- File existence checks ---
assert os.path.isfile(config_file), f"Config file not found: {config_file}"
assert os.path.isfile(checkpoint_file), f"Checkpoint file not found: {checkpoint_file}"
assert os.path.isfile(img_path), f"Image not found: {img_path}"

# --- Initialize model and run inference ---
pose_model = init_model(config_file, checkpoint_file, device='cuda')
image = mmcv.imread(img_path)
results = inference_topdown(pose_model, image)
data_sample = merge_data_samples(results)
pred_instances = data_sample.pred_instances
keypoints = pred_instances.keypoints  # (N, num_keypoints, 2) or (N, num_keypoints, 3)

# --- Visualization ---
image_draw = cv2.imread(img_path)

for kp in keypoints:
    # Draw skeleton lines
    for i, j in COCO_SKELETON:
        pt1 = kp[i]
        pt2 = kp[j]
        if len(pt1) == 3 and len(pt2) == 3:
            x1, y1, s1 = pt1
            x2, y2, s2 = pt2
            if s1 > 0.3 and s2 > 0.3:
                cv2.line(image_draw, (int(x1), int(y1)), (int(x2), int(y2)), SKELETON_COLOR, 2)
        else:
            x1, y1 = pt1[:2]
            x2, y2 = pt2[:2]
            cv2.line(image_draw, (int(x1), int(y1)), (int(x2), int(y2)), SKELETON_COLOR, 2)
    # Draw all keypoints
    for idx, point in enumerate(kp):
        if len(point) == 3:
            x, y, score = point
            if score > 0.3:
                color = WRIST_COLOR if idx in [9, 10] else KEYPOINT_COLOR
                cv2.circle(image_draw, (int(x), int(y)), 5, color, -1)
        else:
            x, y = point[:2]
            color = WRIST_COLOR if idx in [9, 10] else KEYPOINT_COLOR
            cv2.circle(image_draw, (int(x), int(y)), 5, color, -1)

# --- Save and display the result ---
cv2.imwrite('demo_with_skeleton.jpg', image_draw)
image_rgb = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 8))
plt.imshow(image_rgb)
plt.axis('off')
plt.title('RTMPose-S: All Persons, Keypoints & Skeleton')
plt.show()
