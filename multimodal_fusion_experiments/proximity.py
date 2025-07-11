import cv2
import os
import numpy as np
from ultralytics import YOLO

# Global Counters
global_tp_gloves = 0
global_fp_gloves = 0
global_fn_gloves = 0
global_tp_nogloves = 0
global_fp_nogloves = 0
global_fn_nogloves = 0

# Constants
WRIST_INDEXES = [9, 10]
DISTANCE_THRESHOLD = 40
IOU_THRESHOLD = 0.5

# Paths
dataset_path = "C:/Users/dalab/Desktop/azimjaan21/my_PAPERS/AAAI/dataset_july2/valid"
image_folder = os.path.join(dataset_path, "images")
label_folder = os.path.join(dataset_path, "labels")
save_folder = "exp3_results[40]"
os.makedirs(save_folder, exist_ok=True)

# Load Models
pose_model = YOLO("weights/yolo11s-pose.pt")
segmentation_model = YOLO( r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\glove_seg\results\yolo11s\run\weights\best.pt.pt")

# ------------------ Helper Functions ------------------ #

def calculate_iou(mask1, mask2):
    blank = np.zeros((height, width), dtype=np.uint8)
    m1 = cv2.fillPoly(blank.copy(), [mask1], 1)
    m2 = cv2.fillPoly(blank.copy(), [mask2], 1)
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return intersection / union if union > 0 else 0.0

def point_to_line_distance(point, line_start, line_end):
    line_vec = np.array(line_end) - np.array(line_start)
    if np.dot(line_vec, line_vec) == 0:
        return np.linalg.norm(point - line_start)
    t = max(0, min(1, np.dot(point - line_start, line_vec) / np.dot(line_vec, line_vec)))
    projection = line_start + t * line_vec
    return np.linalg.norm(point - projection)

def is_wrist_near_mask(wrist, mask_pts, threshold=DISTANCE_THRESHOLD):
    if cv2.pointPolygonTest(mask_pts, wrist, False) >= 0:
        return True
    for i in range(len(mask_pts) - 1):
        if point_to_line_distance(wrist, mask_pts[i], mask_pts[i + 1]) < threshold:
            return True
    return False

def parse_yolo_seg_label(label_path, img_width, img_height):
    gt_gloves = []
    gt_no_gloves = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                data = line.strip().split()
                class_id = int(data[0])
                points = np.array([float(x) for x in data[1:]]).reshape(-1, 2)
                points[:, 0] *= img_width
                points[:, 1] *= img_height
                polygon = points.astype(np.int32)
                if class_id == 0:
                    gt_gloves.append(polygon)
                elif class_id == 1:
                    gt_no_gloves.append(polygon)
    return gt_gloves, gt_no_gloves

def draw_results(image, wrists, gt_gloves, gt_no_gloves,
                 valid_glove_masks, valid_no_glove_masks,
                 unmatched_gt_gloves, unmatched_gt_nogloves):
    overlay = image.copy()
    # Draw wrists
    for wrist in wrists:
        cv2.circle(overlay, wrist, 5, (255, 255, 0), -1)  # Cyan
    # GT Masks outlines
    for gt in gt_gloves:
        cv2.polylines(overlay, [gt], True, (0, 200, 0), 1)
    for gt in gt_no_gloves:
        cv2.polylines(overlay, [gt], True, (0, 0, 200), 1)
    # Predicted masks
    for mask in valid_glove_masks:
        cv2.polylines(overlay, [mask], True, (0, 255, 0), 2)
    for mask in valid_no_glove_masks:
        cv2.polylines(overlay, [mask], True, (0, 0, 255), 2)
    # Missed GTs (FN)
    for fn in unmatched_gt_gloves:
        cv2.polylines(overlay, [fn], True, (255, 0, 0), 2)
    for fn in unmatched_gt_nogloves:
        cv2.polylines(overlay, [fn], True, (255, 0, 0), 2)
    return overlay

def calculate_metrics():
    precision_g = global_tp_gloves / (global_tp_gloves + global_fp_gloves + 1e-6)
    recall_g = global_tp_gloves / (global_tp_gloves + global_fn_gloves + 1e-6)
    f1_g = 2 * (precision_g * recall_g) / (precision_g + recall_g + 1e-6)
    precision_ng = global_tp_nogloves / (global_tp_nogloves + global_fp_nogloves + 1e-6)
    recall_ng = global_tp_nogloves / (global_tp_nogloves + global_fn_nogloves + 1e-6)
    f1_ng = 2 * (precision_ng * recall_ng) / (precision_ng + recall_ng + 1e-6)
    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS")
    print("="*50)
    print("\n=== GLOVES ===")
    print(f"TP: {global_tp_gloves} | FP: {global_fp_gloves} | FN: {global_fn_gloves}")
    print(f"Precision: {precision_g:.4f}")
    print(f"Recall:    {recall_g:.4f}")
    print(f"F1-Score:  {f1_g:.4f}")
    print("\n=== NO GLOVES ===")
    print(f"TP: {global_tp_nogloves} | FP: {global_fp_nogloves} | FN: {global_fn_nogloves}")
    print(f"Precision: {precision_ng:.4f}")
    print(f"Recall:    {recall_ng:.4f}")
    print(f"F1-Score:  {f1_ng:.4f}")
    print("\n" + "="*50)
    print("CONFUSION MATRIX SUMMARY")
    print("="*50)
    print(f"{'Class':<10} | {'TP':<6} | {'FP':<6} | {'FN':<6}")
    print(f"{'-'*32}")
    print(f"{'Gloves':<10} | {global_tp_gloves:<6} | {global_fp_gloves:<6} | {global_fn_gloves:<6}")
    print(f"{'No-Gloves':<10} | {global_tp_nogloves:<6} | {global_fp_nogloves:<6} | {global_fn_nogloves:<6}")

# ------------------ Main Evaluation Loop ------------------ #

for img_file in os.listdir(image_folder):
    if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(image_folder, img_file)
    image = cv2.imread(img_path)
    height, width, _ = image.shape

    label_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + '.txt')
    gt_gloves, gt_no_gloves = parse_yolo_seg_label(label_path, width, height)

    pose_results = pose_model.predict(img_path, task="pose", device="cuda", conf=0.25, save=False)
    seg_results = segmentation_model.predict(img_path, task="segment", device="cuda", conf=0.25, save=False)

    # Wrist keypoints
    wrist_keypoints = []
    for result in pose_results:
        if result.keypoints is not None:
            for kp in result.keypoints.xy:
                if len(kp) > max(WRIST_INDEXES):
                    wrist_keypoints.append((int(kp[9][0]), int(kp[9][1])))
                    wrist_keypoints.append((int(kp[10][0]), int(kp[10][1])))

    # Segmentation outputs
    glove_masks, no_glove_masks = [], []
    for result in seg_results:
        if result.masks is not None:
            for mask, cls in zip(result.masks.xy, result.boxes.cls):
                mask_pts = mask.astype(np.int32)
                if int(cls) == 0:
                    glove_masks.append(mask_pts)
                else:
                    no_glove_masks.append(mask_pts)

    valid_glove_masks = [mask for mask in glove_masks if any(is_wrist_near_mask(w, mask) for w in wrist_keypoints)]
    valid_no_glove_masks = [mask for mask in no_glove_masks if any(is_wrist_near_mask(w, mask) for w in wrist_keypoints)]

    matched_gt_gloves = [False] * len(gt_gloves)
    matched_gt_nogloves = [False] * len(gt_no_gloves)

    # Gloves matching
    for mask in valid_glove_masks:
        matched = False
        for i, gt_mask in enumerate(gt_gloves):
            if not matched_gt_gloves[i] and calculate_iou(mask, gt_mask) > IOU_THRESHOLD:
                matched_gt_gloves[i] = True
                global_tp_gloves += 1
                matched = True
                break
        if not matched:
            global_fp_gloves += 1
    global_fn_gloves += matched_gt_gloves.count(False)

    # No-gloves matching
    for mask in valid_no_glove_masks:
        matched = False
        for i, gt_mask in enumerate(gt_no_gloves):
            if not matched_gt_nogloves[i] and calculate_iou(mask, gt_mask) > IOU_THRESHOLD:
                matched_gt_nogloves[i] = True
                global_tp_nogloves += 1
                matched = True
                break
        if not matched:
            global_fp_nogloves += 1
    global_fn_nogloves += matched_gt_nogloves.count(False)

    # Save visualization
    unmatched_gt_gloves = [gt_gloves[i] for i, m in enumerate(matched_gt_gloves) if not m]
    unmatched_gt_nogloves = [gt_no_gloves[i] for i, m in enumerate(matched_gt_nogloves) if not m]
    visual_img = draw_results(image, wrist_keypoints, gt_gloves, gt_no_gloves,
                              valid_glove_masks, valid_no_glove_masks,
                              unmatched_gt_gloves, unmatched_gt_nogloves)
    cv2.imwrite(os.path.join(save_folder, img_file), visual_img)

# Final Metrics Output
calculate_metrics()
