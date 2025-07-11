import cv2
import os
import numpy as np
from ultralytics import YOLO
import time

# Global Counters for overall metrics
global_tp_gloves_raw = 0
global_fp_gloves_raw = 0
global_fn_gloves_raw = 0
global_tp_nogloves_raw = 0
global_fp_nogloves_raw = 0
global_fn_nogloves_raw = 0

global_tp_gloves_fused = 0
global_fp_gloves_fused = 0
global_fn_gloves_fused = 0
global_tp_nogloves_fused = 0
global_fp_nogloves_fused = 0
global_fn_nogloves_fused = 0

# Additional counters for wrist failures (TP masks lost due to wrist anchoring)
global_wrist_fail_gloves = 0
global_wrist_fail_nogloves = 0

# Constants
WRIST_INDEXES = [9, 10]
DISTANCE_THRESHOLD = 40
IOU_THRESHOLD = 0.5

# Paths
dataset_path = "C:/Users/dalab/Desktop/azimjaan21/RESEARCH/glove_seg/eval_dataset/multimodal"
image_folder = os.path.join(dataset_path, "images")
label_folder = os.path.join(dataset_path, "labels")
save_folder = "multimodal_fusion_experiments/exp1_yolov8s_seg_pose8"
os.makedirs(save_folder, exist_ok=True)

success_folder = os.path.join(save_folder, "wrist_success")
failure_folder = os.path.join(save_folder, "wrist_failure")
os.makedirs(success_folder, exist_ok=True)
os.makedirs(failure_folder, exist_ok=True)

# Load Models
pose_model = YOLO("weights/yolov8s-pose.pt")
segmentation_model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\glove_seg\results\yolov8s\run\weights\best.pt")

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

def print_metrics(tp, fp, fn, label):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    print(f"=== {label} ===")
    print(f"TP: {tp} | FP: {fp} | FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print()

def calculate_and_print_all_metrics():
    print("="*60)
    print("FINAL EVALUATION METRICS (RAW SEGMENTATION PREDICTIONS)")
    print("="*60)
    print_metrics(global_tp_gloves_raw, global_fp_gloves_raw, global_fn_gloves_raw, "Gloves")
    print_metrics(global_tp_nogloves_raw, global_fp_nogloves_raw, global_fn_nogloves_raw, "No Gloves")
    print(f"Wrist Failures (Gloves): {global_wrist_fail_gloves}")
    print(f"Wrist Failures (No Gloves): {global_wrist_fail_nogloves}")
    print("\n" + "="*60)
    print("FINAL EVALUATION METRICS (AFTER WRIST ANCHORING FILTERING - FUSION)")
    print("="*60)
    print_metrics(global_tp_gloves_fused, global_fp_gloves_fused, global_fn_gloves_fused, "Gloves")
    print_metrics(global_tp_nogloves_fused, global_fp_nogloves_fused, global_fn_nogloves_fused, "No Gloves")
    print("="*60)

# ------------------ Main Evaluation Loop ------------------ #

total_time = 0
total_images = 0

for img_file in os.listdir(image_folder):
    if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(image_folder, img_file)
    image = cv2.imread(img_path)
    height, width, _ = image.shape

    label_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + '.txt')
    gt_gloves, gt_no_gloves = parse_yolo_seg_label(label_path, width, height)

    start_time = time.time()

    pose_results = pose_model.predict(img_path, task="pose", device="cuda", conf=0.25, save=False)
    seg_results = segmentation_model.predict(img_path, task="segment", device="cuda", conf=0.25, save=False)

    end_time = time.time()
    total_time += (end_time - start_time)
    total_images += 1

    # Extract wrist keypoints (both wrists)
    wrist_keypoints = []
    for result in pose_results:
        if result.keypoints is not None:
            for kp in result.keypoints.xy:
                if len(kp) > max(WRIST_INDEXES):
                    wrist_keypoints.append((int(kp[9][0]), int(kp[9][1])))
                    wrist_keypoints.append((int(kp[10][0]), int(kp[10][1])))

    # Extract segmentation masks per class (all predicted masks)
    glove_masks, no_glove_masks = [], []
    for result in seg_results:
        if result.masks is not None:
            for mask, cls in zip(result.masks.xy, result.boxes.cls):
                mask_pts = mask.astype(np.int32)
                if int(cls) == 0:
                    glove_masks.append(mask_pts)
                else:
                    no_glove_masks.append(mask_pts)

    # === Step 1: Metrics on raw segmentation predictions (before wrist filtering) ===
    matched_gt_gloves_raw = [False] * len(gt_gloves)
    matched_pred_gloves_raw_idx = set()

    for i_pred, mask_pred in enumerate(glove_masks):
        for i_gt, mask_gt in enumerate(gt_gloves):
            if not matched_gt_gloves_raw[i_gt] and calculate_iou(mask_pred, mask_gt) > IOU_THRESHOLD:
                matched_gt_gloves_raw[i_gt] = True
                matched_pred_gloves_raw_idx.add(i_pred)
                break

    tp_gloves_raw = len(matched_pred_gloves_raw_idx)
    fp_gloves_raw = len(glove_masks) - tp_gloves_raw
    fn_gloves_raw = matched_gt_gloves_raw.count(False)

    matched_gt_nogloves_raw = [False] * len(gt_no_gloves)
    matched_pred_nogloves_raw_idx = set()

    for i_pred, mask_pred in enumerate(no_glove_masks):
        for i_gt, mask_gt in enumerate(gt_no_gloves):
            if not matched_gt_nogloves_raw[i_gt] and calculate_iou(mask_pred, mask_gt) > IOU_THRESHOLD:
                matched_gt_nogloves_raw[i_gt] = True
                matched_pred_nogloves_raw_idx.add(i_pred)
                break

    tp_nogloves_raw = len(matched_pred_nogloves_raw_idx)
    fp_nogloves_raw = len(no_glove_masks) - tp_nogloves_raw
    fn_nogloves_raw = matched_gt_nogloves_raw.count(False)

    # === Step 2: Wrist anchoring filter ===
    valid_glove_masks = []
    valid_no_glove_masks = []
    wrist_success_masks_glove = []
    wrist_success_masks_noglove = []

    for mask in glove_masks:
        near = any(is_wrist_near_mask(w, mask) for w in wrist_keypoints)
        if near:
            valid_glove_masks.append(mask)
        else:
            wrist_success_masks_glove.append(mask)

    for mask in no_glove_masks:
        near = any(is_wrist_near_mask(w, mask) for w in wrist_keypoints)
        if near:
            valid_no_glove_masks.append(mask)
        else:
            wrist_success_masks_noglove.append(mask)

    # === Step 3: Wrist failure counting (TP lost due to filtering) ===
    wrist_failed_gt_gloves = set()
    for i_gt, gt_mask in enumerate(gt_gloves):
        if matched_gt_gloves_raw[i_gt]:
            matched_pred_for_gt = [i for i, mask in enumerate(glove_masks) if calculate_iou(mask, gt_mask) > IOU_THRESHOLD]
            for idx in matched_pred_for_gt:
                found_in_valid = any(np.array_equal(glove_masks[idx], vm) for vm in valid_glove_masks)
                if not found_in_valid and i_gt not in wrist_failed_gt_gloves:
                    global_wrist_fail_gloves += 1
                    wrist_failed_gt_gloves.add(i_gt)
                    # Save wrist failure image (optional, comment if too noisy)
                    fail_img = draw_results(image.copy(), wrist_keypoints, gt_gloves, gt_no_gloves,
                                            [glove_masks[idx]], [], [], [])
                    cv2.imwrite(os.path.join(failure_folder, f"wrist_fail_glove_{img_file}"), fail_img)
                    break

    wrist_failed_gt_nogloves = set()
    for i_gt, gt_mask in enumerate(gt_no_gloves):
        if matched_gt_nogloves_raw[i_gt]:
            matched_pred_for_gt = [i for i, mask in enumerate(no_glove_masks) if calculate_iou(mask, gt_mask) > IOU_THRESHOLD]
            for idx in matched_pred_for_gt:
                found_in_valid = any(np.array_equal(no_glove_masks[idx], vm) for vm in valid_no_glove_masks)
                if not found_in_valid and i_gt not in wrist_failed_gt_nogloves:
                    global_wrist_fail_nogloves += 1
                    wrist_failed_gt_nogloves.add(i_gt)
                    fail_img = draw_results(image.copy(), wrist_keypoints, gt_gloves, gt_no_gloves,
                                            [], [no_glove_masks[idx]], [], [])
                    cv2.imwrite(os.path.join(failure_folder, f"wrist_fail_noglove_{img_file}"))
                    break

    # === Step 4: Calculate metrics after wrist filtering (fusion) ===
    matched_gt_gloves_fused = [False] * len(gt_gloves)
    matched_pred_gloves_fused_idx = set()

    for i_pred, mask_pred in enumerate(valid_glove_masks):
        for i_gt, mask_gt in enumerate(gt_gloves):
            if not matched_gt_gloves_fused[i_gt] and calculate_iou(mask_pred, mask_gt) > IOU_THRESHOLD:
                matched_gt_gloves_fused[i_gt] = True
                matched_pred_gloves_fused_idx.add(i_pred)
                break

    tp_gloves_fused = len(matched_pred_gloves_fused_idx)
    fp_gloves_fused = len(valid_glove_masks) - tp_gloves_fused
    fn_gloves_fused = matched_gt_gloves_fused.count(False)

    matched_gt_nogloves_fused = [False] * len(gt_no_gloves)
    matched_pred_nogloves_fused_idx = set()

    for i_pred, mask_pred in enumerate(valid_no_glove_masks):
        for i_gt, mask_gt in enumerate(gt_no_gloves):
            if not matched_gt_nogloves_fused[i_gt] and calculate_iou(mask_pred, mask_gt) > IOU_THRESHOLD:
                matched_gt_nogloves_fused[i_gt] = True
                matched_pred_nogloves_fused_idx.add(i_pred)
                break

    tp_nogloves_fused = len(matched_pred_nogloves_fused_idx)
    fp_nogloves_fused = len(valid_no_glove_masks) - tp_nogloves_fused
    fn_nogloves_fused = matched_gt_nogloves_fused.count(False)

    # === Step 5: Update global counters ===
    global_tp_gloves_raw += tp_gloves_raw
    global_fp_gloves_raw += fp_gloves_raw
    global_fn_gloves_raw += fn_gloves_raw

    global_tp_nogloves_raw += tp_nogloves_raw
    global_fp_nogloves_raw += fp_nogloves_raw
    global_fn_nogloves_raw += fn_nogloves_raw

    global_tp_gloves_fused += tp_gloves_fused
    global_fp_gloves_fused += fp_gloves_fused
    global_fn_gloves_fused += fn_gloves_fused

    global_tp_nogloves_fused += tp_nogloves_fused
    global_fp_nogloves_fused += fp_nogloves_fused
    global_fn_nogloves_fused += fn_nogloves_fused

    # === Step 6: Save wrist success false positives (filtered masks that don't match GT) ===
    for mask in wrist_success_masks_glove:
        matched_any = any(calculate_iou(mask, gt_mask) > IOU_THRESHOLD for gt_mask in gt_gloves)
        if not matched_any:
            success_img = draw_results(image.copy(), wrist_keypoints, gt_gloves, gt_no_gloves,
                                       [mask], [], [], [])
            cv2.imwrite(os.path.join(success_folder, f"wrist_success_fp_glove_{img_file}"), success_img)


    for mask in wrist_success_masks_noglove:
        matched_any = any(calculate_iou(mask, gt_mask) > IOU_THRESHOLD for gt_mask in gt_no_gloves)
        if not matched_any:
            success_img = draw_results(image.copy(), wrist_keypoints, gt_gloves, gt_no_gloves,
                                       [], [mask], [], [])
            cv2.imwrite(os.path.join(success_folder, f"wrist_success_fp_noglove_{img_file}"), success_img)

    # === Step 7: Save visualization for this image ===
    unmatched_gt_gloves_raw = [gt_gloves[i] for i, matched in enumerate(matched_gt_gloves_raw) if not matched]
    unmatched_gt_nogloves_raw = [gt_no_gloves[i] for i, matched in enumerate(matched_gt_nogloves_raw) if not matched]

    visual_img = draw_results(image, wrist_keypoints, gt_gloves, gt_no_gloves,
                              valid_glove_masks, valid_no_glove_masks,
                              unmatched_gt_gloves_raw, unmatched_gt_nogloves_raw)
    cv2.imwrite(os.path.join(save_folder, img_file), visual_img)

# Calculate and print FPS
fps = total_images / total_time if total_time > 0 else 0.0
print(f"\nProcessed {total_images} images in {total_time:.3f} seconds.")
print(f"Average FPS (pose + segmentation): {fps:.2f}")

# Final Metrics Output
calculate_and_print_all_metrics()
