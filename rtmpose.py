from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import mmcv
import os

# File paths
config_file = "configs/rtmpose-s_8xb256-420e_coco-256x192.py"
checkpoint_file = "rtmpose-s.pth"  # Make sure this matches your config
img_path = "demo.jpg"  # Replace with your image filename

# Check files exist
assert os.path.isfile(config_file), f"Config file not found: {config_file}"
assert os.path.isfile(checkpoint_file), f"Checkpoint file not found: {checkpoint_file}"
assert os.path.isfile(img_path), f"Image not found: {img_path}"

# Initialize RTMPose-S model
pose_model = init_model(config_file, checkpoint_file, device='cuda')

# Load image
image = mmcv.imread(img_path)

# Run pose inference
results = inference_topdown(pose_model, image)

# Merge results to get keypoints as ndarray
data_sample = merge_data_samples(results)
pred_instances = data_sample.pred_instances
keypoints = pred_instances.keypoints  # Already a NumPy array

# Extract wrist keypoints (COCO format: 9=left_wrist, 10=right_wrist)
WRIST_INDEXES = [9, 10]
wrist_keypoints = []
for kp in keypoints:
    person_wrists = []
    for idx in WRIST_INDEXES:
        values = kp[idx]
        if len(values) == 3:
            x, y, score = values
            if score > 0.3:
                person_wrists.append((int(x), int(y)))
            else:
                person_wrists.append(None)
        elif len(values) == 2:
            x, y = values
            person_wrists.append((int(x), int(y)))
        else:
            person_wrists.append(None)
    wrist_keypoints.append(person_wrists)


print("Wrist keypoints:", wrist_keypoints)
