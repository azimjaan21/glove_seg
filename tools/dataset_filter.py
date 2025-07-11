

import os

# Set paths
images_dir = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\glove_seg\eval_dataset\multimodal\valid_img"
labels_dir = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\glove_seg\eval_dataset\multimodal\valid\labels"

# Get base filenames
image_basenames = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
label_basenames = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt')}

# Find the unmatched one(s)
unmatched_labels = label_basenames - image_basenames

# Print and delete
for base in unmatched_labels:
    txt_path = os.path.join(labels_dir, base + ".txt")
    print(f"🛑 Unmatched label: {txt_path}")
    if os.path.exists(txt_path):
        os.remove(txt_path)
        print(f"✅ Deleted: {txt_path}")

print(f"\nFinished. {len(unmatched_labels)} label file(s) removed.")
