import os
from ultralytics import YOLO
import glob

def main():
    # Load model
    model = YOLO("C:/Users/dalab/Desktop/azimjaan21/RESEARCH/glove_seg/results/yolo11s/run/weights/best.pt")

    # Get all image paths
    image_folder = "C:/Users/dalab/Desktop/azimjaan21/RESEARCH/glove_seg/eval_dataset/yolo/valid/images/"
    image_paths = glob.glob(os.path.join(image_folder, "*.*"))

    print(f"🔍 Found {len(image_paths)} images.")

    # Output folder will be: runs/predict/exp/
    for i, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            print(f"❌ Skipped missing file: {img_path}")
            continue

        try:
            results = model.predict(
                source=img_path,
                imgsz=640,
                conf=0.25,
                device=0,
                save=True,
                verbose=False
            )
            print(f"✅ {i+1}/{len(image_paths)} Processed: {os.path.basename(img_path)}")
        except Exception as e:
            print(f"⚠️ Error on {img_path}: {e}")
            continue

    print("\n🎯 Inference complete. Results saved in 'runs/predict/'.")

if __name__ == "__main__":
    main()
