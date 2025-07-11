from ultralytics import YOLO

def main():
    model = YOLO("C:/Users/dalab/Desktop/azimjaan21/RESEARCH/glove_seg/results/yolov8s/run/weights/best.pt")
    metrics = model.val(data= r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\glove_seg\eval_dataset\yolo\valid\eval_yolo.yaml")

    print(f"mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()