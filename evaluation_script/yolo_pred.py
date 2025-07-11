from ultralytics import YOLO

def main():
    model = YOLO("C:/Users/dalab/Desktop/azimjaan21/RESEARCH/glove_seg/results/yolov8s/run/weights/best.pt")

    predict_results = model.predict(
        source="demo.jpg", 
        imgsz=640, 
        conf=0.25, 
        device=0, 
        save=True
    )

    print("✅ Inference complete. Results saved in 'runs/predict/'.")

if __name__ == "__main__":
    main()
