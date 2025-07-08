from ultralytics import YOLO

def main():
    model = YOLO('weights/yolov8s-seg.pt')
    model.train(
        data=r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\glove_seg\data\yolo\gloves.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        project='results/yolov8s',
        verbose=True,
        workers=2,
        name='run'
    )

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support() 
    main()
