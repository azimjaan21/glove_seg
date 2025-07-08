from ultralytics import YOLO

def main():
    model = YOLO('weights/yolo11s-seg.pt')
    model.train(
        data=r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\glove_seg\data\yolo\gloves.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        workers=2,
        project='results/yolo11s',
        verbose=True,
        name='run'
    )

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support() 
    main()
