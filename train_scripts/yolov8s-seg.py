from ultralytics import YOLO

model = YOLO('weights/yolov8s-seg.pt')
model.train(
    data='configs/yolov8s_seg.py',
    epochs=100,
    imgsz=640,
    batch=16,
    project='results/yolov8s',
    verbose=True,
    name='run'
)
