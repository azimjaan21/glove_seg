from ultralytics import YOLO

model = YOLO('weights/yolo11s-seg.pt')
model.train(
    data='configs/yolo11s_seg.py',
    epochs=100,
    imgsz=640,
    batch=16,
    project='results/yolo11s',
    verbose=True,
    name='run'
)
