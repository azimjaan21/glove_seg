import time
from ultralytics import YOLO
import cv2

model = YOLO("C:/Users/dalab/Desktop/azimjaan21/RESEARCH/glove_seg/results/yolov8s/run/weights/best.pt")
img = cv2.imread("demo.jpg")

# Warm-up
model.predict(source=img, imgsz=512, device=0, verbose=False)

# Timed inference
start = time.time()
for _ in range(100):
    model.predict(source=img, imgsz=512, device=0, verbose=False, save=False)
end = time.time()

fps = 100 / (end - start)
print(f"⚡ Real FPS: {fps:.2f}")
