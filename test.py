from ultralytics import YOLO
import os

# Build a YOLOv9c model from pretrained weight
model = YOLO("train/PTQ_224_416_new/best_int8.tflite")


# # Run inference with the YOLOv9c model on the 'bus.jpg' image
results = model("Test/Video/test.mp4", show = True, conf=0.5, imgsz=(224,416))