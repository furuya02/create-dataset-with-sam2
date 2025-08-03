from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

model.train(data="./dataset/yolo_seg/data.yaml", epochs=10, batch=8, workers=4)
