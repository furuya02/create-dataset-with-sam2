from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="./dataset/yolo_od/data.yaml", epochs=10, batch=8, workers=4)
