import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "./best.pt"
CONF_THRESHOLD = 0.80
IOU_THRESHOLD = 0.5
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

COLORS = [
    (0, 0, 200),
    (0, 200, 0),
    (200, 0, 0),
    (200, 200, 0),
    (0, 200, 200),
    (200, 0, 200),
]

LINE_WIDTH = 5
FONT_SCALE = 1.5
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
THICKNESS = 5


def draw_label(box, img, color, label, line_thickness=3):
    x1, y1, x2, y2 = map(int, box)
    text_size = cv2.getTextSize(
        label, 0, fontScale=FONT_SCALE, thickness=line_thickness
    )[0]
    cv2.rectangle(img, (x1, y1), (x1 + text_size[0], y1 - text_size[1] - 2), color, -1)
    cv2.putText(
        img,
        label,
        (x1, y1 - 3),
        FONT_FACE,
        FONT_SCALE,
        [225, 255, 255],
        thickness=line_thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.rectangle(img, (x1, y1), (x2, y2), color, LINE_WIDTH)


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise IOError("カメラが開けません")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    ret, frame = cap.read()
    if not ret or frame is None:
        raise IOError("カメラから画像が取得できません")
    print(f"frame.shape: {frame.shape}")
    print(f"Camera resolution is sufficient: {frame.shape}")

    while True:
        try:
            ret, img = cap.read()
            if ret is False:
                raise IOError("カメラから画像が取得できません")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = model(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            names = results[0].names
            classes = results[0].boxes.cls
            confs = results[0].boxes.conf
            boxes = results[0].boxes
            for box, cls, conf in zip(boxes, classes, confs):
                if conf < CONF_THRESHOLD:
                    continue
                name = names[int(cls)]
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                cv2.rectangle(
                    img, (x1, y1), (x2, y2), COLORS[int(cls) % len(COLORS)], 2
                )
                cv2.putText(
                    img,
                    f"{name} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COLORS[int(cls) % len(COLORS)],
                    2,
                )
            image_resized = cv2.resize(img, (FRAME_WIDTH * 2, FRAME_HEIGHT * 2))
            cv2.imshow("YOLO", image_resized)
            cv2.waitKey(1)
        except KeyboardInterrupt:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
