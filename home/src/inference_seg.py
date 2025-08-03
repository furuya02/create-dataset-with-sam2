import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "./best.pt"
CONF_THRESHOLD = 0.85
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


def overlay(image, mask, color, alpha, resize=None):
    """マスクを画像に重ねる"""
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()
    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)
    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    return image_combined


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
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise IOError("カメラから画像が取得できません")
            h, w, _ = frame.shape
            results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
            result = results[0]
            image = frame.copy()
            if result.masks is not None:
                for r in results:
                    boxes = r.boxes
                    conf_list = r.boxes.conf.tolist()
                for i, (seg, box) in enumerate(
                    zip(result.masks.data.cpu().numpy(), boxes)
                ):
                    seg = cv2.resize(seg, (w, h))
                    color = COLORS[int(box.cls) % len(COLORS)]
                    image = overlay(image, seg, color, 0.5)
                    class_id = int(box.cls)
                    box_xyxy = box.xyxy.tolist()[0]
                    class_name = result.names[class_id]
                    draw_label(
                        box_xyxy,
                        image,
                        color,
                        f"{class_name} {conf_list[i]:.2f}",
                        line_thickness=3,
                    )
            image_resized = cv2.resize(image, (FRAME_WIDTH * 2, FRAME_HEIGHT * 2))
            cv2.imshow("YOLO", image_resized)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
