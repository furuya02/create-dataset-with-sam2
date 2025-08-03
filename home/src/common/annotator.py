# SuperVisionを使用して、マスク画像を生成するためのクラス

import supervision as sv
import numpy as np
import cv2


class Annotator:
    def __init__(self):
        COLORS = [
            "#FF1493",
            "#FFD700",
            "#FF6347",
        ]
        self.mask_annotator = sv.MaskAnnotator(
            color=sv.ColorPalette.from_hex(COLORS), color_lookup=sv.ColorLookup.CLASS
        )

    def set_mask(self, frame_org, masks, object_ids):
        frame = frame_org.copy()
        masks = self.__ensure_3d_masks(masks)
        detections = self.__create_detections(masks, object_ids)
        frame = self.mask_annotator.annotate(scene=frame, detections=detections)
        frame = self.__annotate_with_rectangle(frame, detections)
        return frame

    def __ensure_3d_masks(self, masks):
        if masks.ndim == 2:
            masks = masks[np.newaxis, :, :]
        return masks

    def __create_detections(self, masks, object_ids):
        return sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks,
            class_id=np.array(object_ids),
        )

    def __annotate_with_rectangle(self, frame, detections):
        rect = detections[0].xyxy[0]
        frame = cv2.rectangle(
            frame,
            (int(rect[0]), int(rect[1])),
            (int(rect[2]), int(rect[3])),
            (0, 0, 255),
            2,
        )
        return frame
