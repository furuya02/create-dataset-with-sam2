import numpy as np
import cv2


class MaskUtil:
    def __init__(self, masks, frame):
        self.masks = masks
        self.frame = frame

        # マスクの範囲取得
        mask_indexes = np.where(masks)
        y_min = np.min(mask_indexes[0])
        y_max = np.max(mask_indexes[0])
        x_min = np.min(mask_indexes[1])
        x_max = np.max(mask_indexes[1])
        self.box = np.array([x_min, y_min, x_max, y_max])

        self.__create_segmentation_data()
        self.__create_transparent_image()

    # 透過画像の生成
    def __create_transparent_image(self):

        # boxの範囲で元画像を切り取る
        copy_image = self.frame.copy()
        white_back_image = copy_image[
            self.box[1] : self.box[3], self.box[0] : self.box[2]
        ]

        # boxの範囲でマスクを切り取る
        part_of_mask = self.masks[self.box[1] : self.box[3], self.box[0] : self.box[2]]

        transparent_image = cv2.cvtColor(white_back_image, cv2.COLOR_BGR2BGRA)
        transparent_image[np.logical_not(part_of_mask), 3] = 0
        self.transparent_image = transparent_image

    # セグメンテーション用のデータ生成
    def __create_segmentation_data(self):
        # 細かいノイズを除去するため。マスクの正規化（白黒２値画層を生成し、最大サイズの輪郭を再取得する）

        # マスクによって黒画像の上に白を描画する
        height, width, _ = self.frame.shape
        tmp_black_image = np.full(np.array([height, width, 1]), 0, dtype=np.uint8)
        tmp_white_image = np.full(np.array([height, width, 1]), 255, dtype=np.uint8)
        tmp_black_image[:] = np.where(
            self.masks[:height, :width, np.newaxis] == True,
            tmp_white_image,
            tmp_black_image,
        )

        # 輪郭の取得
        contours, _ = cv2.findContours(
            tmp_black_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 最も面積が大きい輪郭を選択
        max_contours = max(contours, key=lambda x: cv2.contourArea(x))
        # 黒画面に一番大きい輪郭だけ塗りつぶして描画する
        black_image = np.full(np.array([height, width, 1]), 0, dtype=np.uint8)
        black_image = cv2.drawContours(
            black_image, [max_contours], -1, color=255, thickness=-1
        )

        # boxの範囲で元画像を切り取る
        black_image = black_image[self.box[1] : self.box[3], self.box[0] : self.box[2]]

        # 輪郭を保存
        contours, _ = cv2.findContours(
            black_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        text = ""
        for data in contours[0].tolist():
            d = data[0]
            x = d[0]
            y = d[1]
            text += "{},{},".format(x, y)

        self.text = text
