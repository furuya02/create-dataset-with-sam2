"""
annotationで切り出した画像データと背景を組み合わせてSegmentation用のデータセットを作成する
"""

import glob
import random
import os
import shutil
import numpy as np
import cv2
from PIL import Image


def get_class_names_from_annotation():
    """
    dataset/annotation配下のフォルダ名を取得してクラス名として返す

    Returns:
        list: クラス名のリスト
    """
    annotation_path = "./dataset/annotation"
    if not os.path.exists(annotation_path):
        print(
            f"警告: {annotation_path} が見つかりません。デフォルトクラス名を使用します。"
        )
        return ["TOMATO", "DUCK"]

    class_names = []
    for item in os.listdir(annotation_path):
        item_path = os.path.join(annotation_path, item)
        if os.path.isdir(item_path):
            class_names.append(item)

    if not class_names:
        print(
            "警告: annotationフォルダにクラスフォルダが見つかりません。デフォルトクラス名を使用します。"
        )
        return ["TOMATO", "DUCK"]

    class_names.sort()  # アルファベット順にソート
    print(f"検出されたクラス: {class_names}")
    return class_names


def generate_colors(num_classes):
    """
    クラス数に応じてカラーリストを生成する

    Args:
        num_classes (int): クラス数

    Returns:
        list: BGR形式のカラータプルのリスト
    """
    colors = [
        (0, 0, 175),  # 赤
        (175, 0, 0),  # 青
        (0, 175, 0),  # 緑
        (175, 175, 0),  # シアン
        (175, 0, 175),  # マゼンタ
        (0, 175, 175),  # 黄色
        (100, 50, 200),  # 紫
        (200, 100, 50),  # オレンジ
        (50, 200, 100),  # 薄緑
        (100, 200, 200),  # 薄青
    ]

    # 必要な数だけ色を返す（不足する場合は繰り返し）
    return [colors[i % len(colors)] for i in range(num_classes)]


# クラス名を自動取得
CLASS_NAME = get_class_names_from_annotation()
# クラス数に応じて色を生成
COLORS = generate_colors(len(CLASS_NAME))

MAX = 3000  # 生成する画像数

BACKGROUND_IMAGE_PATH = "./dataset/background"
TARGET_IMAGE_PATH = "./dataset/annotation"
OUTPUT_PATH = "./dataset/merge"


BASE_WIDTH = 200  # 商品の基本サイズは、背景画像とのバランスより、横幅を200を基準とする
BACK_WIDTH = 640  # 背景画像ファイルのサイズを合わせる必要がある
BACK_HEIGHT = 480  # 背景画像ファイルのサイズを合わせる必要がある


# 背景画像取得クラス
class Background:
    def __init__(self, backPath):
        self.__backPath = backPath

    def get(self):
        imagePath = random.choice(glob.glob(self.__backPath + "/*.png"))
        return cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)


# 検出対象取得クラス (base_widthで指定された横幅を基準にリサイズされる)
class Target:
    def __init__(self, target_path, base_width, class_name):
        self.__target_path = target_path
        self.__base_width = base_width
        self.__class_name = class_name

    def get(self, class_id):
        # 切り出し画像
        class_name = self.__class_name[class_id]
        image_path = random.choice(
            glob.glob(self.__target_path + "/" + class_name + "/png/*.png")
        )
        target_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # 基準（横）サイズに基づきリサイズされる
        h, w, _ = target_image.shape
        aspect = h / w
        target_image = cv2.resize(
            target_image, (int(self.__base_width * aspect), self.__base_width)
        )
        bai = self.__base_width / w

        # labelも基準サイズへのリサイズに併せて、変換する( x * bai )
        target_label = ""

        label_text = ""
        label_path = image_path.replace("/png/", "/seg/").replace(".png", ".txt")
        with open(label_path, encoding="utf-8") as f:
            label_text = f.read()

        orign_label_list = label_text.split(",")
        for label in orign_label_list:
            if label != "":
                target_label += "{},".format(float(label) * bai)

        return target_image, target_label


# 変換クラス
class Transformer:
    def __init__(self, width, height):
        self.__width = width
        self.__height = height
        self.__min_scale = 0.3
        self.__max_scale = 1

    def warp(self, target_image):
        # サイズ変更
        target_image, scale = self.__resize(target_image)

        # 配置位置決定
        h, w, _ = target_image.shape
        left = random.randint(0, self.__width - w)
        top = random.randint(0, self.__height - h)
        rect = ((left, top), (left + w, top + h))

        # 背景面との合成
        new_image = self.__synthesize(target_image, left, top)
        return (new_image, rect, scale)

    def __resize(self, img):
        scale = random.uniform(self.__min_scale, self.__max_scale)
        w, h, _ = img.shape
        return cv2.resize(img, (int(w * scale), int(h * scale))), scale

    def __rote(self, target_image, angle):
        h, w, _ = target_image.shape
        rate = h / w
        scale = 1
        if rate < 0.9 or 1.1 < rate:
            scale = 0.9
        elif rate < 0.8 or 1.2 < rate:
            scale = 0.6
        center = (int(w / 2), int(h / 2))
        trans = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(target_image, trans, (w, h))

    def __synthesize(self, target_image, left, top):
        background_image = np.zeros((self.__height, self.__width, 4), np.uint8)
        back_pil = Image.fromarray(background_image)
        front_pil = Image.fromarray(target_image)
        back_pil.paste(front_pil, (left, top), front_pil)
        return np.array(back_pil)


class Effecter:

    # Gauss
    def gauss(self, img, level):
        return cv2.blur(img, (level * 2 + 1, level * 2 + 1))

    # Noise
    def noise(self, img):
        img = img.astype("float64")
        img[:, :, 0] = self.__single_channel_noise(img[:, :, 0])
        img[:, :, 1] = self.__single_channel_noise(img[:, :, 1])
        img[:, :, 2] = self.__single_channel_noise(img[:, :, 2])
        return img.astype("uint8")

    def __single_channel_noise(self, single):
        diff = 255 - single.max()
        noise = np.random.normal(0, random.randint(1, 100), single.shape)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = diff * noise
        noise = noise.astype(np.uint8)
        dst = single + noise
        return dst


# バウンディングボックス描画
def box(frame, rect, class_id):
    ((x1, y1), (x2, y2)) = rect
    label = "{}".format(CLASS_NAME[class_id])
    img = cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[class_id], 2)
    img = cv2.rectangle(img, (x1, y1), (x1 + 150, y1 - 20), COLORS[class_id], -1)
    cv2.putText(
        img,
        label,
        (x1 + 2, y1 - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return img


# 背景と商品の合成
def marge_image(background_image, front_image):
    back_pil = Image.fromarray(background_image)
    front_pil = Image.fromarray(front_image)
    back_pil.paste(front_pil, (0, 0), front_pil)
    return np.array(back_pil)


# 1画像分のデータを保持するクラス
class MergeData:
    def __init__(self, rate):
        self.__rects = []
        self.__images = []
        self.__class_ids = []
        self.__rate = rate

    def get_class_ids(self):
        return self.__class_ids

    def max(self):
        return len(self.__rects)

    def get(self, i):
        return (self.__images[i], self.__rects[i], self.__class_ids[i])

    # 追加（重複率が指定値以上の場合は失敗する）
    def append(self, target_image, rect, class_id):
        conflict = False
        for i in range(len(self.__rects)):
            iou = self.__multiplicity(self.__rects[i], rect)
            if iou > self.__rate:
                conflict = True
                break
        if conflict == False:
            self.__rects.append(rect)
            self.__images.append(target_image)
            self.__class_ids.append(class_id)
            return True
        return False

    # 重複率
    def __multiplicity(self, a, b):
        (ax_mn, ay_mn) = a[0]
        (ax_mx, ay_mx) = a[1]
        (bx_mn, by_mn) = b[0]
        (bx_mx, by_mx) = b[1]
        a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
        b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)
        abx_mn = max(ax_mn, bx_mn)
        aby_mn = max(ay_mn, by_mn)
        abx_mx = min(ax_mx, bx_mx)
        aby_mx = min(ay_mx, by_mx)
        w = max(0, abx_mx - abx_mn + 1)
        h = max(0, aby_mx - aby_mn + 1)
        intersect = w * h
        return intersect / (a_area + b_area - intersect)


# 各クラスのデータ数が同一になるようにカウントする
class Counter:
    def __init__(self, max):
        self.__counter = np.zeros(max)

    def get(self):
        n = np.argmin(self.__counter)
        return int(n)

    def inc(self, index):
        self.__counter[index] += 1

    def print(self):
        print(self.__counter)


def main():

    # 出力先の初期化
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.mkdir(OUTPUT_PATH)

    target = Target(TARGET_IMAGE_PATH, BASE_WIDTH, CLASS_NAME)
    background = Background(BACKGROUND_IMAGE_PATH)

    transformer = Transformer(BACK_WIDTH, BACK_HEIGHT)
    # manifest = Manifest(CLASS_NAME)
    counter = Counter(len(CLASS_NAME))
    effecter = Effecter()

    no = 0

    while True:
        # 背景画像の取得
        background_image = background.get()
        height, width, _ = background_image.shape
        # mergeデータ
        merge_data = MergeData(0.1)
        label_seg = ""
        label_od = ""
        for _ in range(20):
            # 現時点で作成数の少ないクラスIDを取得
            class_id = counter.get()
            # 切り出しデータの取得
            target_image, targhet_label = target.get(class_id)
            # 変換
            (transform_image, rect, scale) = transformer.warp(target_image)
            frame = marge_image(background_image, transform_image)

            # 商品の追加（重複した場合は、失敗する）
            ret = merge_data.append(transform_image, rect, class_id)
            if ret:
                counter.inc(class_id)

                # Segmentpoint
                label_seg += "{}".format(class_id)
                for i, l in enumerate(targhet_label.split(",")):
                    if l != "":
                        # 変換したscaleに併せてLabelも変換する
                        z = int(float(l) * scale)
                        # 貼り付け位置へシフトと、全体画像を基準としてた0..1の正規化
                        if i % 2 == 0:
                            # X座標
                            x = z + rect[0][0]
                            label_seg += " {:.5f}".format(x / width)
                        else:
                            # Y座標
                            y = z + rect[0][1]
                            label_seg += " {:.5f}".format(y / height)
                label_seg += "\n"

                # Object Detection (YOLO形式: クラスID 中心X 中心Y 幅 高さ)
                ((x1, y1), (x2, y2)) = rect

                # バウンディングボックスの中心座標と幅・高さを計算
                center_x = (x1 + x2) / 2.0 / width  # 正規化された中心X座標
                center_y = (y1 + y2) / 2.0 / height  # 正規化された中心Y座標
                bbox_width = (x2 - x1) / width  # 正規化された幅
                bbox_height = (y2 - y1) / height  # 正規化された高さ

                # YOLO形式でラベルを追加
                label_od += "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    class_id, center_x, center_y, bbox_width, bbox_height
                )

        print("max:{}".format(merge_data.max()))
        frame = background_image
        for index in range(merge_data.max()):
            (target_image, _, _) = merge_data.get(index)
            # 合成
            frame = marge_image(frame, target_image)

        # アルファチャンネル削除
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # エフェクト
        frame = effecter.gauss(frame, random.randint(0, 2))
        frame = effecter.noise(frame)

        # 画像名
        png_file_name = "{:05d}.png".format(no)
        # Segmentpoint
        seg_file_name = "{:05d}.seg".format(no)
        # Object Detection
        od_file_name = "{:05d}.od".format(no)

        no += 1

        # 画像保存
        cv2.imwrite("{}/{}".format(OUTPUT_PATH, png_file_name), frame)

        # テキスト保存
        with open("{}/{}".format(OUTPUT_PATH, seg_file_name), mode="w") as f:
            f.write(label_seg)
        with open("{}/{}".format(OUTPUT_PATH, od_file_name), mode="w") as f:
            f.write(label_od)

        # manifest追加
        # manifest.appned(fileName, merge_data, frame.shape[0], frame.shape[1])

        for i in range(merge_data.max()):
            (_, rect, class_id) = merge_data.get(i)
            # バウンディングボックス描画（確認用）
            frame = box(frame, rect, class_id)

        counter.print()
        print("no:{}".format(no))
        if MAX <= no:
            break

        # 表示（確認用）
        cv2.imshow("frame", frame)
        cv2.waitKey(1)


main()
