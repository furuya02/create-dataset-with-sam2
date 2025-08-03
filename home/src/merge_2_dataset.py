"""
mergeデータをyolo形式に変換する
"""

import glob
import os
import shutil
import numpy as np
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
        return ["object"]

    class_names = []
    for item in os.listdir(annotation_path):
        item_path = os.path.join(annotation_path, item)
        if os.path.isdir(item_path):
            class_names.append(item)

    if not class_names:
        print(
            "警告: annotationフォルダにクラスフォルダが見つかりません。デフォルトクラス名を使用します。"
        )
        return ["object"]

    class_names.sort()  # アルファベット順にソート
    print(f"検出されたクラス: {class_names}")
    return class_names


def create_data_yaml(output_path, class_names=None):
    """
    YOLO形式のdata.yamlファイルを生成する

    Args:
        output_path (str): 出力ディレクトリのパス
        class_names (list): クラス名のリスト（Noneの場合はannotationフォルダから自動取得）
    """
    if class_names is None:
        class_names = get_class_names_from_annotation()

    # Docker環境用のパスを使用（/src/dataset/yolo_seg）
    docker_path = "/src/dataset/yolo_seg"

    yaml_content = f"""# YOLO dataset configuration
path: {docker_path}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')

# Classes
nc: {len(class_names)}  # number of classes
names: {class_names}  # class names
"""

    yaml_file_path = os.path.join(output_path, "data.yaml")
    with open(yaml_file_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"data.yaml ファイルを作成しました: {yaml_file_path}")
    print(f"クラス数: {len(class_names)}, クラス名: {class_names}")


def convert_merge_to_seg(input_path, output_path):
    # 出力先の初期化
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    stages = ["train", "val"]

    train_path = "{}/train".format(output_path)
    train_images_path = "{}/images".format(train_path)
    train_labels_path = "{}/labels".format(train_path)

    val_path = "{}/val".format(output_path)
    val_images_path = "{}/images".format(val_path)
    val_labels_path = "{}/labels".format(val_path)

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)

    os.makedirs(val_path, exist_ok=True)
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)

    image_files = glob.glob("{}/*.png".format(input_path))
    for i, input_image_file in enumerate(image_files):
        basename = os.path.splitext(os.path.basename(input_image_file))[0]
        if i % 10 < 8:
            stage = stages[0]
        else:
            stage = stages[1]

        input_label_file = "{}/{}.seg".format(input_path, basename)
        output_label_file = "{}/{}/labels/{}.txt".format(output_path, stage, basename)
        output_image_file = "{}/{}/images/{}.png".format(output_path, stage, basename)
        shutil.copy(input_label_file, output_label_file)
        shutil.copy(input_image_file, output_image_file)

    # data.yamlファイルを生成
    create_data_yaml(output_path)


def convert_merge(input_path, output_path, ext):
    # 出力先の初期化
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    stages = ["train", "val"]

    train_path = "{}/train".format(output_path)
    train_images_path = "{}/images".format(train_path)
    train_labels_path = "{}/labels".format(train_path)

    val_path = "{}/val".format(output_path)
    val_images_path = "{}/images".format(val_path)
    val_labels_path = "{}/labels".format(val_path)

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)

    os.makedirs(val_path, exist_ok=True)
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)

    image_files = glob.glob("{}/*.png".format(input_path))
    for i, input_image_file in enumerate(image_files):
        basename = os.path.splitext(os.path.basename(input_image_file))[0]
        if i % 10 < 8:
            stage = stages[0]
        else:
            stage = stages[1]

        input_label_file = "{}/{}.{}".format(input_path, basename, ext)
        output_label_file = "{}/{}/labels/{}.txt".format(output_path, stage, basename)
        output_image_file = "{}/{}/images/{}.png".format(output_path, stage, basename)
        shutil.copy(input_label_file, output_label_file)
        shutil.copy(input_image_file, output_image_file)

    create_data_yaml(output_path)


def main():
    input_path = "./dataset/merge"
    output_seg_path = "./dataset/yolo_seg"
    output_od_path = "./dataset/yolo_od"

    convert_merge(input_path, output_seg_path, "seg")
    convert_merge(input_path, output_od_path, "od")

    print(f"データセット変換が完了しました: {output_seg_path}")


if __name__ == "__main__":
    main()
