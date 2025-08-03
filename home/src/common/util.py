import os
import shutil
import glob
import cv2
import torch


# 作業ディレクトリの初期化
def init_dir(
    output_path,
    source_path,
    frame_path,
    png_path,
    txt_path,
    segment_path,
    annotation_path,
):
    paths = [
        output_path,
        source_path,
        frame_path,
        png_path,
        txt_path,
        segment_path,
        annotation_path,
    ]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


# 動画の諸元（フレーム数、幅、高さ、FPS）を取得する
def get_video_specifications(video_path):
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return (count, width, height, fps)


# 動画からJPEG画層を切り出す
def create_frames_using_ffmpeg(video_path, output_path, frame_width, frame_height, fps):
    os.system(
        f"ffmpeg -i {video_path} -r {fps} -q:v 2 -vf scale={frame_width}:{frame_height} -start_number 0 {output_path}/'%05d.jpeg'"
    )
    return sorted(glob.glob(f"{output_path}/*.jpeg"))


# cuDNN 畳み込みで TensorFloat-32 テンソル コアを使用できる場所を制御する
# https://pytorch.org/docs/2.1/notes/cuda.html#tf32-on-ampere
def allow_tf32():

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
