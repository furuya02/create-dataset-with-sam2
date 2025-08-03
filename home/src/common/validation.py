import os


def validate_video_file(video_path: str):
    """動画ファイルの有効性を検証"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise ValueError(f"Unsupported video format: {video_path}")