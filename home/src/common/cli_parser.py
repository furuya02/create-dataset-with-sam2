import argparse


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description="Video annotation with SAM2")
    parser.add_argument("video", help="Path to the video file (mp4)")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.8,
        help="Scale factor for frame resolution (default: 0.8)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frame rate for extracted frames (default: 10)",
    )
    return parser.parse_args()