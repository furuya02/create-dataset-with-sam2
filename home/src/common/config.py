from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class Config:
    """設定クラス"""

    sam2_home: str = "/segment-anything-2"
    home: str = "/src"
    checkpoint: str = None
    config: str = "sam2_hiera_t.yaml"
    output_path: str = None
    target_frame_idx: int = 0
    target_object_id: int = 0
    preview_window_pos: Tuple[int, int] = (500, 200)

    # Jetson環境でのSegmentation fault対策
    disable_opencv_display: bool = True  # OpenCV表示を無効化
    enable_headless_mode: bool = True  # ヘッドレスモード
    batch_size: int = 1  # バッチサイズを最小に

    def __post_init__(self):
        if self.checkpoint is None:
            self.checkpoint = f"{self.sam2_home}/checkpoints/sam2_hiera_tiny.pt"
        if self.output_path is None:
            self.output_path = f"{self.home}/dataset/annotation/"


@dataclass
class VideoConfig:
    """動画処理設定クラス"""

    video_path: str
    scale: float = 0.8
    fps: int = 10
    base_name: str = None
    source_path: str = None
    frame_path: str = None
    preview_path: str = None
    png_path: str = None
    txt_path: str = None
    segment_path: str = None

    def __post_init__(self):
        if self.base_name is None:
            self.base_name = Path(self.video_path).stem

    def setup_paths(self, output_path: str):
        """各種パスを設定"""
        self.source_path = f"{output_path}/{self.base_name}"
        self.frame_path = f"{self.source_path}/frame"
        self.preview_path = f"{self.source_path}/preview"
        self.png_path = f"{self.source_path}/png"
        self.txt_path = f"{self.source_path}/txt"
        self.segment_path = f"{self.source_path}/seg"
