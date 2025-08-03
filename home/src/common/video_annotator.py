import logging
from typing import Tuple, List
import cv2
import torch
import numpy as np
import gc
import os
from sam2.build_sam import build_sam2_video_predictor
from common.util import init_dir, get_video_specifications
from common.annotator import Annotator


class VideoAnnotator:
    """動画アノテーション処理クラス"""

    def __init__(self, config, video_config):
        self.config = config
        self.video_config = video_config
        self.logger = logging.getLogger(__name__)
        self.sam2_model = None
        self.inference_state = None

        # Jetson AGX Orin最適化設定
        self._setup_jetson_optimizations()

    def _setup_jetson_optimizations(self):
        """Jetson環境でのメモリ最適化設定"""
        # CUDA関連設定
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        # CPU使用制限
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"

        # PyTorchメモリ設定
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # CUDAメモリ使用量を制限
            torch.cuda.set_per_process_memory_fraction(0.7)  # 70%に制限

    def setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def print_config(self):
        """設定情報を表示"""
        self.logger.info(f"VIDEO_PATH: {self.video_config.video_path}")
        self.logger.info(f"OUTPUT_PATH: {self.config.output_path}")
        self.logger.info(f"SOURCE_PATH: {self.video_config.source_path}")
        self.logger.info(f"CHECKPOINT: {self.config.checkpoint}")
        self.logger.info(f"CONFIG: {self.config.config}")

    def get_video_specs(self) -> Tuple[int, int, int, float]:
        """動画の仕様を取得"""
        try:
            return get_video_specifications(self.video_config.video_path)
        except Exception as e:
            self.logger.error(f"Failed to get video specifications: {e}")
            raise

    def initialize_directories(self):
        """ディレクトリを初期化"""
        try:
            init_dir(
                self.config.output_path,
                self.video_config.source_path,
                self.video_config.frame_path,
                self.video_config.png_path,
                self.video_config.txt_path,
                self.video_config.segment_path,
                self.video_config.preview_path,
            )
            self.logger.info("Directories initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize directories: {e}")
            raise

    def setup_device(self) -> torch.device:
        """デバイスを設定"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {device}")
        return device

    def setup_sam2_model(self):
        """SAM2モデルを初期化"""
        try:
            # メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            self.sam2_model = build_sam2_video_predictor(
                self.config.config, self.config.checkpoint
            )

            # 推論状態の初期化前にメモリクリーンアップ
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.inference_state = self.sam2_model.init_state(
                video_path=self.video_config.frame_path
            )
            self.sam2_model.reset_state(self.inference_state)
            self.logger.info("SAM2 model initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize SAM2 model: {e}")
            # エラー時もメモリクリーンアップ
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def process_preview_frame(
        self,
        frame_idx: int,
        masks: np.ndarray,
        object_ids: List,
        annotator: Annotator,
        source_frames: List[str],
    ):
        """プレビューフレームを処理（Segmentation fault対策版）"""
        frame = None
        annotated_frame = None

        try:
            # メモリ事前クリーンアップ
            gc.collect()

            frame_path = source_frames[frame_idx]
            frame = cv2.imread(frame_path)

            if frame is None:
                self.logger.warning(f"Failed to load frame: {frame_path}")
                return

            # アノテーション処理
            annotated_frame = annotator.set_mask(frame, masks, object_ids)

            # ファイル保存
            output_path = f"{self.video_config.preview_path}/{self.video_config.base_name}_{frame_idx:05}.png"
            success = cv2.imwrite(output_path, annotated_frame)

            if not success:
                self.logger.warning(f"Failed to save frame: {output_path}")

            # OpenCV表示を条件付きで実行（Segmentation fault対策）
            if not getattr(self.config, "disable_opencv_display", True):
                try:
                    cv2.imshow("annotated_frame", annotated_frame)
                    if frame_idx == 0:
                        cv2.moveWindow(
                            "annotated_frame", *self.config.preview_window_pos
                        )
                    cv2.waitKey(1)
                except Exception as display_error:
                    self.logger.warning(
                        f"Display error (non-critical): {display_error}"
                    )

        except Exception as e:
            self.logger.error(
                f"Error in process_preview_frame at frame {frame_idx}: {e}"
            )

        finally:
            # 確実なメモリクリーンアップ
            try:
                if frame is not None:
                    del frame
                if annotated_frame is not None:
                    del annotated_frame
                gc.collect()

                # 定期的なCUDAメモリクリーンアップ
                if frame_idx % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as cleanup_error:
                self.logger.warning(f"Cleanup error: {cleanup_error}")
