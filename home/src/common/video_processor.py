import logging
from typing import List
import cv2
import numpy as np
from common.config import Config, VideoConfig
from common.video_annotator import VideoAnnotator
from common.mask_util import MaskUtil
from common.annotator import Annotator
from common.bounding_box import BoundingBox
from common.util import create_frames_using_ffmpeg, allow_tf32


class VideoProcessor:
    """動画処理のメインワークフローを管理するクラス"""

    def __init__(self, config: Config, video_config: VideoConfig):
        self.config = config
        self.video_config = video_config
        self.annotator_processor = VideoAnnotator(config, video_config)
        self.logger = logging.getLogger(__name__)

    def setup(self):
        """初期設定を実行"""
        # ログ設定
        self.annotator_processor.setup_logging()

        # 設定情報を表示
        self.annotator_processor.print_config()

        # ディレクトリを初期化
        self.annotator_processor.initialize_directories()

        # デバイスを設定
        device = self.annotator_processor.setup_device()
        allow_tf32()

        return device

    def get_video_specs(self):
        """動画の仕様を取得し、フレーム仕様を計算"""
        video_count, video_width, video_height, video_fps = (
            self.annotator_processor.get_video_specs()
        )

        self.logger.info(
            f"VIDEO_COUNT: {video_count}, VIDEO_WIDTH: {video_width}, "
            f"VIDEO_HEIGHT: {video_height}, VIDEO_FPS: {video_fps}"
        )

        # フレームの仕様を計算
        frame_width = int(video_width * self.video_config.scale)
        frame_height = int(video_height * self.video_config.scale)
        frame_skip = video_fps / self.video_config.fps

        self.logger.info(
            f"FRAME_WIDTH: {frame_width}, FRAME_HEIGHT: {frame_height}, "
            f"FRAME_SKIP: {frame_skip}"
        )

        return frame_width, frame_height, video_fps

    def generate_frames(self, frame_width: int, frame_height: int) -> List[str]:
        """フレームを生成"""
        source_frames = create_frames_using_ffmpeg(
            self.video_config.video_path,
            self.video_config.frame_path,
            frame_width,
            frame_height,
            self.video_config.fps,
        )
        self.logger.info(f"Generated {len(source_frames)} frames")
        return source_frames

    def setup_sam2_and_get_target(self, source_frames: List[str]):
        """SAM2モデルを設定し、ターゲットボックスを取得"""
        # ユーティリティクラスを初期化
        bounding_box = BoundingBox()

        # ターゲットのバウンディングボックスを取得
        target_box = bounding_box.get_box(source_frames[self.config.target_frame_idx])

        # SAM2モデルを初期化
        self.annotator_processor.setup_sam2_model()

        # プロンプトを追加
        _, object_ids, mask_logits = (
            self.annotator_processor.sam2_model.add_new_points_or_box(
                inference_state=self.annotator_processor.inference_state,
                frame_idx=self.config.target_frame_idx,
                obj_id=self.config.target_object_id,
                points=None,
                labels=None,
                clear_old_points=True,
                normalize_coords=True,
                box=target_box,
            )
        )

        return object_ids, mask_logits

    def process_video(self, source_frames: List[str]):
        """動画の推論処理を実行"""
        annotator = Annotator()

        self.logger.info("Starting video annotation processing...")

        for (
            frame_idx,
            object_ids,
            mask_logits,
        ) in self.annotator_processor.sam2_model.propagate_in_video(
            self.annotator_processor.inference_state
        ):
            # マスクを処理
            masks = (mask_logits > 0.0).cpu().numpy()
            masks = np.squeeze(masks).astype(bool)

            # フレームを読み込み
            frame_path = source_frames[frame_idx]
            frame = cv2.imread(frame_path)

            # マスクデータから結果を生成
            mask_util = MaskUtil(masks, frame)

            # 透過画像を保存
            self._save_transparent_image(mask_util, frame_idx)

            # テキストデータを保存
            # self._save_text_data(mask_util, frame_idx)
            # セグメンテーション用のアノテーションデータを保温
            self._save_segmentation_data(mask_util, frame_idx)

            # プレビューフレームを処理
            self.annotator_processor.process_preview_frame(
                frame_idx, masks, object_ids, annotator, source_frames
            )

            # 進捗をログ出力
            if frame_idx % 10 == 0:
                self.logger.info(f"Processed frame {frame_idx}")

    def _save_transparent_image(self, mask_util: MaskUtil, frame_idx: int):
        """透過画像を保存"""
        png_output_path = f"{self.video_config.png_path}/{self.video_config.base_name}_{frame_idx:05}.png"
        cv2.imwrite(png_output_path, mask_util.transparent_image)

    # def _save_text_data(self, mask_util: MaskUtil, frame_idx: int):
    #     """テキストデータを保存"""
    #     txt_output_path = f"{self.video_config.txt_path}/{self.video_config.base_name}_{frame_idx:05}.txt"
    #     with open(txt_output_path, mode="w") as f:
    #         f.write(mask_util.text)

    def _save_segmentation_data(self, mask_util: MaskUtil, frame_idx: int):
        """セグメンテーションデータを保存"""
        seg_output_path = f"{self.video_config.segment_path}/{self.video_config.base_name}_{frame_idx:05}.txt"
        with open(seg_output_path, mode="w") as f:
            f.write(mask_util.text)

    def run(self):
        """全体の処理を実行"""
        try:
            # 初期設定
            self.setup()

            # 動画仕様を取得
            frame_width, frame_height, video_fps = self.get_video_specs()

            # フレームを生成
            source_frames = self.generate_frames(frame_width, frame_height)

            # SAM2を設定してターゲットを取得
            self.setup_sam2_and_get_target(source_frames)

            # 動画処理を実行
            self.process_video(source_frames)

            self.logger.info("Video annotation processing completed successfully")

        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            raise
        finally:
            cv2.destroyAllWindows()
