from common.config import Config, VideoConfig
from common.video_processor import VideoProcessor
from common.cli_parser import parse_arguments
from common.validation import validate_video_file


def main():
    """メイン関数"""
    # コマンドライン引数を解析
    args = parse_arguments()
    
    # 動画ファイルの有効性を検証
    validate_video_file(args.video)
    
    # 設定クラスを初期化
    config = Config()
    video_config = VideoConfig(
        video_path=args.video, 
        scale=args.scale, 
        fps=args.fps
    )
    video_config.setup_paths(config.output_path)
    
    # プロセッサーを初期化して実行
    processor = VideoProcessor(config, video_config)
    processor.run()


if __name__ == "__main__":
    main()