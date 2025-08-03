# SAM2を使用したYOLOデータセットの生成

SAM2（Segment Anything Model 2）を使用して動画から機械学習用データセットを作成するための包括的なツールセットです。動画ファイルからフレームを抽出し、オブジェクトセグメンテーションのアノテーションを自動生成し、YOLOv8などで使用可能なデータセットを作成できます。

## 概要

このプロジェクトは、Meta社のSAM2モデルを活用した完全な機械学習パイプラインを提供します：

1. **動画アノテーション**: 動画内のオブジェクトを自動セグメンテーション
2. **データ合成**: アノテーションデータと背景を組み合わせてデータセットを拡張
3. **YOLO形式変換**: セグメンテーション・物体検出用データセットを自動生成
4. **モデル学習**: YOLOv8でのファインチューニング
5. **推論テスト**: 学習済みモデルでのリアルタイム推論

## 主な機能

- 🎥 動画ファイル（MP4）からのフレーム抽出
- 🤖 SAM2を使用した自動オブジェクトセグメンテーション
- 🖼️ マスク画像とセグメンテーション座標の生成
- 📊 プレビュー機能付きインタラクティブアノテーション
- 🔄 背景画像との合成によるデータ拡張
- � YOLO形式データセットの自動生成（セグメンテーション・物体検出）
- 🏋️ YOLOv8での自動ファインチューニング
- 🔍 リアルタイム推論とテスト機能
- �🐳 Docker環境での簡単実行

## データ処理フロー

```
1. 動画ファイル → [video_2_annotation.py] → アノテーションデータ
2. アノテーション + 背景画像 → [annotation_2_marge.py] → 合成データ
3. 合成データ → [merge_2_dataset.py] → YOLO形式データセット
4. データセット → [train_seg.py / train_od.py] → 学習済みモデル
5. 学習済みモデル → [check_inference.py] → 推論テスト
```

## プロジェクト構成

```
├── annotation/          # 生成されたアノテーションデータ
│   ├── SAMPLE/
│   │   ├── frame/       # 抽出されたフレーム画像（SAM2用）
│   │   │   ├── 00000.jpeg
│   │   │   ├── 00001.jpeg
│   │   │   └── ...
│   │   ├── png/         # マスク画像（透過PNG）
│   │   │   ├── SAMPLE_00000.png
│   │   │   ├── SAMPLE_00001.png
│   │   │   └── ...
│   │   ├── preview/     # プレビュー画像（確認用）
│   │   │   ├── SAMPLE_00000.png
│   │   │   ├── SAMPLE_00001.png
│   │   │   └── ...
│   │   └── seg/         # セグメンテーション座標データ
│   │       ├── SAMPLE_00000.txt
│   │       ├── SAMPLE_00001.txt
│   │       └── ...
│   └── TOMATO/
│       ├── frame/
│       ├── png/
│       ├── preview/
│       └── seg/
├── dataset/
│   ├── background/      # 背景画像（合成用）
│   ├── merge/          # 合成データ
│   └── yolo_seg/       # YOLO形式データセット
│       ├── train/
│       ├── val/
│       └── data.yaml
├── home/
│   ├── docker-build.sh  # Dockerイメージビルドスクリプト
│   ├── docker-run.sh    # Docker実行スクリプト
│   ├── Dockerfile       # Docker設定ファイル
│   └── src/
│       ├── video_2_annotation.py      # 動画→アノテーション
│       ├── annotation_2_marge.py      # アノテーション→合成データ
│       ├── merge_2_dataset.py         # 合成データ→YOLO形式
│       ├── train_seg.py               # セグメンテーション学習
│       ├── train_od.py                # 物体検出学習
│       ├── check_inference.py         # 推論テスト
│       └── common/                    # 共通モジュール
└── sample_mp4/          # サンプル動画ファイル
```

## 必要な環境

- Docker
- NVIDIA GPU（推奨）
- CUDA対応環境

## セットアップ

### 1. リポジトリのクローン

```bash
git clone <このリポジトリのURL>
cd 2025.07.19_CreateDataset_with_SAM2
```

### 2. Dockerイメージのビルド

```bash
cd home
chmod +x docker-build.sh
./docker-build.sh
```

### 3. Docker環境の実行

```bash
chmod +x docker-run.sh
./docker-run.sh
```

## 使用方法

### ステップ1: 動画からアノテーションデータを作成

```bash
# Docker コンテナ内で実行
python3 video_2_annotation.py SAMPLE.mp4

# オプション指定
python3 video_2_annotation.py SAMPLE.mp4 --scale 0.5 --fps 5
```

**パラメータ:**
- `video`: 処理する動画ファイルのパス（必須）
- `--scale`: フレーム解像度のスケール係数（デフォルト: 0.8）
- `--fps`: 抽出するフレームレート（デフォルト: 10）

### ステップ2: アノテーションデータから合成データを作成

```bash
python3 annotation_2_marge.py
```

背景画像（`dataset/background/`）とアノテーションデータを組み合わせて、データ拡張を行います。

### ステップ3: YOLO形式データセットを生成

```bash
python3 merge_2_dataset.py
```

合成データからYOLOv8で使用可能なデータセット（`dataset/yolo_seg/`）を自動生成します。

### ステップ4: モデルの学習

```bash
# セグメンテーション
python3 train_seg.py

# 物体検出
python3 train_od.py
```

### ステップ5: 推論テスト

```bash
python3 check_inference.py
```

学習済みモデルを使用してリアルタイム推論をテストします。

## 複数動画のマージ

同じクラスの複数の動画を処理する場合：

1. 各動画を個別に処理：
   ```bash
   python video_2_annotation.py /src/SAMPLE_1.mp4
   python video_2_annotation.py /src/SAMPLE_2.mp4
   ```

2. 生成されたフォルダをマージ：
   ```bash
   # annotation/SAMPLE_1/ と annotation/SAMPLE_2/ を annotation/SAMPLE/ にマージ
   # ファイル名は重複しないため、直接コピー可能
   ```

## 出力データ形式

### アノテーションデータ
```
annotation/クラス名/
├── frame/          # SAM2用フレーム画像（JPEG、ファイル名は00000.jpeg形式）
│   ├── 00000.jpeg
│   ├── 00001.jpeg
│   └── ...
├── png/            # 透過マスク画像（PNG）
│   ├── クラス名_00000.png
│   ├── クラス名_00001.png
│   └── ...
├── preview/        # アノテーション確認用プレビュー画像
│   ├── クラス名_00000.png
│   ├── クラス名_00001.png
│   └── ...
└── seg/            # セグメンテーション座標データ（テキスト）
    ├── クラス名_00000.txt
    ├── クラス名_00001.txt
    └── ...
```

### YOLO形式データセット
```
dataset/yolo_seg/
├── train/
│   ├── images/     # 学習用画像
│   └── labels/     # 学習用ラベル
├── val/
│   ├── images/     # 検証用画像
│   └── labels/     # 検証用ラベル
└── data.yaml       # データセット設定ファイル
```

**重要な注意事項:**
- `frame/`内のファイル名は`00000.jpeg`形式（SAM2の要求仕様）
- 他のファイルは`クラス名_00000.png`形式でベースネーム付き
- 複数動画のマージ時もファイル名競合は発生しません

## 設定

### 主要設定項目

`src/common/config.py`で以下の設定を変更できます：

- `checkpoint`: SAM2モデルのチェックポイントファイル
- `config`: SAM2設定ファイル（デフォルト: sam2_hiera_t.yaml）
- `output_path`: 出力ディレクトリ
- `target_frame_idx`: ターゲットフレームのインデックス
- `disable_opencv_display`: OpenCV表示の無効化（Jetson環境対応）

## 実行時の注意事項

### 処理時間について
- **video_2_annotation**: 動画の長さとフレーム数に応じて時間がかかります
- **annotation_2_marge**: 大量の合成データ生成のため、相当な時間を要します
- **train_seg/train_od**: GPU使用での学習時間が必要です

### 安定実行のために
- **SSH接続での実行は避ける**: 長時間処理でセッション切断のリスクがあります
- **ローカル端末での実行を推奨**: 処理が途中で止まらないようにしてください
- **十分なディスク容量を確保**: 大量の画像データが生成されます

### GPU環境
- NVIDIA GPU + CUDA環境を強く推奨
- メモリ不足時はバッチサイズや画像解像度を調整してください

## 依存関係

主要なライブラリ：
- SAM2 (Segment Anything Model 2)
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- FFmpeg
- PIL (Pillow)

## トラブルシューティング

### よくある問題

1. **GPU メモリ不足**
   - `--scale`パラメータを小さくしてフレーム解像度を下げる
   - `--fps`を下げてフレーム数を減らす

2. **Docker 実行エラー**
   - NVIDIA Dockerがインストールされているか確認
   - `docker-run.sh`の権限を確認

3. **動画ファイルが認識されない**
   - MP4形式であることを確認
   - ファイルパスが正しいことを確認

4. **フレーム命名エラー**
   - SAM2は`00000.jpeg`形式のファイル名を要求します
   - ベースネーム付きファイル名は使用できません

## ライセンス

このプロジェクトは研究・開発目的で作成されています。SAM2の利用規約に従ってご使用ください。

## 更新履歴

- **2025.07.19**: プロジェクト初回作成
- **2025.07.20**: 
  - 動画アノテーション機能の実装
  - Docker環境の構築
  - SAM2統合完了
- **2025.07.21**: 
  - データ合成機能追加（annotation_2_marge.py）
  - YOLO形式データセット生成機能追加（merge_2_dataset.py）
  - 自動クラス名検出機能実装
  - セグメンテーション・物体検出両対応
- **2025.07.26**: 
  - 完全パイプライン統合
  - 複数動画マージ機能
  - パフォーマンス最適化

## ベストプラクティス

### データ品質向上のために
1. **多様な背景画像を用意**: `dataset/background/`に様々な背景を配置
2. **複数角度での動画撮影**: 同一オブジェクトを異なる角度から撮影
3. **適切なフレームレート**: オブジェクトの動きに応じてFPSを調整

### 効率的な作業フロー
1. **小規模テスト**: 少数フレームでパイプラインをテスト
2. **段階的スケールアップ**: 問題ないことを確認してから大規模実行
3. **定期的なチェックポイント**: 各段階で結果を確認

## 貢献

バグ報告や機能改善の提案は、Issueまたはプルリクエストでお願いします。

## 参考リンク

- [SAM2 公式リポジトリ](https://github.com/facebookresearch/segment-anything-2)
- [Meta AI Research](https://ai.meta.com/)
