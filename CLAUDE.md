# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

This is a SAM2-based machine learning pipeline for creating YOLO datasets from video files. The project runs in a Docker environment with NVIDIA GPU support.

### Docker Commands

Build the Docker image:
```bash
cd home && ./docker-build.sh
```

Run the Docker container:
```bash
cd home && ./docker-run.sh
```

### Core Pipeline Commands

The pipeline follows a sequential processing workflow. All commands should be run inside the Docker container:

1. **Video to Annotation**: Extract frames and generate SAM2 annotations
```bash
python3 video_2_annotation.py VIDEO_FILE.mp4 [--scale 0.8] [--fps 10]
```

2. **Annotation to Merged Data**: Combine annotations with background images
```bash
python3 annotation_2_marge.py
```

3. **Generate YOLO Dataset**: Convert merged data to YOLO format
```bash
python3 merge_2_dataset.py
```

4. **Train Models**:
```bash
python3 train_seg.py    # Segmentation model
python3 train_od.py     # Object detection model
```

5. **Run Inference**:
```bash
python3 inference_seg.py    # Segmentation inference
python3 inference_od.py     # Object detection inference
```

## Architecture Overview

### Project Structure
```
home/src/
├── video_2_annotation.py      # Entry point: video → SAM2 annotations
├── annotation_2_marge.py      # Data augmentation: annotations + backgrounds
├── merge_2_dataset.py         # YOLO dataset generation
├── train_seg.py/train_od.py   # Model training scripts
├── inference_seg.py/inference_od.py  # Inference scripts
└── common/                    # Shared modules
    ├── config.py              # Configuration classes
    ├── video_processor.py     # Main video processing logic
    ├── annotator.py           # SAM2 annotation wrapper
    ├── video_annotator.py     # Video-specific annotation logic
    ├── mask_util.py           # Mask processing utilities
    ├── util.py                # General utilities (contains os.system call)
    ├── bounding_box.py        # Bounding box operations
    ├── cli_parser.py          # Command line argument parsing
    └── validation.py          # Input validation
```

### Data Flow
1. **Input**: MP4 video files
2. **Frame Extraction**: FFmpeg extracts frames at specified FPS/scale
3. **SAM2 Processing**: Generates masks and segmentation coordinates
4. **Data Augmentation**: Combines objects with background images
5. **YOLO Format**: Converts to standard YOLO dataset structure
6. **Output**: Trained YOLOv8 models for segmentation/detection

### Configuration System
The project uses dataclass-based configuration in `common/config.py`:
- `Config`: Main SAM2 and processing settings
- `VideoConfig`: Video-specific parameters (scale, fps, paths)

Key configuration points:
- SAM2 checkpoint: `/segment-anything-2/checkpoints/sam2_hiera_tiny.pt`
- Output directory: `/src/dataset/annotation/`
- Headless mode enabled for Jetson compatibility
- OpenCV display disabled by default

### Critical Dependencies
- SAM2 (Meta's Segment Anything Model 2)
- YOLOv8 (Ultralytics)
- PyTorch with CUDA support
- OpenCV for image processing
- FFmpeg for video frame extraction

### Security Note
`common/util.py:45-46` contains an `os.system()` call for FFmpeg execution. This creates a potential command injection vulnerability if video paths contain malicious characters. The function should be refactored to use `subprocess.run()` with proper argument sanitization.

### File Naming Conventions
- SAM2 frames: `00000.jpeg` format (no basename prefix)
- Other files: `CLASSNAME_00000.png` format
- Output structure maintains class-based directory organization

### GPU Requirements
- NVIDIA GPU with CUDA support strongly recommended
- Docker runtime requires `--runtime nvidia`
- Memory limits: 8GB RAM, 12GB swap, 4GB shared memory