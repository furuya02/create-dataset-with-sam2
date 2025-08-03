# --------------------------------------------------------------------
#  Pytorch running on Jetson L4T (36.4)
# --------------------------------------------------------------------  
FROM dustynv/l4t-pytorch:r36.4.0
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_EXTRA_INDEX_URL="" \
    PIP_INDEX_URL="https://pypi.org/simple/" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
# System Package Installation
RUN rm -f /etc/pip.conf /root/.pip/pip.conf ~/.pip/pip.conf 2>/dev/null || true
# Install system dependencies for GUI and OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # OpenGL/Mesa libraries
        libgl1-mesa-glx \
        libgl1-mesa-dri \
        # GUI libraries
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        libfontconfig1 \
        # Python GUI support
        python3-tk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Python Package Installation
RUN pip install \
        --trusted-host pypi.org \
        --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org \
        --index-url https://pypi.org/simple/ \
        # Core dependencies (specific versions for compatibility)
        numpy==1.24.3 \
        pillow \
        # Computer vision
        opencv-python \
        # Machine learning
        ultralytics \
        # Scientific computing
        matplotlib \
        pandas \
        # Utilities
        pyyaml \
        gitpython

# --------------------------------------------------------------------
# SAM2 Installation
# --------------------------------------------------------------------  
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y ffmpeg wget git

RUN pip install numpy==1.26.4 opencv-python tqdm matplotlib hydra-core omegaconf

# SAM2に必要な追加依存関係をインストール
RUN pip install iopath pillow

RUN git clone https://github.com/facebookresearch/segment-anything-2.git

ENV SAM2_HOME=/segment-anything-2
WORKDIR ${SAM2_HOME}/

# setup.pyでtorchとtorchvisionの以下のバージョンを要求するため、無効化する
RUN sed -i -e '/^[[:space:]]*"torch>=2.5.1",/d' -e '/^[[:space:]]*"torchvision>=0.20.1",/d' setup.py
# SAM2のsetup.pyから依存関係を確認してインストール
RUN pip install tqdm hydra-core iopath pillow

# SAM2を依存関係無視でインストール
RUN pip install -e . --no-deps --force-reinstall -q
RUN python3 setup.py build_ext --inplace

RUN pip install -q supervision[assets] jupyter_bbox_widget

# checkpoints
RUN mkdir -p ${SAM2_HOME}/checkpoints
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt -P ${SAM2_HOME}/checkpoints
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt -P ${SAM2_HOME}/checkpoints
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt -P ${SAM2_HOME}/checkpoints
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -P ${SAM2_HOME}/checkpoints

ENV HOME=/src
WORKDIR ${HOME}/
