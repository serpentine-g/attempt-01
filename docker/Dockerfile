# Dockerfile for Image-to-3D Gaussian Splat Competition
# Build Time Target: < 3 hours
# Startup Time Target: < 1 hour

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libjpeg-dev \
    libwebp-dev \
    libwebp7 \
    zlib1g-dev \
    libtiff-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (no ToS issues, uses conda-forge by default)
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

ENV PATH="/opt/conda/bin:${PATH}"

# Create conda environment with Python 3.10
RUN conda create -n trellis2 python=3.10 -y

# Activate environment and install PyTorch
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate trellis2 && \
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Copy application code
COPY . /app/

# Install Python dependencies
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate trellis2 && \
    pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard psutil packaging && \
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 && \
    pip uninstall -y pillow pillow-simd 2>/dev/null || true && \
    (pip install pillow-simd --no-cache-dir || pip install pillow --no-cache-dir) && \
    pip install kornia timm

# Install flash-attention
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate trellis2 && \
    pip install flash-attn==2.7.3 --no-build-isolation

# Install nvdiffrast
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate trellis2 && \
    mkdir -p /tmp/extensions && \
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    pip install /tmp/extensions/nvdiffrast --no-build-isolation && \
    rm -rf /tmp/extensions/nvdiffrast

# Install nvdiffrec
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate trellis2 && \
    mkdir -p /tmp/extensions && \
    git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec && \
    pip install /tmp/extensions/nvdiffrec --no-build-isolation && \
    rm -rf /tmp/extensions/nvdiffrec

# Install CuMesh
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate trellis2 && \
    mkdir -p /tmp/extensions && \
    git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive && \
    pip install /tmp/extensions/CuMesh --no-build-isolation && \
    rm -rf /tmp/extensions/CuMesh

# Install FlexGEMM
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate trellis2 && \
    mkdir -p /tmp/extensions && \
    git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive && \
    pip install /tmp/extensions/FlexGEMM --no-build-isolation && \
    rm -rf /tmp/extensions/FlexGEMM

# Install o-voxel (download Eigen separately since .git not available in Docker)
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate trellis2 && \
    mkdir -p /tmp/extensions && \
    cp -r o-voxel /tmp/extensions/o-voxel && \
    cd /tmp/extensions/o-voxel/third_party && \
    rm -rf eigen && \
    git clone --depth 1 https://gitlab.com/libeigen/eigen.git eigen && \
    cd /app && \
    pip install /tmp/extensions/o-voxel --no-build-isolation && \
    rm -rf /tmp/extensions/o-voxel

# Install FastAPI and Uvicorn for the server
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate trellis2 && \
    pip install fastapi uvicorn[standard] python-multipart

# Set environment variables
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Expose port 10006
EXPOSE 10006

# Copy and set entrypoint script
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

