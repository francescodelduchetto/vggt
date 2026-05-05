# ── Base image ───────────────────────────────────────────────────────────────
# CUDA 12.1 + cuDNN 8, Ubuntu 22.04.  Change the tag to match your host driver
# (any driver that supports CUDA 12.1 works; see https://docs.nvidia.com/deploy/cuda-compatibility/).
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ── System dependencies ───────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        python3.11-venv \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python / pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && python -m pip install --upgrade pip

# ── Python dependencies ───────────────────────────────────────────────────────
# Install PyTorch with CUDA 12.1 wheels first (torch.org index)
RUN pip install --no-cache-dir \
        torch==2.3.1 \
        torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cu121

# Install remaining requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install matplotlib and opencv (needed by generate_depth.py but not in requirements.txt)
RUN pip install --no-cache-dir matplotlib opencv-python-headless

# ── Copy source code ──────────────────────────────────────────────────────────
WORKDIR /app
COPY . /app

# Install the vggt package in editable mode so imports work
RUN pip install --no-cache-dir -e .

# ── Runtime configuration ─────────────────────────────────────────────────────
# Default output directory (can be overridden by bind-mounting /output)
RUN mkdir -p /output

# Hugging Face cache inside the image so the model can be pre-baked if desired
ENV HF_HOME=/app/.hf_cache

# ── Entrypoint ────────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "/app/generate_depth.py"]
# Default: show help
CMD ["--help"]
