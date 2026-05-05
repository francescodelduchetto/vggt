# Docker Usage — VGGT Depth Map Generator

Generate per-frame depth maps from a video (or a folder of images) using the
[VGGT model](https://github.com/facebookresearch/vggt), fully containerised and
ready to run on any NVIDIA-equipped machine.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Docker Engine | ≥ 24 |
| NVIDIA Container Toolkit | latest (`nvidia-ctk`) |
| NVIDIA driver | CUDA 12.1-compatible (≥ 530.x) |

Install the NVIDIA Container Toolkit if you haven't already:
```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Quick start

### 1. Build the image

```bash
docker build -t vggt-depth .
```

> The first build downloads PyTorch and other dependencies (~5 GB).  
> Subsequent builds are fast thanks to Docker layer caching.

### 2. Run on a video

```bash
docker run --rm --gpus all \
  -v /path/to/myvideo.mp4:/input/video.mp4:ro \
  -v /path/to/output:/output \
  vggt-depth \
    --input /input/video.mp4 \
    --output /output
```

### 3. Run on a folder of images

```bash
docker run --rm --gpus all \
  -v /path/to/images:/input:ro \
  -v /path/to/output:/output \
  vggt-depth \
    --input /input \
    --output /output
```

---

## Using Docker Compose

Edit the `docker-compose.yml` to point `./input` and `./output` at your host
directories, then:

```bash
# Build (once)
docker compose build

# Run with a video file
docker compose run --rm vggt-depth --input /input/myvideo.mp4 --output /output

# Run with an image directory
docker compose run --rm vggt-depth --input /input --output /output
```

---

## CLI reference

```
usage: generate_depth.py [-h] --input INPUT [--output OUTPUT] [--fps FPS]
                         [--max_frames MAX_FRAMES] [--colormap COLORMAP]
                         [--output_video_fps OUTPUT_VIDEO_FPS] [--no_video]
                         [--model_url MODEL_URL]

Generate depth maps from a video using VGGT

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to an input video file or a directory containing images.
  --output OUTPUT       Directory where depth maps and the output video will be saved
                        (default: /output).
  --fps FPS             How many frames per second to sample from the video
                        (default: 1.0). Has no effect when --input is a directory.
  --max_frames MAX_FRAMES
                        Maximum number of frames to process (default: 200).
  --colormap COLORMAP   Matplotlib colormap used to render the depth maps
                        (default: inferno).
  --output_video_fps OUTPUT_VIDEO_FPS
                        FPS of the output depth-map video. Defaults to --fps for
                        video input or 10 for image directories.
  --no_video            Skip creating the output depth-map video; save PNG only.
  --model_url MODEL_URL
                        URL to download model weights from (default: VGGT-1B).
```

---

## Output structure

```
/output/
├── depth_maps/
│   ├── depth_000000.png   ← colorised depth map for frame 0
│   ├── depth_000001.png
│   └── …
├── depth_raw.npz          ← raw float32 depth values, shape (S, H, W, 1)
└── depth_video.mp4        ← depth maps assembled into a video
```

The raw `.npz` file can be loaded in Python for further processing:

```python
import numpy as np
data = np.load("/output/depth_raw.npz")
depth = data["depth"]  # shape: (S, H, W, 1)
```

---

## Notes

* **Model weights** are downloaded automatically on first run from Hugging Face
  (~4 GB).  To avoid re-downloading, mount a persistent cache volume:
  ```bash
  docker run --rm --gpus all \
    -v vggt_model_cache:/app/.hf_cache \
    -v /path/to/myvideo.mp4:/input/video.mp4:ro \
    -v /path/to/output:/output \
    vggt-depth --input /input/video.mp4 --output /output
  ```
* **Memory**: processing 100 frames requires ~21 GB GPU memory (H100 benchmark).
  Use `--max_frames` and/or `--fps` to reduce frame count for GPUs with less VRAM.
* VGGT processes **all frames jointly** (it's a multi-view transformer), so
  splitting a long video into shorter segments with `--max_frames` may give
  better results than forcing all frames at once.
