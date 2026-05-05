#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate depth maps from a video using the VGGT model.

Usage:
    python generate_depth.py --input /path/to/video.mp4 --output /path/to/output/
    python generate_depth.py --input /path/to/images/ --output /path/to/output/
"""

import argparse
import os
import sys
import glob
import shutil

import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate depth maps from a video using VGGT")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an input video file or a directory containing images.",
    )
    parser.add_argument(
        "--output",
        default="/output",
        help="Directory where depth maps and the output video will be saved (default: /output).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="How many frames per second to sample from the video (default: 1.0). "
             "Has no effect when --input is a directory of images.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=200,
        help="Maximum number of frames to process (default: 200). "
             "Limits memory usage for long videos.",
    )
    parser.add_argument(
        "--colormap",
        default="inferno",
        help="Matplotlib colormap used to render the depth maps (default: inferno).",
    )
    parser.add_argument(
        "--output_video_fps",
        type=float,
        default=None,
        help="FPS of the output depth-map video. "
             "Defaults to the same value as --fps when reading a video, or 10 when reading images.",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Skip creating the output depth-map video and only save PNG images.",
    )
    parser.add_argument(
        "--model_url",
        default="https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
        help="URL to download the VGGT model weights from "
             "(default: VGGT-1B on Hugging Face).",
    )
    return parser.parse_args()


def extract_frames_from_video(video_path: str, target_fps: float, max_frames: int, out_dir: str):
    """Extract frames from a video at the given fps and write them to out_dir."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 25.0

    frame_interval = max(1, int(round(src_fps / target_fps)))
    print(f"Source FPS: {src_fps:.2f}  →  sampling every {frame_interval} frame(s) "
          f"(effective {src_fps / frame_interval:.2f} fps)")

    frame_idx = 0
    saved = 0
    image_paths = []

    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            path = os.path.join(out_dir, f"{saved:06d}.png")
            cv2.imwrite(path, frame)
            image_paths.append(path)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Extracted {saved} frames from video.")
    return image_paths


def collect_image_paths(directory: str):
    """Return sorted image paths from a directory."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, ext)))
        paths.extend(glob.glob(os.path.join(directory, ext.upper())))
    paths = sorted(set(paths))
    if not paths:
        raise RuntimeError(f"No images found in directory: {directory}")
    return paths


def load_model(model_url: str, device: str):
    """Load and return the VGGT model."""
    from vggt.models.vggt import VGGT

    print("Loading VGGT model...")
    model = VGGT()
    print(f"Downloading / loading weights from: {model_url}")
    state_dict = torch.hub.load_state_dict_from_url(model_url, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    print("Model loaded.")
    return model


def run_inference(model, image_paths: list, device: str) -> dict:
    """Run VGGT inference and return the predictions dict with numpy arrays."""
    from vggt.utils.load_fn import load_and_preprocess_images

    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
        if device == "cuda"
        else torch.float32
    )

    images = load_and_preprocess_images(image_paths).to(device)
    print(f"Running inference on {len(image_paths)} frames (dtype={dtype}) …")

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype, enabled=(device == "cuda")):
            predictions = model(images)

    # Move tensors to CPU numpy
    result = {}
    for key, val in predictions.items():
        if isinstance(val, torch.Tensor):
            result[key] = val.cpu().float().numpy().squeeze(0)  # remove batch dim

    print("Inference complete.")
    return result


def depth_to_colormap(depth: np.ndarray, colormap: str = "inferno") -> np.ndarray:
    """Convert a single (H, W) or (H, W, 1) depth array to a uint8 RGB image."""
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    # Normalise to [0, 1]
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-8:
        norm = np.zeros_like(depth)
    else:
        norm = (depth - d_min) / (d_max - d_min)
    cmap = plt.get_cmap(colormap)
    rgb = cmap(norm)[:, :, :3]  # (H, W, 3) float in [0,1]
    return (rgb * 255).astype(np.uint8)


def save_depth_maps(depth_maps: np.ndarray, out_dir: str, colormap: str) -> list:
    """
    Save per-frame depth maps as PNG images.

    Args:
        depth_maps: numpy array of shape (S, H, W, 1) or (S, H, W)
        out_dir: directory to save the images
        colormap: matplotlib colormap name

    Returns:
        List of saved file paths.
    """
    paths = []
    for i in range(depth_maps.shape[0]):
        frame_depth = depth_maps[i]
        rgb = depth_to_colormap(frame_depth, colormap)
        path = os.path.join(out_dir, f"depth_{i:06d}.png")
        # cv2 expects BGR
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        paths.append(path)
    print(f"Saved {len(paths)} depth map PNG(s) to {out_dir}")
    return paths


def save_raw_depth(depth_maps: np.ndarray, out_dir: str):
    """Save raw depth values as a .npz file for downstream use."""
    path = os.path.join(out_dir, "depth_raw.npz")
    np.savez_compressed(path, depth=depth_maps)
    print(f"Raw depth data saved to {path}")


def create_depth_video(frame_paths: list, out_path: str, fps: float):
    """Combine depth-map PNG images into an mp4 video."""
    if not frame_paths:
        return
    sample = cv2.imread(frame_paths[0])
    h, w = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for p in frame_paths:
        frame = cv2.imread(p)
        writer.write(frame)
    writer.release()
    print(f"Depth video saved to {out_path}")


def main():
    args = parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("WARNING: CUDA not available – running on CPU (this will be slow).")

    # ── Output directory ─────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    # ── Collect input frames ──────────────────────────────────────────────────
    frames_dir = os.path.join(args.output, "_frames")
    os.makedirs(frames_dir, exist_ok=True)

    input_is_video = False
    if os.path.isfile(args.input):
        # Treat as video
        input_is_video = True
        image_paths = extract_frames_from_video(
            args.input, args.fps, args.max_frames, frames_dir
        )
        output_video_fps = args.output_video_fps if args.output_video_fps else args.fps
    elif os.path.isdir(args.input):
        image_paths = collect_image_paths(args.input)
        if len(image_paths) > args.max_frames:
            print(f"Capping input images at {args.max_frames} (found {len(image_paths)}).")
            image_paths = image_paths[: args.max_frames]
        output_video_fps = args.output_video_fps if args.output_video_fps else 10.0
    else:
        sys.exit(f"ERROR: --input '{args.input}' is neither a file nor a directory.")

    if not image_paths:
        sys.exit("ERROR: No frames to process.")

    print(f"Total frames to process: {len(image_paths)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(args.model_url, device)

    # ── Inference ─────────────────────────────────────────────────────────────
    predictions = run_inference(model, image_paths, device)

    # depth shape: (S, H, W, 1)
    depth_maps = predictions.get("depth")
    if depth_maps is None:
        sys.exit("ERROR: Model did not return depth predictions.")

    # ── Save outputs ──────────────────────────────────────────────────────────
    depth_png_dir = os.path.join(args.output, "depth_maps")
    os.makedirs(depth_png_dir, exist_ok=True)

    frame_paths = save_depth_maps(depth_maps, depth_png_dir, args.colormap)
    save_raw_depth(depth_maps, args.output)

    if not args.no_video:
        video_out = os.path.join(args.output, "depth_video.mp4")
        create_depth_video(frame_paths, video_out, output_video_fps)

    # Clean up temporary frame directory if it was created for a video input
    if input_is_video and os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)

    print("\nDone. Output files:")
    for f in sorted(os.listdir(args.output)):
        print(f"  {os.path.join(args.output, f)}")


if __name__ == "__main__":
    main()
