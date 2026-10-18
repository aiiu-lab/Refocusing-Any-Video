#!/usr/bin/env python3
"""Run Video Depth Anything on a video and save an 8-bit depth-map video."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from video_depth_anything.video_depth import VideoDepthAnything


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a per-frame depth/disparity map with Video Depth Anything."
    )
    parser.add_argument("-i", "--input", required=True, help="Input video path.")
    parser.add_argument("-o", "--output", required=True, help="Output depth-map video path.")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("VDA_CHECKPOINT"),
        help="Video Depth Anything checkpoint. Defaults to checkpoints/video_depth_anything_<encoder>.pth.",
    )
    parser.add_argument(
        "--encoder",
        choices=sorted(MODEL_CONFIGS),
        default=os.environ.get("VDA_ENCODER", "vitl"),
        help="VDA encoder variant.",
    )
    parser.add_argument("--input-size", type=int, default=518, help="VDA inference input size.")
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "auto"),
        help="Runtime device: auto, cuda, cuda:0, cpu, etc.",
    )
    parser.add_argument("--fp32", action="store_true", help="Disable autocast and run VDA in fp32.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output FPS. Defaults to the input video's FPS.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for quick tests or memory-limited runs.",
    )
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=None,
        help="Optionally downscale frames before depth estimation.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert the normalized map before saving.",
    )
    return parser.parse_args(argv)


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    return REPO_ROOT / "checkpoints" / f"video_depth_anything_{args.encoder}.pth"


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        if device.index is not None:
            torch.cuda.set_device(device.index)
        return "cuda"

    return device.type


def resize_long_edge(frame: np.ndarray, max_long_edge: int) -> np.ndarray:
    height, width = frame.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_long_edge:
        return frame

    scale = max_long_edge / long_edge
    out_width = max(1, int(round(width * scale)))
    out_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)


def read_video(path: Path, max_frames: int | None, max_long_edge: int | None) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if max_long_edge:
            frame_rgb = resize_long_edge(frame_rgb, max_long_edge)
        frames.append(frame_rgb)
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()
    if not frames:
        raise ValueError(f"No frames were read from: {path}")

    return np.stack(frames, axis=0), float(fps)


def load_model(checkpoint: Path, encoder: str, device: str) -> VideoDepthAnything:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"VDA checkpoint not found: {checkpoint}. Pass --checkpoint or set VDA_CHECKPOINT."
        )

    print(f"Loading Video Depth Anything ({encoder}) from {checkpoint} ...")
    model = VideoDepthAnything(**MODEL_CONFIGS[encoder])
    state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    print("Model ready.")
    return model


def normalize_depth(depths: np.ndarray, invert: bool = False) -> np.ndarray:
    depths = depths.astype(np.float32)
    finite = np.isfinite(depths)
    if not finite.any():
        return np.zeros(depths.shape, dtype=np.uint8)

    d_min = depths[finite].min()
    d_max = depths[finite].max()
    if d_max - d_min <= 1e-6:
        depth_norm = np.zeros(depths.shape, dtype=np.float32)
    else:
        depth_norm = (depths - d_min) / (d_max - d_min)
        depth_norm = np.nan_to_num(depth_norm, nan=0.0, posinf=1.0, neginf=0.0)
        depth_norm = np.clip(depth_norm, 0.0, 1.0)

    if invert:
        depth_norm = 1.0 - depth_norm

    return (depth_norm * 255.0 + 0.5).astype(np.uint8)


def write_depth_video(depth_uint8: np.ndarray, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames, height, width = depth_uint8.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_path}")

    for idx in range(total_frames):
        frame_bgr = cv2.cvtColor(depth_uint8[idx], cv2.COLOR_GRAY2BGR)
        writer.write(frame_bgr)
    writer.release()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint = resolve_checkpoint(args)
    device = resolve_device(args.device)
    fp32 = args.fp32 or device != "cuda"

    frames, input_fps = read_video(Path(args.input), args.max_frames, args.max_long_edge)
    output_fps = args.fps if args.fps is not None else input_fps
    print(f"Input: {args.input}")
    print(f"Frames/resolution: {frames.shape[0]} frames at {frames.shape[2]}x{frames.shape[1]}")
    print(f"Depth output: {args.output}")
    print(f"Device/precision: {device} / {'fp32' if fp32 else 'autocast'}")

    model = load_model(checkpoint, args.encoder, device)
    with torch.inference_mode():
        depths, _ = model.infer_video_depth(
            frames,
            output_fps,
            input_size=args.input_size,
            device=device,
            fp32=fp32,
        )

    depth_uint8 = normalize_depth(depths, invert=args.invert)
    write_depth_video(depth_uint8, Path(args.output), output_fps)
    print(
        f"Done. Saved {depth_uint8.shape[0]} frames to {args.output} "
        f"with range [{depth_uint8.min()}, {depth_uint8.max()}]."
    )


if __name__ == "__main__":
    main()


"""
python infer_video_depth.py --input ./input.mp4 --output ./depth.mp4
"""