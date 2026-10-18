"""Create bokeh/refocus input videos from RGB and depth videos.

This script replaces the previous static and time-varying demo scripts:

  * tools/preprocessing/create_bokeh_video.py
  * create_infer_video_timevary_depth.py

It supports both a constant focal plane (`--focal-plane`, historically
`--zero-depth`) and per-frame focal planes (`--focal-plane-file`,
`--focal-plane-values`, or `--time-varying`).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rav.renderers.bokeh import CombinedBokehRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render bokeh video from RGB and depth videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="RGB input video path.")
    parser.add_argument("--depth-input", required=True, help="Depth video path.")
    parser.add_argument("--output", default="./results/bokeh/created_blur_video.mp4")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quality", type=int, default=9)

    parser.add_argument("--arnet-path", default="checkpoints/bokehme/arnet.pth")
    parser.add_argument("--iunet-path", default="checkpoints/bokehme/iunet.pth")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu'.")

    parser.add_argument("--K", type=float, default=21.0)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--defocus-scale", type=float, default=10.0)
    parser.add_argument("--gamma-min", type=float, default=1.0)
    parser.add_argument("--gamma-max", type=float, default=5.0)

    parser.add_argument(
        "--zero-depth",
        "--focal-plane",
        dest="zero_depth",
        type=float,
        default=None,
        help="Constant focal plane. If omitted, it is sampled from the first depth frame.",
    )
    parser.add_argument(
        "--zero-depth-file",
        "--focal-plane-file",
        dest="zero_depth_file",
        default=None,
        help="Text file containing one focal-plane value per line, or comma-separated values.",
    )
    parser.add_argument(
        "--zero-depth-values",
        "--focal-plane-values",
        dest="zero_depth_values",
        default=None,
        help="Comma-separated per-frame focal-plane values, e.g. '0.8,0.7,0.6'.",
    )
    parser.add_argument(
        "--time-varying",
        action="store_true",
        help="Generate per-frame focal planes. Uses linear mode if start/end are set, otherwise random walk.",
    )
    parser.add_argument("--zero-depth-start", "--focal-plane-start", dest="zero_depth_start", type=float, default=None)
    parser.add_argument("--zero-depth-end", "--focal-plane-end", dest="zero_depth_end", type=float, default=None)
    parser.add_argument("--zero-depth-delta", "--focal-plane-delta", dest="zero_depth_delta", type=float, default=0.05)
    parser.add_argument("--zero-depth-range", "--focal-plane-range", dest="zero_depth_range", type=float, nargs=2, default=(0.05, 0.95))
    parser.add_argument(
        "--fit-zero-depth-seq",
        "--fit-focal-plane-seq",
        dest="fit_zero_depth_seq",
        choices=("interpolate", "error"),
        default="interpolate",
        help="What to do when a provided focal-plane sequence length differs from the video length.",
    )

    parser.add_argument("--height", type=int, default=None, help="Optional output height.")
    parser.add_argument("--width", type=int, default=None, help="Optional output width.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save-frames", action="store_true", help="Save rendered PNG frames next to the output video.")
    parser.add_argument("--save-components", action="store_true", help="Also save classical/neural/defocus component videos.")
    parser.add_argument("--preview", action="store_true", help="Show a matplotlib preview of frame 0.")
    parser.add_argument("--preview-path", default=None, help="Optional path to save the preview image.")

    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_video_bgr(video_path: str | Path) -> np.ndarray:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from video: {video_path}")

    video = np.stack(frames, axis=0)
    print(f"Loaded {len(video)} frames from {video_path}")
    return video


def resize_video(video: np.ndarray, width: int | None, height: int | None, interpolation: int) -> np.ndarray:
    if width is None and height is None:
        return video

    src_h, src_w = video.shape[1:3]
    width = width or src_w
    height = height or src_h

    resized = np.empty((video.shape[0], height, width, video.shape[3]), dtype=video.dtype)
    for i, frame in enumerate(video):
        resized[i] = cv2.resize(frame, (width, height), interpolation=interpolation)
    return resized


def prepare_rgb_video(video_bgr: np.ndarray, width: int | None, height: int | None) -> np.ndarray:
    video_bgr = resize_video(video_bgr, width, height, cv2.INTER_LINEAR)
    video_rgb = video_bgr[..., ::-1].copy()
    return video_rgb.astype(np.float32) / 255.0


def prepare_depth_video(video_bgr: np.ndarray, width: int | None, height: int | None) -> np.ndarray:
    video_bgr = resize_video(video_bgr, width, height, cv2.INTER_NEAREST)
    depth = video_bgr[..., 0:1]
    return depth.astype(np.float32) / 255.0


def sample_zero_depth_from_depth(depth_frame: np.ndarray, zero_depth_range: Sequence[float]) -> float:
    hist, _ = np.histogram(depth_frame.flatten(), bins=10)
    bin_start = random.choices([0.1 * i for i in range(10)], weights=hist, k=1)[0]
    sampled = random.uniform(bin_start, bin_start + 0.1)
    return float(np.clip(sampled, zero_depth_range[0], zero_depth_range[1]))


def parse_float_sequence(value: str) -> list[float]:
    tokens = []
    for line in value.splitlines():
        tokens.extend(part.strip() for part in line.split(","))
    return [float(token) for token in tokens if token]


def load_zero_depth_file(path: str | Path) -> list[float]:
    with open(path, "r", encoding="utf-8") as f:
        return parse_float_sequence(f.read())


def fit_sequence_length(values: Sequence[float], target_len: int, mode: str) -> list[float]:
    values = list(values)
    if len(values) == target_len:
        return values
    if mode == "error":
        raise ValueError(f"Expected {target_len} focal-plane values, got {len(values)}.")
    if not values:
        raise ValueError("Zero-depth sequence is empty.")
    if len(values) == 1:
        return values * target_len

    src_x = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    fitted = np.interp(dst_x, src_x, np.asarray(values, dtype=np.float32))
    print(f"Interpolated focal-plane sequence from {len(values)} to {target_len} values.")
    return fitted.astype(np.float32).tolist()


def random_walk_zero_depth(
    start: float,
    num_frames: int,
    delta: float,
    zero_depth_range: Sequence[float],
) -> list[float]:
    zd_min, zd_max = zero_depth_range
    values = [float(np.clip(start, zd_min, zd_max))]
    sign = random.choice([-1, 1])
    for _ in range(num_frames - 1):
        step = random.uniform(0.0, delta)
        next_value = float(np.clip(values[-1] + sign * step, zd_min, zd_max))
        values.append(next_value)
        if next_value <= zd_min or next_value >= zd_max:
            sign *= -1
    return values


def build_zero_depth(args: argparse.Namespace, depth_video: np.ndarray) -> float | list[float]:
    num_frames = depth_video.shape[0]

    if args.zero_depth_file is not None:
        values = load_zero_depth_file(args.zero_depth_file)
        return fit_sequence_length(values, num_frames, args.fit_zero_depth_seq)

    if args.zero_depth_values is not None:
        values = parse_float_sequence(args.zero_depth_values)
        return fit_sequence_length(values, num_frames, args.fit_zero_depth_seq)

    start = args.zero_depth_start
    if start is None:
        start = args.zero_depth
    if start is None:
        start = sample_zero_depth_from_depth(depth_video[0], args.zero_depth_range)

    if args.time_varying:
        if args.zero_depth_end is not None:
            return np.linspace(start, args.zero_depth_end, num_frames, dtype=np.float32).tolist()
        return random_walk_zero_depth(start, num_frames, args.zero_depth_delta, args.zero_depth_range)

    return float(start)


def make_render_batch(
    rgb_video: np.ndarray,
    depth_video: np.ndarray,
    zero_depth: float | Sequence[float],
    K: float,
    gamma: float,
    device: str,
) -> dict[str, torch.Tensor]:
    frames = torch.from_numpy(rgb_video).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    depths = torch.from_numpy(depth_video).permute(0, 3, 1, 2).unsqueeze(0).to(device)

    if isinstance(zero_depth, (list, tuple, np.ndarray)):
        zero_depth_tensor = torch.tensor([zero_depth], dtype=torch.float32, device=device)
    else:
        zero_depth_tensor = torch.tensor([zero_depth], dtype=torch.float32, device=device)

    return {
        "frames": frames.float(),
        "depths": depths.float(),
        "zero_depth": zero_depth_tensor,
        "K": torch.tensor([K], dtype=torch.float32, device=device),
        "gamma": torch.tensor([gamma], dtype=torch.float32, device=device),
    }


def tensor_video_to_numpy(video: torch.Tensor, normalized_minus_one_to_one: bool = True) -> np.ndarray:
    """Convert [T, C, H, W] tensor to uint8 RGB video [T, H, W, C]."""
    if normalized_minus_one_to_one:
        video = (video + 1.0) / 2.0
    video = video.clamp(0.0, 1.0)
    video_np = video.permute(0, 2, 3, 1).cpu().numpy()
    return (video_np * 255.0).round().astype(np.uint8)


def defocus_to_numpy_video(defocus: torch.Tensor) -> np.ndarray:
    """Convert [T, 1, H, W] defocus tensor to uint8 RGB visualization."""
    values = defocus[:, 0].detach().cpu().numpy()
    max_abs = float(np.max(np.abs(values)))
    if max_abs < 1e-8:
        normalized = np.zeros_like(values, dtype=np.float32)
    else:
        normalized = (values / max_abs + 1.0) * 0.5
    gray = (np.clip(normalized, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def save_video_rgb(frames_rgb: np.ndarray, output_path: str | Path, fps: int, quality: int) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(output_path), fps=fps, quality=quality, macro_block_size=1) as writer:
        for frame in frames_rgb:
            writer.append_data(frame)
    print(f"Saved video: {output_path}")


def save_frames_rgb(frames_rgb: np.ndarray, output_video_path: str | Path) -> None:
    output_video_path = Path(output_video_path)
    frames_dir = output_video_path.with_suffix("")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, frame in enumerate(frames_rgb):
        Image.fromarray(frame).save(frames_dir / f"{frame_idx:04d}.png")
    print(f"Saved {len(frames_rgb)} frames: {frames_dir}")


def save_zero_depth_metadata(
    output_path: str | Path,
    zero_depth: float | Sequence[float],
    K: float,
    gamma: float,
) -> None:
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(".json")
    payload = {
        "zero_depth": list(zero_depth) if isinstance(zero_depth, (list, tuple, np.ndarray)) else float(zero_depth),
        "K": float(K),
        "gamma": float(gamma),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metadata: {metadata_path}")


def save_preview(
    rgb_video: np.ndarray,
    depth_video: np.ndarray,
    rendered_video: np.ndarray,
    preview_path: str | None,
    show_preview: bool,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb_video[0])
    axes[0].set_title("RGB")
    axes[1].imshow(rendered_video[0])
    axes[1].set_title("Rendered")
    axes[2].imshow(depth_video[0, ..., 0], cmap="gray")
    axes[2].set_title("Depth")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    if preview_path is not None:
        preview_path = str(preview_path)
        Path(preview_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_path, dpi=150, bbox_inches="tight")
        print(f"Saved preview: {preview_path}")

    if show_preview:
        plt.show()
    else:
        plt.close(fig)


def render_video(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)

    rgb_bgr = read_video_bgr(args.input)
    depth_bgr = read_video_bgr(args.depth_input)
    if rgb_bgr.shape[0] != depth_bgr.shape[0]:
        raise ValueError(f"RGB video has {rgb_bgr.shape[0]} frames, depth video has {depth_bgr.shape[0]} frames.")

    rgb_video = prepare_rgb_video(rgb_bgr, args.width, args.height)
    depth_video = prepare_depth_video(depth_bgr, args.width, args.height)
    zero_depth = build_zero_depth(args, depth_video)

    if isinstance(zero_depth, list):
        print(f"Using time-varying focal_plane: {len(zero_depth)} values, {min(zero_depth):.4f}..{max(zero_depth):.4f}")
    else:
        print(f"Using constant focal_plane: {zero_depth:.4f}")
    print(f"Rendering with K={args.K}, gamma={args.gamma}, device={device}")

    renderer = CombinedBokehRenderer(
        arnet_path=args.arnet_path,
        iunet_path=args.iunet_path,
        defocus_scale=args.defocus_scale,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        device=device,
    )
    batch = make_render_batch(rgb_video, depth_video, zero_depth, args.K, args.gamma, device)

    with torch.no_grad():
        rendered = renderer.render_batch(batch)

    rendered_rgb = tensor_video_to_numpy(rendered["bokeh_pred"][0])
    save_video_rgb(rendered_rgb, args.output, args.fps, args.quality)
    save_zero_depth_metadata(args.output, zero_depth, args.K, args.gamma)

    if args.save_frames:
        save_frames_rgb(rendered_rgb, args.output)

    if args.save_components:
        output_path = Path(args.output)
        stem = output_path.with_suffix("")
        save_video_rgb(tensor_video_to_numpy(rendered["bokeh_classical"][0]), f"{stem}_classical.mp4", args.fps, args.quality)
        save_video_rgb(tensor_video_to_numpy(rendered["bokeh_neural"][0]), f"{stem}_neural.mp4", args.fps, args.quality)
        save_video_rgb(defocus_to_numpy_video(rendered["defocus_map"][0]), f"{stem}_defocus.mp4", args.fps, args.quality)

    if args.preview or args.preview_path:
        save_preview(rgb_video, depth_video, rendered_rgb, args.preview_path, args.preview)


def main() -> None:
    args = parse_args()
    render_video(args)


if __name__ == "__main__":
    main()


"""
# Render bokeh video from RGB video + depth video. you need:
- An RGB video (e.g. simon.mp4)
- A depth video (e.g. simon_depth.mp4) with depth encoded as grayscale (0=near, 1=far). You can create a depth video using a model like Video-Depth-Anything

# How To Run

- Static focal plane, like old tools/preprocessing/create_bokeh_video.py:

python tools/preprocessing/create_bokeh_video.py \
  --input ./data/examples/simon.mp4 \
  --depth-input ./data/examples/simon_depth.mp4 \
  --focal-plane 0.176 \
  --K 21 --gamma 2.0 \
  --output ./results/bokeh/simon_K21_static.mp4 \
  --save-frames

- Time-varying focal plane, like old create_infer_video_timevary_depth.py:

python tools/preprocessing/create_bokeh_video.py \
  --input ./data/examples/breaking_bad3.mp4 \
  --depth-input ./data/examples/breaking_bad3_depth.mp4 \
  --time-varying \
  --focal-plane-start 0.78 \
  --focal-plane-end 0.05 \
  --K 17 --gamma 2.0 \
  --output ./results/bokeh/breaking_bad3_K17_timevary.mp4
"""