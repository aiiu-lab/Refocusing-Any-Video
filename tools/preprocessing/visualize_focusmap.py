"""Visualize focus/defocus maps from a depth video.

The output video maps pixels close to the focal plane to white and pixels far
from the focal plane to black. The focal plane can be a constant scalar, loaded
from a text file, or estimated from the depth video.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


DEFAULT_OUTPUT_NAME = "defocus_heatmap_cinematic.mp4"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_depth_frames(video_path: str | Path) -> tuple[list[np.ndarray], float, int, int]:
    """Read every frame of a depth video as 2D uint8 grayscale arrays."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open depth video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from depth video: {video_path}")

    return frames, fps, width, height


def smooth_series(values: np.ndarray, window: int | None) -> np.ndarray:
    """Smooth a 1D series without zero-padding edge bias."""
    if window is None or window <= 1:
        return values
    window = int(window)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def compute_focal_planes(
    depth_frames: Sequence[np.ndarray],
    mode: str = "center",
    center_frac: float = 0.2,
    smooth_window: int | None = 15,
) -> np.ndarray:
    """Estimate one focal-plane depth value in [0, 1] for each frame.

    Modes:
      center: mean depth of a centered crop
      peak: histogram peak of the full frame
      median: median depth of the full frame
    """
    if not 0.0 < center_frac <= 1.0:
        raise ValueError(f"center_frac must be in (0, 1], got {center_frac}")

    planes = np.zeros(len(depth_frames), dtype=np.float32)
    for i, frame in enumerate(depth_frames):
        depth = frame.astype(np.float32) / 255.0

        if mode == "center":
            h, w = depth.shape
            crop_h = max(1, int(h * center_frac))
            crop_w = max(1, int(w * center_frac))
            y0 = (h - crop_h) // 2
            x0 = (w - crop_w) // 2
            planes[i] = depth[y0:y0 + crop_h, x0:x0 + crop_w].mean()
        elif mode == "peak":
            hist, edges = np.histogram(depth, bins=64, range=(0.0, 1.0))
            peak_idx = int(np.argmax(hist))
            planes[i] = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])
        elif mode == "median":
            planes[i] = float(np.median(depth))
        else:
            raise ValueError(f"Unknown focal-plane mode: {mode}")

    return smooth_series(planes, smooth_window)


def parse_float_sequence(text: str) -> list[float]:
    tokens = []
    for line in text.splitlines():
        tokens.extend(part.strip() for part in line.split(","))
    return [float(token) for token in tokens if token]


def load_focal_plane_file(path: str | Path) -> list[float]:
    with open(path, "r", encoding="utf-8") as f:
        return parse_float_sequence(f.read())


def fit_focal_planes(values: Sequence[float], num_frames: int, mode: str = "interpolate") -> np.ndarray:
    """Fit provided focal-plane values to the depth-video frame count."""
    values = list(values)
    if not values:
        raise ValueError("Focal-plane sequence is empty.")
    if len(values) == num_frames:
        return np.asarray(values, dtype=np.float32)
    if mode == "error":
        raise ValueError(f"Expected {num_frames} focal-plane values, got {len(values)}.")
    if len(values) == 1:
        return np.full(num_frames, float(values[0]), dtype=np.float32)

    src_x = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    fitted = np.interp(dst_x, src_x, np.asarray(values, dtype=np.float32))
    print(f"Interpolated focal-plane sequence from {len(values)} to {num_frames} values.")
    return fitted.astype(np.float32)


def build_focal_planes(args: argparse.Namespace, depth_frames: Sequence[np.ndarray]) -> np.ndarray:
    num_frames = len(depth_frames)
    if args.focal_plane_file is not None:
        return fit_focal_planes(load_focal_plane_file(args.focal_plane_file), num_frames, args.fit_focal_plane_seq)
    if args.focal_plane_values is not None:
        return fit_focal_planes(parse_float_sequence(args.focal_plane_values), num_frames, args.fit_focal_plane_seq)
    if args.focal_plane is not None:
        return np.full(num_frames, args.focal_plane, dtype=np.float32)
    return compute_focal_planes(
        depth_frames,
        mode=args.auto_focal_mode,
        center_frac=args.center_frac,
        smooth_window=args.smooth_window,
    )


def get_cinematic_defocus_palette() -> np.ndarray:
    """Return a 256-entry BGR grayscale LUT: black is far, white is focused."""
    ramp = np.arange(256, dtype=np.uint8)
    return np.stack([ramp, ramp, ramp], axis=-1).reshape(256, 1, 3)


def s_curve(x: np.ndarray, k: float = 0.5, mid_pt: float = 0.5) -> np.ndarray:
    """Piecewise power S-curve mapping x in [0, 1] to [0, 1].

    k = 1.0 is identity. k < 1.0 emphasizes values near 0 and 1; k > 1.0
    makes the curve gentler near the ends. mid_pt controls where the two halves
    meet and must be inside (0, 1).
    """
    if not 0.0 < mid_pt < 1.0:
        raise ValueError(f"mid_pt must be in (0, 1), got {mid_pt}")
    if k <= 0.0:
        raise ValueError(f"s_curve_k must be positive, got {k}")

    x = np.clip(x, 0.0, 1.0)
    lower = mid_pt * np.power(x / mid_pt, k)
    upper = 1.0 - (1.0 - mid_pt) * np.power((1.0 - x) / (1.0 - mid_pt), k)
    return np.where(x < mid_pt, lower, upper)


def _open_ffmpeg_writer(video_path: str | Path, fps: float, width: int, height: int, crf: int = 8, preset: str = "slow"):
    """Spawn ffmpeg to encode raw BGR frames, or return None if unavailable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv444p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _resolve_output_path(output_dir: str | Path, output_name: str = DEFAULT_OUTPUT_NAME) -> Path:
    output_dir = Path(output_dir)
    if output_dir.suffix.lower() in VIDEO_EXTENSIONS:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / output_name


def defocus_heatmap_frame(
    depth_frame: np.ndarray,
    focal_plane: float,
    lut: np.ndarray,
    dist_scale: float = 1.0,
    s_curve_k: float = 1.0,
    mid_pt: float = 0.5,
) -> np.ndarray:
    """Create one BGR visualization frame from one depth frame."""
    if depth_frame.ndim == 3:
        depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
    depth = depth_frame.astype(np.float32) / 255.0

    dist = np.clip(np.abs(depth - focal_plane) * dist_scale, 0.0, 1.0)
    if s_curve_k != 1.0:
        dist = s_curve(dist, k=s_curve_k, mid_pt=mid_pt)
    intensity = ((1.0 - dist) * 255.0).round().astype(np.uint8)
    return cv2.applyColorMap(intensity, lut)


def save_defocus_heatmap(
    depth_frames: Sequence[np.ndarray],
    focal_planes: Sequence[float],
    output_dir: str | Path,
    fps: float,
    height: int,
    width: int,
    dist_scale: float = 1.0,
    s_curve_k: float = 1.0,
    crf: int = 16,
    mid_pt: float = 0.5,
    output_name: str = DEFAULT_OUTPUT_NAME,
    preset: str = "slow",
) -> Path:
    """Save a defocus heatmap video and return its path.

    output_dir can be either a directory or a full .mp4 path.
    """
    if len(depth_frames) != len(focal_planes):
        raise ValueError(f"Depth frame count ({len(depth_frames)}) != focal-plane count ({len(focal_planes)})")

    output_path = _resolve_output_path(output_dir, output_name)
    proc = _open_ffmpeg_writer(output_path, fps, width, height, crf=crf, preset=preset)
    writer = None
    if proc is None:
        print("ffmpeg not found on PATH, falling back to OpenCV mp4v (lower quality).")
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for {output_path}")

    lut = get_cinematic_defocus_palette()
    print("Generating defocus heatmap: focused pixels are white; distant pixels are black.")

    try:
        for frame, focal_plane in zip(depth_frames, focal_planes):
            heatmap = defocus_heatmap_frame(
                frame,
                float(focal_plane),
                lut,
                dist_scale=dist_scale,
                s_curve_k=s_curve_k,
                mid_pt=mid_pt,
            )
            if heatmap.shape[1] != width or heatmap.shape[0] != height:
                heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)

            if proc is not None:
                proc.stdin.write(heatmap.tobytes())
            else:
                writer.write(heatmap)
    finally:
        if proc is not None:
            proc.stdin.close()
            return_code = proc.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
        elif writer is not None:
            writer.release()

    print(f"Heatmap video saved to: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a depth video's focus/defocus map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--depth-input", "--input", required=True, help="Depth video path.")
    parser.add_argument("--output", default=None, help="Output .mp4 path. Overrides --output-dir/--output-name.")
    parser.add_argument("--output-dir", default=None, help="Directory for output video. Defaults to depth input folder.")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)

    focal_group = parser.add_mutually_exclusive_group()
    focal_group.add_argument("--focal-plane", type=float, default=None, help="Constant focal plane in [0, 1].")
    focal_group.add_argument("--focal-plane-file", default=None, help="Text file of focal plane values.")
    focal_group.add_argument("--focal-plane-values", default=None, help="Comma-separated focal plane values.")
    parser.add_argument("--fit-focal-plane-seq", choices=("interpolate", "error"), default="interpolate")

    parser.add_argument("--auto-focal-mode", choices=("center", "peak", "median"), default="center")
    parser.add_argument("--center-frac", type=float, default=0.2)
    parser.add_argument("--smooth-window", type=int, default=15)

    parser.add_argument("--dist-scale", type=float, default=1.0)
    parser.add_argument("--s-curve-k", type=float, default=1.0)
    parser.add_argument("--mid-pt", type=float, default=0.5)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--fps", type=float, default=None, help="Override source video FPS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    depth_path = Path(args.depth_input)
    depth_frames, source_fps, width, height = load_depth_frames(depth_path)
    fps = args.fps or source_fps
    print(f"Loaded {len(depth_frames)} frames at {fps:.2f} fps, {width}x{height}")

    focal_planes = build_focal_planes(args, depth_frames)
    print(f"Focal plane range: {float(focal_planes.min()):.4f} to {float(focal_planes.max()):.4f}")

    output_target = args.output or Path(args.output_dir or depth_path.parent) / args.output_name
    save_defocus_heatmap(
        depth_frames,
        focal_planes,
        output_target,
        fps,
        height,
        width,
        dist_scale=args.dist_scale,
        s_curve_k=args.s_curve_k,
        crf=args.crf,
        mid_pt=args.mid_pt,
        output_name=args.output_name,
        preset=args.preset,
    )


if __name__ == "__main__":
    main()


"""
- Example constant focal plane:

python tools/preprocessing/visualize_focusmap.py \
  --depth-input ./long_video_refocus_example/depth/american_psycho4.mp4 \
  --focal-plane 0.5644 \
  --output-dir ./long_video_refocus_example/focus_map_output \
  --dist-scale 1.0 \
  --s-curve-k 2.0 \
  --crf 4

- Example auto focal plane:

python tools/preprocessing/visualize_focusmap.py \
  --depth-input ./long_video_refocus_example/depth/american_psycho4.mp4 \
  --auto-focal-mode center \
  --center-frac 0.2 \
  --smooth-window 15

"""
