#!/usr/bin/env python3
"""Create a paper figure for facial landmark trajectories and face masking.

The script reads the original video, applies one fixed crop before MediaPipe
inference, and runs ``FaceLandmarker`` in VIDEO mode.  It retains all 478 face
landmarks while visualizing the custom 32-point subset used by this repository.

Two subfigures are produced:

* ``(a) Landmark trajectories`` shows three temporally ordered crops.  Nine
  representative landmarks are connected across crops by dashed lines while
  all 32 custom landmarks remain visible as markers.
* ``(b) Face-region crop mask`` shows the same three original frames as
  separate, non-overlapping cards in time order.  Each card's fixed face crop
  is opaque while the area outside that crop is semi-transparent.

The combined figure and each standalone panel are saved as both PNG and PDF.
By default, landmark arrays, metadata, per-frame binary masks, and individual
RGBA crop cards are also saved to make the result reproducible.

Example
-------
python tools/evaluation/plot_face_landmark_mask_figure.py \
  --input-video ./data/examples/input.mp4 \
  --model-path ./checkpoints/mediapipe/face_landmarker.task \
  --crop 96,300,656,1020 \
  --frames 5 37 74 \
  --output-dir ./results/face_landmark_mask_figure

Frame indices are zero-based.  Crop coordinates are ``x1,y1,x2,y2`` with an
exclusive right and bottom edge, matching NumPy's ``frame[y1:y2, x1:x2]``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# Some compute nodes expose a read-only ~/.config.  Keep Matplotlib's cache in
# a writable temporary directory unless the caller explicitly selected one.
if "MPLCONFIGDIR" not in os.environ:
    matplotlib_cache = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NUM_MEDIAPIPE_LANDMARKS = 478
DEFAULT_CROP = (96, 300, 656, 1020)
DEFAULT_FRAMES = (5, 37, 74)
DEFAULT_OUTSIDE_CROP_ALPHA = 0.30
DEFAULT_FRAME_GAP = 0.015

# Okabe-Ito colors plus distinct marker shapes provide redundant encoding for
# readers with color-vision deficiencies.
LANDMARK_GROUPS = (
    ("Eyebrows", tuple(range(0, 6)), "#E69F00", "^"),
    ("Nose", tuple(range(6, 13)), "#0072B2", "D"),
    ("Eyes", tuple(range(13, 25)), "#009E73", "o"),
    ("Mouth", tuple(range(25, 32)), "#D55E00", "s"),
)

# A sparse, semantically distributed subset keeps temporal correspondence
# legible while all 32 landmarks remain visible as markers in each crop.
TRAJECTORY_CUSTOM_INDICES = (1, 4, 8, 13, 16, 19, 22, 25, 26)

@dataclass(frozen=True)
class CropBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass
class ExtractionResult:
    all_landmarks: np.ndarray
    selected_full_frames: dict[int, np.ndarray]
    selected_crops: dict[int, np.ndarray]
    fps: float
    frame_width: int
    frame_height: int
    missing_frame_indices: list[int]


def parse_crop(value: str) -> CropBox:
    """Parse an x1,y1,x2,y2 crop specification."""
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "crop must contain four integers: x1,y1,x2,y2"
        ) from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError(
            "crop must contain four integers: x1,y1,x2,y2"
        )
    x1, y1, x2, y2 = values
    if min(values) < 0 or x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError(
            "crop coordinates must be non-negative, with x2>x1 and y2>y1"
        )
    return CropBox(x1, y1, x2, y2)


def safe_fps(capture: cv2.VideoCapture) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    return fps if np.isfinite(fps) and fps > 0.0 else 30.0


def load_custom_map_32() -> np.ndarray:
    """Load the repository's canonical custom landmark mapping."""
    try:
        from tools.preprocessing.mediapipe_landmarks import CUSTOM_MAP_32
    except ImportError as exc:
        raise ImportError(
            "MediaPipe and the repository landmark helper are required. "
            "Activate the project's environment and install mediapipe first."
        ) from exc

    mapping = np.asarray(CUSTOM_MAP_32, dtype=np.int64)
    if mapping.shape != (32,):
        raise ValueError(
            "tools.preprocessing.mediapipe_landmarks.CUSTOM_MAP_32 must contain "
            f"32 indices, but found shape {mapping.shape}."
        )
    if np.any(mapping < 0) or np.any(mapping >= NUM_MEDIAPIPE_LANDMARKS):
        raise ValueError("CUSTOM_MAP_32 contains an out-of-range MediaPipe index.")
    return mapping


def validate_requested_frames(frame_indices: Sequence[int]) -> tuple[int, ...]:
    frames = tuple(int(index) for index in frame_indices)
    if len(frames) != 3:
        raise ValueError(
            f"Exactly three frame indices are required; received {len(frames)}."
        )
    if any(index < 0 for index in frames):
        raise ValueError("Frame indices must be zero-based non-negative integers.")
    if tuple(sorted(frames)) != frames or len(set(frames)) != len(frames):
        raise ValueError("--frames must contain three unique indices in time order.")
    return frames


def validate_crop(crop: CropBox, frame_width: int, frame_height: int) -> None:
    if crop.x2 > frame_width or crop.y2 > frame_height:
        raise ValueError(
            f"Crop {crop.as_tuple()} exceeds the input frame size "
            f"{frame_width}x{frame_height}."
        )


def create_landmarker(model_path: Path, min_face_confidence: float):
    """Create a MediaPipe FaceLandmarker configured for VIDEO mode."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required. Activate the project's environment and "
            "install a version that provides MediaPipe Tasks FaceLandmarker."
        ) from exc

    if not model_path.is_file():
        raise FileNotFoundError(f"MediaPipe model file not found: {model_path}")

    options = FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=min_face_confidence,
        min_face_presence_confidence=min_face_confidence,
        min_tracking_confidence=min_face_confidence,
    )
    return mp, FaceLandmarker.create_from_options(options)


def result_to_crop_landmarks(result, crop: CropBox) -> np.ndarray:
    """Convert one MediaPipe result to 478 pixel coordinates in crop space."""
    coordinates = np.full(
        (NUM_MEDIAPIPE_LANDMARKS, 2), np.nan, dtype=np.float32
    )
    if not result.face_landmarks:
        return coordinates

    landmarks = result.face_landmarks[0]
    count = min(NUM_MEDIAPIPE_LANDMARKS, len(landmarks))
    for index in range(count):
        point = landmarks[index]
        coordinates[index] = (
            float(point.x) * crop.width,
            float(point.y) * crop.height,
        )
    return coordinates


def extract_video_landmarks(
    input_video: Path,
    model_path: Path,
    crop: CropBox,
    requested_frames: Sequence[int],
    min_face_confidence: float,
) -> ExtractionResult:
    """Run VIDEO-mode detection over the original video's fixed face crop."""
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video}")

    fps = safe_fps(capture)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    validate_crop(crop, frame_width, frame_height)

    keep_frames = set(requested_frames)
    selected_full_frames: dict[int, np.ndarray] = {}
    selected_crops: dict[int, np.ndarray] = {}
    frame_landmarks: list[np.ndarray] = []
    missing_frame_indices: list[int] = []
    frame_index = 0
    previous_timestamp_ms = -1

    mp, landmarker = create_landmarker(model_path, min_face_confidence)
    try:
        with landmarker:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                crop_bgr = frame_bgr[crop.y1 : crop.y2, crop.x1 : crop.x2]
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(crop_rgb),
                )
                timestamp_ms = int(round(frame_index * 1000.0 / fps))
                timestamp_ms = max(timestamp_ms, previous_timestamp_ms + 1)
                previous_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                coordinates = result_to_crop_landmarks(result, crop)
                if not np.isfinite(coordinates).any():
                    missing_frame_indices.append(frame_index)
                frame_landmarks.append(coordinates)

                if frame_index in keep_frames:
                    selected_full_frames[frame_index] = frame_bgr.copy()
                    selected_crops[frame_index] = crop_bgr.copy()
                frame_index += 1
    finally:
        capture.release()

    if not frame_landmarks:
        raise RuntimeError(f"No readable frames were found in {input_video}.")

    missing_requested = sorted(keep_frames - selected_full_frames.keys())
    if missing_requested:
        raise IndexError(
            f"Requested frame(s) {missing_requested} are outside the decoded "
            f"video range 0..{len(frame_landmarks) - 1}."
        )

    all_landmarks = np.stack(frame_landmarks).astype(np.float32)
    return ExtractionResult(
        all_landmarks=all_landmarks,
        selected_full_frames=selected_full_frames,
        selected_crops=selected_crops,
        fps=fps,
        frame_width=frame_width,
        frame_height=frame_height,
        missing_frame_indices=missing_frame_indices,
    )


def require_landmarks(
    all_landmarks: np.ndarray, frame_indices: Sequence[int], description: str
) -> None:
    missing = [
        index
        for index in frame_indices
        if not np.isfinite(all_landmarks[index]).all()
    ]
    if missing:
        raise RuntimeError(
            f"MediaPipe did not return a complete face mesh for {description} "
            f"frame(s) {missing}. Select other frames or lower "
            "--min-face-confidence."
        )


def rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def landmark_group_for_point(point_index: int) -> tuple[str, tuple[int, ...], str, str]:
    for group in LANDMARK_GROUPS:
        if point_index in group[1]:
            return group
    raise KeyError(f"Custom landmark {point_index} is not assigned to a group.")


def draw_landmark_panel(
    ax,
    crops_bgr: Sequence[np.ndarray],
    custom_landmarks: Sequence[np.ndarray],
    frame_indices: Sequence[int],
) -> None:
    """Draw the three-crop trajectory panel onto one Matplotlib axis."""
    crop_height, crop_width = crops_bgr[0].shape[:2]
    if any(crop.shape[:2] != (crop_height, crop_width) for crop in crops_bgr):
        raise ValueError("All selected crops must have the same dimensions.")

    gap = max(12, int(round(crop_width * 0.045)))
    composite_width = len(crops_bgr) * crop_width + (len(crops_bgr) - 1) * gap
    composite = np.full((crop_height, composite_width, 3), 245, dtype=np.uint8)
    x_offsets: list[int] = []
    for crop_index, crop_bgr in enumerate(crops_bgr):
        offset = crop_index * (crop_width + gap)
        x_offsets.append(offset)
        composite[:, offset : offset + crop_width] = rgb(crop_bgr)

    ax.imshow(composite, interpolation="none")

    # Draw connections first, then points, so landmark markers remain crisp.
    for point_index in TRAJECTORY_CUSTOM_INDICES:
        _, _, color, _ = landmark_group_for_point(point_index)
        for time_index in range(len(frame_indices) - 1):
            point_a = custom_landmarks[time_index][point_index]
            point_b = custom_landmarks[time_index + 1][point_index]
            if not (np.isfinite(point_a).all() and np.isfinite(point_b).all()):
                continue
            ax.plot(
                [x_offsets[time_index] + point_a[0], x_offsets[time_index + 1] + point_b[0]],
                [point_a[1], point_b[1]],
                color=color,
                linewidth=0.65,
                linestyle=(0, (3.0, 2.5)),
                alpha=0.68,
                zorder=2,
            )

    for time_index, frame_landmarks in enumerate(custom_landmarks):
        for _, point_indices, color, marker in LANDMARK_GROUPS:
            points = frame_landmarks[np.asarray(point_indices)]
            finite = np.isfinite(points).all(axis=1)
            if not finite.any():
                continue
            ax.scatter(
                x_offsets[time_index] + points[finite, 0],
                points[finite, 1],
                s=15,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )

    for offset in x_offsets:
        ax.add_patch(
            Rectangle(
                (offset - 0.5, -0.5),
                crop_width,
                crop_height,
                fill=False,
                edgecolor="white",
                linewidth=0.9,
                zorder=4,
            )
        )
    ax.set_title(
        "(a) Landmark trajectories",
        loc="left",
        fontsize=10,
        fontweight="semibold",
        pad=4,
    )
    ax.set_xlim(-0.5, composite_width - 0.5)
    ax.set_ylim(crop_height - 0.5, -0.5)
    ax.axis("off")


def make_crop_region_mask(
    frame_height: int, frame_width: int, crop: CropBox
) -> np.ndarray:
    """Return a binary mask for the fixed rectangular face crop."""
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    mask[crop.y1 : crop.y2, crop.x1 : crop.x2] = 255
    return mask


def make_retained_crop_rgba(
    frame_bgr: np.ndarray,
    crop: CropBox,
    outside_crop_alpha: float,
) -> np.ndarray:
    """Make an RGBA card with an opaque crop and translucent surroundings."""
    frame_rgb = rgb(frame_bgr)
    alpha = np.full(
        frame_rgb.shape[:2],
        int(round(255.0 * outside_crop_alpha)),
        dtype=np.uint8,
    )
    alpha[crop.y1 : crop.y2, crop.x1 : crop.x2] = 255
    return np.dstack((frame_rgb, alpha))


def alpha_composite_at(
    canvas_rgb: np.ndarray, card_rgba: np.ndarray, x_offset: int, y_offset: int
) -> None:
    """Alpha-composite one RGBA card onto an RGB canvas in-place."""
    height, width = card_rgba.shape[:2]
    destination = canvas_rgb[
        y_offset : y_offset + height, x_offset : x_offset + width
    ].astype(np.float32)
    source = card_rgba[..., :3].astype(np.float32)
    alpha = card_rgba[..., 3:4].astype(np.float32) / 255.0
    blended = source * alpha + destination * (1.0 - alpha)
    canvas_rgb[y_offset : y_offset + height, x_offset : x_offset + width] = (
        np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    )


def build_frame_strip(
    frames_bgr: Sequence[np.ndarray],
    crop: CropBox,
    outside_crop_alpha: float,
    frame_gap: float,
) -> np.ndarray:
    """Build a chronological row of separate, non-overlapping RGBA cards."""
    frame_height, frame_width = frames_bgr[0].shape[:2]
    if any(frame.shape[:2] != (frame_height, frame_width) for frame in frames_bgr):
        raise ValueError("All source frames must have the same dimensions.")

    gap = max(1, int(round(frame_width * frame_gap)))
    strip_width = len(frames_bgr) * frame_width + (len(frames_bgr) - 1) * gap
    canvas = np.full((frame_height, strip_width, 3), 255, dtype=np.uint8)

    # Time advances left to right.  White gaps keep every frame distinct.
    for time_index, frame_bgr in enumerate(frames_bgr):
        x_offset = time_index * (frame_width + gap)
        card_rgba = make_retained_crop_rgba(frame_bgr, crop, outside_crop_alpha)
        alpha_composite_at(canvas, card_rgba, x_offset, 0)
    return canvas


def draw_mask_panel(
    ax,
    frames_bgr: Sequence[np.ndarray],
    crop: CropBox,
    outside_crop_alpha: float,
    frame_gap: float,
) -> None:
    """Draw three masked source frames separately in chronological order."""
    strip = build_frame_strip(frames_bgr, crop, outside_crop_alpha, frame_gap)
    ax.imshow(strip, interpolation="none")
    ax.set_title(
        "(b) Face-region crop mask",
        loc="left",
        fontsize=10,
        fontweight="semibold",
        pad=4,
    )
    ax.set_xlim(-0.5, strip.shape[1] - 0.5)
    ax.set_ylim(strip.shape[0] - 0.5, -0.5)
    ax.axis("off")


def save_figure(fig, png_path: Path, pdf_path: Path, dpi: int) -> None:
    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(fig)


def render_figures(
    output_dir: Path,
    output_stem: str,
    crops_bgr: Sequence[np.ndarray],
    custom_landmarks: Sequence[np.ndarray],
    frame_indices: Sequence[int],
    source_frames_bgr: Sequence[np.ndarray],
    crop: CropBox,
    outside_crop_alpha: float,
    frame_gap: float,
    figure_width: float,
    dpi: int,
) -> dict[str, Path]:
    """Render combined and standalone versions of both requested panels."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    output_paths = {
        "combined_png": output_dir / f"{output_stem}.png",
        "combined_pdf": output_dir / f"{output_stem}.pdf",
        "panel_a_png": output_dir / f"{output_stem}_a_landmark_trajectories.png",
        "panel_a_pdf": output_dir / f"{output_stem}_a_landmark_trajectories.pdf",
        "panel_b_png": output_dir / f"{output_stem}_b_face_region_mask.png",
        "panel_b_pdf": output_dir / f"{output_stem}_b_face_region_mask.pdf",
    }

    crop_height, crop_width = crops_bgr[0].shape[:2]
    panel_a_aspect = (3 * crop_width + 2 * crop_width * 0.045) / crop_height
    frame_height, frame_width = source_frames_bgr[0].shape[:2]
    panel_b_aspect = (
        3 * frame_width + 2 * frame_width * frame_gap
    ) / frame_height
    combined_height = (
        figure_width / panel_a_aspect + figure_width / panel_b_aspect + 0.65
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(figure_width, combined_height),
        gridspec_kw={
            "height_ratios": [1.0 / panel_a_aspect, 1.0 / panel_b_aspect],
            "hspace": 0.09,
        },
    )
    draw_landmark_panel(axes[0], crops_bgr, custom_landmarks, frame_indices)
    draw_mask_panel(
        axes[1],
        source_frames_bgr,
        crop,
        outside_crop_alpha,
        frame_gap,
    )
    save_figure(
        fig,
        output_paths["combined_png"],
        output_paths["combined_pdf"],
        dpi,
    )

    panel_a_height = figure_width / panel_a_aspect + 0.45
    fig_a, ax_a = plt.subplots(figsize=(figure_width, panel_a_height))
    draw_landmark_panel(ax_a, crops_bgr, custom_landmarks, frame_indices)
    save_figure(
        fig_a,
        output_paths["panel_a_png"],
        output_paths["panel_a_pdf"],
        dpi,
    )

    panel_b_height = figure_width / panel_b_aspect + 0.45
    fig_b, ax_b = plt.subplots(figsize=(figure_width, panel_b_height))
    draw_mask_panel(
        ax_b,
        source_frames_bgr,
        crop,
        outside_crop_alpha,
        frame_gap,
    )
    save_figure(
        fig_b,
        output_paths["panel_b_png"],
        output_paths["panel_b_pdf"],
        dpi,
    )
    return output_paths


def save_rgba_pdf(card_rgba: np.ndarray, output_path: Path) -> None:
    """Save one retained-crop RGBA card as a borderless PDF page."""
    height, width = card_rgba.shape[:2]
    pdf_width = 6.0
    fig, ax = plt.subplots(figsize=(pdf_width, pdf_width * height / width))
    ax.imshow(card_rgba, interpolation="none")
    ax.axis("off")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0.0,
        transparent=True,
    )
    plt.close(fig)


def save_mask_intermediates(
    output_dir: Path,
    output_stem: str,
    selected_full_frames: dict[int, np.ndarray],
    frame_indices: Sequence[int],
    crop: CropBox,
    outside_crop_alpha: float,
) -> dict[str, Path]:
    """Save each binary mask and individual retained-crop PNG/PDF card."""
    paths: dict[str, Path] = {}
    first_frame = selected_full_frames[frame_indices[0]]
    frame_height, frame_width = first_frame.shape[:2]
    binary_mask = make_crop_region_mask(frame_height, frame_width, crop)

    for frame_index in frame_indices:
        frame_bgr = selected_full_frames[frame_index]
        mask_path = (
            output_dir
            / f"{output_stem}_binary_crop_mask_frame_{frame_index:04d}.png"
        )
        card_png_path = (
            output_dir / f"{output_stem}_retained_crop_frame_{frame_index:04d}.png"
        )
        card_pdf_path = (
            output_dir / f"{output_stem}_retained_crop_frame_{frame_index:04d}.pdf"
        )
        if not cv2.imwrite(str(mask_path), binary_mask):
            raise RuntimeError(f"Could not write binary mask: {mask_path}")

        card_rgba = make_retained_crop_rgba(
            frame_bgr, crop, outside_crop_alpha
        )
        card_bgra = cv2.cvtColor(card_rgba, cv2.COLOR_RGBA2BGRA)
        if not cv2.imwrite(str(card_png_path), card_bgra):
            raise RuntimeError(f"Could not write RGBA crop card: {card_png_path}")
        save_rgba_pdf(card_rgba, card_pdf_path)

        suffix = f"frame_{frame_index:04d}"
        paths[f"binary_mask_{suffix}"] = mask_path
        paths[f"retained_crop_png_{suffix}"] = card_png_path
        paths[f"retained_crop_pdf_{suffix}"] = card_pdf_path
    return paths


def save_intermediate_data(
    output_dir: Path,
    output_stem: str,
    extraction: ExtractionResult,
    custom_landmarks: np.ndarray,
    custom_map: np.ndarray,
    crop: CropBox,
    input_video: Path,
    model_path: Path,
    frame_indices: Sequence[int],
    outside_crop_alpha: float,
    frame_gap: float,
    figure_paths: dict[str, Path],
    deprecated_mask_frame: int | None,
    deprecated_stack_x_shift: float | None,
    deprecated_stack_y_shift: float | None,
) -> dict[str, Path]:
    all_path = output_dir / f"{output_stem}_landmarks_all_478.npy"
    custom_path = output_dir / f"{output_stem}_landmarks_custom_32.npy"
    metadata_path = output_dir / f"{output_stem}_metadata.json"
    np.save(all_path, extraction.all_landmarks)
    np.save(custom_path, custom_landmarks)
    mask_paths = save_mask_intermediates(
        output_dir,
        output_stem,
        extraction.selected_full_frames,
        frame_indices,
        crop,
        outside_crop_alpha,
    )

    metadata = {
        "input_video": str(input_video.resolve()),
        "model_path": str(model_path.resolve()),
        "frame_numbering": "zero-based",
        "decoded_frame_count": int(extraction.all_landmarks.shape[0]),
        "fps": extraction.fps,
        "input_size": [extraction.frame_width, extraction.frame_height],
        "crop_xyxy_exclusive": list(crop.as_tuple()),
        "trajectory_frames": list(frame_indices),
        "trajectory_custom_indices": list(TRAJECTORY_CUSTOM_INDICES),
        "mask_frames": list(frame_indices),
        "missing_detection_frames": extraction.missing_frame_indices,
        "custom_map_32": custom_map.tolist(),
        "outside_crop_alpha": outside_crop_alpha,
        "retained_crop_alpha": 1.0,
        "mask_kind": "RGBA card: opaque crop, semi-transparent surroundings",
        "panel_b_layout": "horizontal_nonoverlapping",
        "frame_gap_fraction": frame_gap,
        "deprecated_mask_frame_argument": deprecated_mask_frame,
        "deprecated_stack_x_shift_argument": deprecated_stack_x_shift,
        "deprecated_stack_y_shift_argument": deprecated_stack_y_shift,
        "landmarks_coordinate_system": "pixel coordinates relative to fixed crop",
        "outputs": {
            name: str(path.resolve())
            for name, path in {**figure_paths, **mask_paths}.items()
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "all_landmarks": all_path,
        "custom_landmarks": custom_path,
        **mask_paths,
        "metadata": metadata_path,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create two paper-ready panels showing MediaPipe landmark "
            "trajectories and a semi-transparent face-region mask."
        )
    )
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("./checkpoints/mediapipe/face_landmarker.task"),
        help="MediaPipe face_landmarker.task file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results/face_landmark_mask_figure"),
    )
    parser.add_argument(
        "--output-stem",
        default="face_landmark_mask_figure",
        help="Base filename for PNG, PDF, and intermediate outputs.",
    )
    parser.add_argument(
        "--crop",
        type=parse_crop,
        default=CropBox(*DEFAULT_CROP),
        metavar="X1,Y1,X2,Y2",
        help=(
            "Fixed face crop applied before MediaPipe inference; right and "
            "bottom coordinates are exclusive (default: 96,300,656,1020)."
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs=3,
        default=list(DEFAULT_FRAMES),
        metavar=("FIRST", "MIDDLE", "LAST"),
        help="Three increasing zero-based frame indices (default: 5 37 74).",
    )
    parser.add_argument(
        "--mask-frame",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--min-face-confidence",
        type=float,
        default=0.5,
        help="Detection, presence, and tracking confidence threshold.",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--outside-crop-alpha",
        type=float,
        default=DEFAULT_OUTSIDE_CROP_ALPHA,
        help=(
            "RGBA opacity outside the retained crop in [0,1] "
            "(default: 0.30); the retained crop is fully opaque."
        ),
    )
    parser.add_argument(
        "--frame-gap",
        type=float,
        default=DEFAULT_FRAME_GAP,
        help=(
            "Gap between the three non-overlapping mask frames as a fraction "
            "of frame width (default: 0.015)."
        ),
    )
    parser.add_argument(
        "--stack-x-shift",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stack-y-shift",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=7.2,
        help="Combined and standalone figure width in inches (default: 7.2).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    parser.add_argument(
        "--no-save-intermediates",
        action="store_true",
        help="Do not save NPY landmark arrays, metadata, or raster masks.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    frames = validate_requested_frames(args.frames)
    if not 0.0 <= args.min_face_confidence <= 1.0:
        raise ValueError("--min-face-confidence must be in [0,1].")
    if not 0.0 <= args.outside_crop_alpha <= 1.0:
        raise ValueError("--outside-crop-alpha must be in [0,1].")
    if not 0.0 <= args.frame_gap <= 0.25:
        raise ValueError("--frame-gap must be in [0,0.25].")
    if args.figure_width <= 0.0:
        raise ValueError("--figure-width must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if not args.output_stem.strip():
        raise ValueError("--output-stem cannot be empty.")
    return frames


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frame_indices = validate_args(args)
    custom_map = load_custom_map_32()

    if args.mask_frame is not None or args.mask_alpha is not None:
        print(
            "Note: --mask-frame and --mask-alpha are deprecated and ignored; "
            "panel (b) uses all three --frames with an opaque retained crop."
        )
    if args.stack_x_shift is not None or args.stack_y_shift is not None:
        print(
            "Note: --stack-x-shift and --stack-y-shift are deprecated and "
            "ignored; panel (b) now shows three non-overlapping frames."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading original video: {args.input_video}")
    print(f"MediaPipe input crop: {args.crop.as_tuple()}")
    extraction = extract_video_landmarks(
        args.input_video,
        args.model_path,
        args.crop,
        frame_indices,
        args.min_face_confidence,
    )
    require_landmarks(
        extraction.all_landmarks,
        frame_indices,
        "selected",
    )

    custom_landmarks_all = extraction.all_landmarks[:, custom_map]
    selected_crops = [extraction.selected_crops[index] for index in frame_indices]
    selected_custom_landmarks = [
        custom_landmarks_all[index] for index in frame_indices
    ]
    selected_full_frames = [
        extraction.selected_full_frames[index] for index in frame_indices
    ]

    figure_paths = render_figures(
        args.output_dir,
        args.output_stem,
        selected_crops,
        selected_custom_landmarks,
        frame_indices,
        selected_full_frames,
        args.crop,
        args.outside_crop_alpha,
        args.frame_gap,
        args.figure_width,
        args.dpi,
    )

    intermediate_paths: dict[str, Path] = {}
    if not args.no_save_intermediates:
        intermediate_paths = save_intermediate_data(
            args.output_dir,
            args.output_stem,
            extraction,
            custom_landmarks_all,
            custom_map,
            args.crop,
            args.input_video,
            args.model_path,
            frame_indices,
            args.outside_crop_alpha,
            args.frame_gap,
            figure_paths,
            args.mask_frame,
            args.stack_x_shift,
            args.stack_y_shift,
        )

    detected_count = (
        extraction.all_landmarks.shape[0] - len(extraction.missing_frame_indices)
    )
    print(
        f"MediaPipe detections: {detected_count}/"
        f"{extraction.all_landmarks.shape[0]} frames"
    )
    print("Created:")
    for path in (*figure_paths.values(), *intermediate_paths.values()):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
