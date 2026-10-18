#!/usr/bin/env python3
"""Plot variance of Laplacian over an image sequence.

The measurement region is selectable. ``face`` mode anchors one person,
matches that identity with InsightFace, and measures an aligned 112x112 face
crop. ``full-frame`` mode measures every readable frame at its native
resolution and does not load or depend on InsightFace.

The sharpness score uses the explicit four-neighbour Laplacian kernel::

    0   1   0
    1  -4   1
    0   1   0

Each measurement image is converted to 8-bit grayscale, filtered into CV_64F
with reflected borders, and reduced with population variance (``ddof=0``).

The output figure follows a timeline layout: the raw per-frame curve is shown
above four representative images from the selected measurement region.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2

# Some compute nodes expose a read-only ~/.config. Put Matplotlib's small font
# cache in a per-user temporary directory unless the caller chose a location.
if "MPLCONFIGDIR" not in os.environ:
    matplotlib_cache = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSIGHTFACE_ROOT = REPO_ROOT / "checkpoints" / "insightface"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LAPLACIAN_KERNEL = np.asarray(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


@dataclass
class MetricRecord:
    sequence_index: int
    frame_index: int
    source_frame_index: int | None
    filename: str
    variance: float
    bbox: np.ndarray | None
    detection_score: float | None
    cosine_similarity: float | None
    alignment_padding_fraction: float | None
    pitch_degrees: float | None
    yaw_degrees: float | None
    roll_degrees: float | None
    crop_path: Path | None
    thumbnail_path: Path
    measurement_region: str
    measurement_width: int
    measurement_height: int


def natural_sort_key(path: Path) -> list[int | str]:
    """Sort frame_2 before frame_10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def frame_number(path: Path, fallback: int) -> int:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else fallback


def parse_point(value: str) -> tuple[float, float]:
    try:
        x_text, y_text = value.split(",", maxsplit=1)
        return float(x_text), float(y_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected X,Y, for example 375,145") from exc


def list_frames(frames_dir: Path) -> list[Path]:
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frame directory does not exist: {frames_dir}")
    paths = sorted(
        (
            path
            for path in frames_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )
    if not paths:
        raise ValueError(f"No supported images found in: {frames_dir}")
    return paths


def resolve_providers(mode: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required. Activate the project's refocus environment first."
        ) from exc

    available = ort.get_available_providers()
    if mode == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. Use --provider cpu or install a compatible onnxruntime-gpu."
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif mode == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )

    # PyTorch CUDA wheels keep CUDA libraries inside site-packages. ORT 1.21+
    # can preload those libraries before the InsightFace sessions are created.
    if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    return providers


def load_face_analyzer(
    insightface_root: Path,
    model_name: str,
    provider_mode: str,
    detector_size: int,
    detection_threshold: float,
    require_pose: bool,
):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise ImportError(
            "insightface is required. Install it in the project's refocus environment."
        ) from exc

    model_dir = insightface_root / "models" / model_name
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"InsightFace model directory not found: {model_dir}. "
            "Pass --insightface-root if the models are stored elsewhere."
        )

    providers = resolve_providers(provider_mode)
    allowed_modules = ["detection", "recognition"]
    if require_pose:
        allowed_modules.append("landmark_3d_68")
    analyzer = FaceAnalysis(
        name=model_name,
        root=str(insightface_root),
        allowed_modules=allowed_modules,
        providers=providers,
    )
    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    analyzer.prepare(
        ctx_id=ctx_id,
        det_thresh=detection_threshold,
        det_size=(detector_size, detector_size),
    )
    print(f"InsightFace providers: {providers}")
    return analyzer


def face_bbox(face) -> np.ndarray:
    return np.asarray(face.bbox, dtype=np.float32)


def bbox_area(bbox: np.ndarray) -> float:
    return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = bbox_area(a) + bbox_area(b) - intersection
    return intersection / union if union > 0.0 else 0.0


def normalized_embedding(face) -> np.ndarray | None:
    embedding = getattr(face, "normed_embedding", None)
    if embedding is None:
        embedding = getattr(face, "embedding", None)
    if embedding is None:
        return None
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(embedding))
    return embedding / norm if norm > 0.0 else None


def face_pose_degrees(face) -> tuple[float, float, float] | None:
    """Return InsightFace pose as (pitch, yaw, roll), in degrees."""
    pose = getattr(face, "pose", None)
    if pose is None:
        return None
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    if pose.size != 3 or not np.all(np.isfinite(pose)):
        return None
    return float(pose[0]), float(pose[1]), float(pose[2])


def choose_anchor_face(faces: Sequence, target_point: tuple[float, float] | None):
    usable = [face for face in faces if normalized_embedding(face) is not None]
    if not usable:
        raise RuntimeError(
            "No face with an identity embedding was detected in the target frame."
        )
    if target_point is None:
        return max(usable, key=lambda face: bbox_area(face_bbox(face)))

    px, py = target_point

    def point_distance(face) -> tuple[int, float]:
        bbox = face_bbox(face)
        inside = bbox[0] <= px <= bbox[2] and bbox[1] <= py <= bbox[3]
        center_x = 0.5 * (float(bbox[0]) + float(bbox[2]))
        center_y = 0.5 * (float(bbox[1]) + float(bbox[3]))
        distance = (center_x - px) ** 2 + (center_y - py) ** 2
        return (0 if inside else 1, distance)

    return min(usable, key=point_distance)


def select_target_face(
    faces: Sequence,
    anchor_embedding: np.ndarray,
    anchor_area: float,
    previous_bbox: np.ndarray | None,
    frame_shape: tuple[int, int],
    min_similarity: float,
    min_area_ratio: float,
):
    """Match the anchor identity; continuity only acts as a small tie-breaker."""
    height, width = frame_shape
    frame_diagonal = float(np.hypot(width, height))
    candidates = []

    for face in faces:
        embedding = normalized_embedding(face)
        if embedding is None:
            continue
        bbox = face_bbox(face)
        if bbox_area(bbox) < anchor_area * min_area_ratio:
            continue
        similarity = float(np.dot(anchor_embedding, embedding))
        if similarity < min_similarity:
            continue

        continuity_iou = (
            bbox_iou(previous_bbox, bbox) if previous_bbox is not None else 0.0
        )
        if previous_bbox is not None:
            prev_center = 0.5 * (previous_bbox[:2] + previous_bbox[2:])
            center = 0.5 * (bbox[:2] + bbox[2:])
            center_score = max(
                0.0, 1.0 - float(np.linalg.norm(center - prev_center)) / frame_diagonal
            )
        else:
            center_score = 0.0

        combined_score = similarity + 0.10 * continuity_iou + 0.03 * center_score
        candidates.append((combined_score, similarity, face))

    if not candidates:
        return None, None
    _, similarity, face = max(candidates, key=lambda item: item[0])
    return face, similarity


def align_face(
    frame_bgr: np.ndarray, face, crop_size: int = 112
) -> tuple[np.ndarray, float]:
    try:
        from insightface.utils import face_align
    except ImportError as exc:
        raise ImportError("Could not import insightface.utils.face_align.") from exc

    landmarks = np.asarray(face.kps, dtype=np.float32)
    if landmarks.shape != (5, 2):
        raise ValueError(f"Expected five 2D landmarks, got {landmarks.shape}.")
    crop = face_align.norm_crop(frame_bgr, landmark=landmarks, image_size=crop_size)

    # Measure which aligned pixels originate outside the source frame. Those
    # pixels are filled with black by norm_crop and would create a false,
    # high-contrast Laplacian edge. Warp an all-valid mask with the identical
    # transform so real black pixels in the image are not mistaken for padding.
    transform_result = face_align.estimate_norm(landmarks, image_size=crop_size)
    # InsightFace 1.x returns only M; some newer forks return (M, pose_index).
    transform = (
        transform_result[0] if isinstance(transform_result, tuple) else transform_result
    )
    source_mask = np.full(frame_bgr.shape[:2], 255, dtype=np.uint8)
    aligned_mask = cv2.warpAffine(
        source_mask,
        transform,
        (crop_size, crop_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    padding_fraction = float(np.mean(aligned_mask == 0))
    return crop, padding_fraction


def variance_of_laplacian(image_bgr: np.ndarray) -> float:
    """Return population variance of the explicit cross-kernel response."""
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.filter2D(
        grayscale,
        ddepth=cv2.CV_64F,
        kernel=LAPLACIAN_KERNEL,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return float(np.var(laplacian, ddof=0))


def discover_source_indices(
    frames_dir: Path, params_json: Path | None, count: int
) -> list[int | None]:
    if params_json is None:
        candidate = frames_dir.parent.parent / "params_used.json"
        params_json = candidate if candidate.is_file() else None
    if params_json is None:
        return [None] * count

    try:
        data = json.loads(params_json.read_text(encoding="utf-8"))
        indices = data.get("frame_indices")
        if not isinstance(indices, list) or len(indices) != count:
            print(f"Warning: ignoring incompatible frame_indices in {params_json}")
            return [None] * count
        return [int(value) for value in indices]
    except (OSError, ValueError, TypeError) as exc:
        print(f"Warning: could not load source frame indices from {params_json}: {exc}")
        return [None] * count


def write_csv(records: Sequence[MetricRecord], output_path: Path) -> None:
    fieldnames = [
        "sequence_index",
        "frame_index",
        "source_frame_index",
        "filename",
        "variance_of_laplacian",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "detection_score",
        "cosine_similarity",
        "alignment_padding_fraction",
        "pitch_degrees",
        "yaw_degrees",
        "roll_degrees",
        "aligned_crop",
        "measurement_region",
        "measurement_width",
        "measurement_height",
    ]

    def format_float(value: float | None, precision: int) -> str:
        if value is None:
            return ""
        return f"{value:.{precision}f}"

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sequence_index": record.sequence_index,
                    "frame_index": record.frame_index,
                    "source_frame_index": ""
                    if record.source_frame_index is None
                    else record.source_frame_index,
                    "filename": record.filename,
                    "variance_of_laplacian": f"{record.variance:.8f}",
                    "bbox_x1": "" if record.bbox is None else f"{record.bbox[0]:.3f}",
                    "bbox_y1": "" if record.bbox is None else f"{record.bbox[1]:.3f}",
                    "bbox_x2": "" if record.bbox is None else f"{record.bbox[2]:.3f}",
                    "bbox_y2": "" if record.bbox is None else f"{record.bbox[3]:.3f}",
                    "detection_score": format_float(record.detection_score, 6),
                    "cosine_similarity": format_float(record.cosine_similarity, 6),
                    "alignment_padding_fraction": format_float(
                        record.alignment_padding_fraction, 6
                    ),
                    "pitch_degrees": format_float(record.pitch_degrees, 6),
                    "yaw_degrees": format_float(record.yaw_degrees, 6),
                    "roll_degrees": format_float(record.roll_degrees, 6),
                    "aligned_crop": ""
                    if record.crop_path is None
                    else str(record.crop_path),
                    "measurement_region": record.measurement_region,
                    "measurement_width": record.measurement_width,
                    "measurement_height": record.measurement_height,
                }
            )


def representative_records(
    records: Sequence[MetricRecord], count: int
) -> list[MetricRecord]:
    count = min(count, len(records))
    if count <= 0:
        return []
    positions = np.linspace(0, len(records) - 1, count)
    indices = np.rint(positions).astype(int)
    return [records[index] for index in indices]


def plot_timeline(
    records: Sequence[MetricRecord],
    total_frames: int,
    frame_start: int,
    frame_end: int,
    output_png: Path,
    output_pdf: Path,
    thumbnail_count: int,
    dpi: int,
    measurement_region: str,
) -> None:
    if not records:
        raise ValueError("No valid measurements are available to plot.")

    scores = np.full(total_frames, np.nan, dtype=np.float64)
    for record in records:
        scores[record.sequence_index] = record.variance

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.linewidth": 1.0,
        }
    )
    figure = plt.figure(figsize=(9.0, 6.2), facecolor="white")

    curve_axis = figure.add_axes([0.137, 0.55, 0.813, 0.39])
    curve_axis.plot(
        np.arange(total_frames),
        scores,
        color="#001ee6",
        linewidth=1.8,
    )
    curve_axis.set_xlabel("Frame index", fontsize=17)
    y_axis_label = "Variance of Laplacian"
    if measurement_region == "face":
        y_axis_label += "\n(aligned face)"
    curve_axis.set_ylabel(y_axis_label, fontsize=16)
    curve_axis.tick_params(axis="both", labelsize=11)
    curve_axis.grid(True, linestyle="--", linewidth=0.8, alpha=0.5, color="#b8b8b8")
    curve_axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    if frame_start == frame_end:
        curve_axis.set_xlim(frame_start - 0.5, frame_end + 0.5)
    else:
        curve_axis.set_xlim(frame_start, frame_end)

    finite_scores = scores[np.isfinite(scores)]
    value_min = float(finite_scores.min())
    value_max = float(finite_scores.max())
    value_range = value_max - value_min
    margin = max(value_range * 0.06, abs(value_max) * 0.01, 1e-6)
    curve_axis.set_ylim(value_min - margin, value_max + margin)

    thumbnails = representative_records(records, thumbnail_count)
    panel_left = 0.052
    panel_right = 0.970
    panel_gap = 0.004
    panel_count = len(thumbnails)
    panel_width = (
        panel_right - panel_left - panel_gap * (panel_count - 1)
    ) / panel_count
    sample_bgr = cv2.imread(str(thumbnails[0].thumbnail_path), cv2.IMREAD_COLOR)
    if sample_bgr is None:
        raise IOError(f"Could not read thumbnail: {thumbnails[0].thumbnail_path}")
    sample_height, sample_width = sample_bgr.shape[:2]
    natural_height = (
        panel_width
        * figure.get_figwidth()
        / figure.get_figheight()
        * sample_height
        / sample_width
    )
    panel_height = min(natural_height, 0.33)
    panel_bottom = 0.105 + (0.33 - panel_height) / 2.0

    for panel_index, record in enumerate(thumbnails):
        left = panel_left + panel_index * (panel_width + panel_gap)
        crop_axis = figure.add_axes([left, panel_bottom, panel_width, panel_height])
        thumbnail_bgr = cv2.imread(str(record.thumbnail_path), cv2.IMREAD_COLOR)
        if thumbnail_bgr is None:
            raise IOError(f"Could not read thumbnail: {record.thumbnail_path}")
        crop_axis.imshow(
            cv2.cvtColor(thumbnail_bgr, cv2.COLOR_BGR2RGB), interpolation="lanczos"
        )
        crop_axis.set_axis_off()

    arrow = FancyArrowPatch(
        (0.047, 0.060),
        (0.900, 0.060),
        transform=figure.transFigure,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.0,
        color="black",
        clip_on=False,
    )
    figure.add_artist(arrow)
    figure.text(
        0.907, 0.060, "Time", ha="left", va="center", fontsize=17, family="serif"
    )

    figure.savefig(output_png, dpi=dpi, facecolor="white")
    figure.savefig(output_pdf, facecolor="white")
    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot frame-wise variance of Laplacian over an aligned face or the full frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Directory containing ordered image frames.",
    )
    parser.add_argument(
        "--measurement-region",
        choices=("face", "full-frame"),
        default="face",
        help=(
            "Pixels used for the sharpness score. Face mode tracks and aligns one identity; "
            "full-frame mode measures every readable frame without loading InsightFace."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to <frames-dir>/../laplacian_face_analysis "
            "or ../laplacian_full_frame_analysis according to --measurement-region."
        ),
    )
    parser.add_argument(
        "--target-point",
        type=parse_point,
        default=None,
        metavar="X,Y",
        help=(
            "Face mode only: point inside the target face in the anchor frame. "
            "Without it, the largest anchor-frame face is used."
        ),
    )
    parser.add_argument(
        "--anchor-frame",
        type=int,
        default=0,
        help="Face mode only: zero-based frame used to initialize the target identity.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First zero-based sequence frame to analyze (inclusive).",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last zero-based sequence frame to analyze (inclusive). Defaults to the final frame.",
    )
    parser.add_argument(
        "--insightface-root",
        type=Path,
        default=DEFAULT_INSIGHTFACE_ROOT,
        help="Face mode only: directory containing InsightFace models.",
    )
    parser.add_argument("--model-name", default="antelopev2")
    parser.add_argument(
        "--provider",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Face mode only: ONNX Runtime execution provider.",
    )
    parser.add_argument("--detector-size", type=int, default=640)
    parser.add_argument("--detection-threshold", type=float, default=0.35)
    parser.add_argument(
        "--min-cosine-similarity",
        type=float,
        default=0.10,
        help="Face mode only: reject detections less similar than this to the anchor identity.",
    )
    parser.add_argument(
        "--min-face-area-ratio",
        type=float,
        default=0.20,
        help="Face mode only: reject candidates smaller than this fraction of the anchor face area.",
    )
    parser.add_argument(
        "--max-alignment-padding",
        type=float,
        default=0.02,
        help="Face mode only: skip crops with more than this fraction sampled outside the frame.",
    )
    parser.add_argument(
        "--max-abs-yaw",
        type=float,
        default=None,
        help="Face mode only: maximum absolute InsightFace yaw in degrees.",
    )
    parser.add_argument(
        "--max-abs-pitch",
        type=float,
        default=None,
        help="Face mode only: maximum absolute InsightFace pitch in degrees.",
    )
    parser.add_argument("--num-thumbnails", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--params-json",
        type=Path,
        default=None,
        help="Optional params_used.json containing original frame_indices for the CSV.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace, frame_count: int) -> None:
    end_frame = frame_count - 1 if args.end_frame is None else args.end_frame
    if args.measurement_region == "face" and (
        args.anchor_frame < 0 or args.anchor_frame >= frame_count
    ):
        raise ValueError(f"--anchor-frame must be in [0, {frame_count - 1}].")
    if args.start_frame < 0 or args.start_frame >= frame_count:
        raise ValueError(f"--start-frame must be in [0, {frame_count - 1}].")
    if end_frame < 0 or end_frame >= frame_count:
        raise ValueError(f"--end-frame must be in [0, {frame_count - 1}].")
    if end_frame < args.start_frame:
        raise ValueError("--end-frame must be greater than or equal to --start-frame.")
    if args.measurement_region == "face":
        if args.detector_size <= 0:
            raise ValueError("--detector-size must be positive.")
        if not 0.0 <= args.detection_threshold <= 1.0:
            raise ValueError("--detection-threshold must be in [0, 1].")
        if not -1.0 <= args.min_cosine_similarity <= 1.0:
            raise ValueError("--min-cosine-similarity must be in [-1, 1].")
        if not 0.0 <= args.min_face_area_ratio <= 1.0:
            raise ValueError("--min-face-area-ratio must be in [0, 1].")
        if not 0.0 <= args.max_alignment_padding <= 1.0:
            raise ValueError("--max-alignment-padding must be in [0, 1].")
        if args.max_abs_yaw is not None and not 0.0 <= args.max_abs_yaw <= 180.0:
            raise ValueError("--max-abs-yaw must be in [0, 180].")
        if args.max_abs_pitch is not None and not 0.0 <= args.max_abs_pitch <= 180.0:
            raise ValueError("--max-abs-pitch must be in [0, 180].")
    if args.num_thumbnails <= 0:
        raise ValueError("--num-thumbnails must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frame_paths = list_frames(args.frames_dir)
    validate_args(args, len(frame_paths))
    end_frame = len(frame_paths) - 1 if args.end_frame is None else args.end_frame
    face_mode = args.measurement_region == "face"

    default_output_name = (
        "laplacian_face_analysis" if face_mode else "laplacian_full_frame_analysis"
    )
    output_dir = args.output_dir or args.frames_dir.parent / default_output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = output_dir / "aligned_face_crops" if face_mode else None

    source_indices = discover_source_indices(
        args.frames_dir, args.params_json, len(frame_paths)
    )
    pose_filter_enabled = face_mode and (
        args.max_abs_yaw is not None or args.max_abs_pitch is not None
    )

    analyzer = None
    anchor_embedding = None
    previous_bbox = None
    anchor_area = None
    if face_mode:
        assert crop_dir is not None
        crop_dir.mkdir(parents=True, exist_ok=True)
        analyzer = load_face_analyzer(
            insightface_root=args.insightface_root,
            model_name=args.model_name,
            provider_mode=args.provider,
            detector_size=args.detector_size,
            detection_threshold=args.detection_threshold,
            require_pose=pose_filter_enabled,
        )

        anchor_image = cv2.imread(str(frame_paths[args.anchor_frame]), cv2.IMREAD_COLOR)
        if anchor_image is None:
            raise IOError(
                f"Could not read anchor image: {frame_paths[args.anchor_frame]}"
            )
        anchor_faces = analyzer.get(anchor_image)
        anchor_face = choose_anchor_face(anchor_faces, args.target_point)
        anchor_embedding = normalized_embedding(anchor_face)
        assert anchor_embedding is not None
        previous_bbox = face_bbox(anchor_face)
        anchor_area = bbox_area(previous_bbox)
    else:
        print(
            "Full-frame mode: InsightFace and all face-specific options are not used."
        )

    print(f"Frames: {len(frame_paths)} from {args.frames_dir}")
    print(
        f"Analysis duration: sequence frames {args.start_frame}-{end_frame} (inclusive)"
    )
    print(f"Measurement region: {args.measurement_region}")
    if face_mode:
        assert previous_bbox is not None
        print(
            f"Anchor: {frame_paths[args.anchor_frame].name}, bbox={previous_bbox.round(1).tolist()}"
        )
    print(f"Laplacian kernel:\n{LAPLACIAN_KERNEL}")

    records: list[MetricRecord] = []
    skipped: list[int] = []
    pose_rejected: list[int] = []

    for sequence_index in range(args.start_frame, end_frame + 1):
        path = frame_paths[sequence_index]
        crop_path = crop_dir / f"{path.stem}_face.png" if crop_dir is not None else None
        frame_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            print(f"Skipping unreadable frame: {path}")
            if crop_path is not None:
                crop_path.unlink(missing_ok=True)
            skipped.append(sequence_index)
            continue

        if not face_mode:
            frame_height, frame_width = frame_bgr.shape[:2]
            records.append(
                MetricRecord(
                    sequence_index=sequence_index,
                    frame_index=frame_number(path, sequence_index),
                    source_frame_index=source_indices[sequence_index],
                    filename=path.name,
                    variance=variance_of_laplacian(frame_bgr),
                    bbox=None,
                    detection_score=None,
                    cosine_similarity=None,
                    alignment_padding_fraction=None,
                    pitch_degrees=None,
                    yaw_degrees=None,
                    roll_degrees=None,
                    crop_path=None,
                    thumbnail_path=path,
                    measurement_region=args.measurement_region,
                    measurement_width=frame_width,
                    measurement_height=frame_height,
                )
            )
            continue

        assert analyzer is not None
        assert anchor_embedding is not None
        assert anchor_area is not None
        assert crop_path is not None
        faces = analyzer.get(frame_bgr)
        matched_face, similarity = select_target_face(
            faces,
            anchor_embedding=anchor_embedding,
            anchor_area=anchor_area,
            previous_bbox=previous_bbox,
            frame_shape=frame_bgr.shape[:2],
            min_similarity=args.min_cosine_similarity,
            min_area_ratio=args.min_face_area_ratio,
        )
        if matched_face is None or similarity is None:
            crop_path.unlink(missing_ok=True)
            skipped.append(sequence_index)
            continue

        # Keep tracking through profile frames even when their metrics will be
        # rejected by the frontal-pose gate.
        bbox = face_bbox(matched_face)
        previous_bbox = bbox

        try:
            crop, padding_fraction = align_face(frame_bgr, matched_face, crop_size=112)
        except (ValueError, cv2.error):
            crop_path.unlink(missing_ok=True)
            skipped.append(sequence_index)
            continue
        if padding_fraction > args.max_alignment_padding:
            crop_path.unlink(missing_ok=True)
            skipped.append(sequence_index)
            continue

        pose = face_pose_degrees(matched_face)
        if pose_filter_enabled and pose is None:
            crop_path.unlink(missing_ok=True)
            pose_rejected.append(sequence_index)
            continue
        pitch, yaw, roll = pose if pose is not None else (np.nan, np.nan, np.nan)
        if args.max_abs_yaw is not None and abs(yaw) > args.max_abs_yaw:
            crop_path.unlink(missing_ok=True)
            pose_rejected.append(sequence_index)
            continue
        if args.max_abs_pitch is not None and abs(pitch) > args.max_abs_pitch:
            crop_path.unlink(missing_ok=True)
            pose_rejected.append(sequence_index)
            continue

        if not cv2.imwrite(str(crop_path), crop):
            raise IOError(f"Could not write aligned crop: {crop_path}")

        records.append(
            MetricRecord(
                sequence_index=sequence_index,
                frame_index=frame_number(path, sequence_index),
                source_frame_index=source_indices[sequence_index],
                filename=path.name,
                variance=variance_of_laplacian(crop),
                bbox=bbox,
                detection_score=float(getattr(matched_face, "det_score", np.nan)),
                cosine_similarity=float(similarity),
                alignment_padding_fraction=padding_fraction,
                pitch_degrees=pitch,
                yaw_degrees=yaw,
                roll_degrees=roll,
                crop_path=crop_path,
                thumbnail_path=crop_path,
                measurement_region=args.measurement_region,
                measurement_width=crop.shape[1],
                measurement_height=crop.shape[0],
            )
        )

    if not records:
        if face_mode:
            raise RuntimeError(
                "The selected identity was not matched in any frames. Try lowering --min-cosine-similarity."
            )
        raise RuntimeError(
            "No readable frames were available in the selected duration."
        )

    artifact_stem = (
        "face_laplacian_variance" if face_mode else "full_frame_laplacian_variance"
    )
    csv_path = output_dir / f"{artifact_stem}.csv"
    png_path = output_dir / f"{artifact_stem}.png"
    pdf_path = output_dir / f"{artifact_stem}.pdf"
    write_csv(records, csv_path)
    plot_timeline(
        records,
        total_frames=len(frame_paths),
        frame_start=args.start_frame,
        frame_end=end_frame,
        output_png=png_path,
        output_pdf=pdf_path,
        thumbnail_count=args.num_thumbnails,
        dpi=args.dpi,
        measurement_region=args.measurement_region,
    )

    selected_frame_count = end_frame - args.start_frame + 1
    if face_mode:
        print(
            f"Valid target-face frames in selected duration: {len(records)}/{selected_frame_count}"
        )
    else:
        print(
            f"Measured readable frames in selected duration: {len(records)}/{selected_frame_count}"
        )
        measurement_shapes = {
            (record.measurement_width, record.measurement_height) for record in records
        }
        if len(measurement_shapes) > 1:
            print(
                "Warning: full-frame dimensions vary across the sequence; "
                "variance-of-Laplacian values are resolution-dependent."
            )
    print(f"Skipped sequence indices: {skipped if skipped else 'none'}")
    if pose_filter_enabled:
        print(
            "Pose filter: "
            f"|yaw| <= {args.max_abs_yaw if args.max_abs_yaw is not None else 'unbounded'} deg, "
            f"|pitch| <= {args.max_abs_pitch if args.max_abs_pitch is not None else 'unbounded'} deg"
        )
        print(
            f"Pose-rejected sequence indices: {pose_rejected if pose_rejected else 'none'}"
        )
    print(f"CSV:  {csv_path}")
    print(f"PNG:  {png_path}")
    print(f"PDF:  {pdf_path}")
    if crop_dir is not None:
        print(f"Crops: {crop_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
