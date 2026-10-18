"""Extract 32 MediaPipe face landmarks from a video or video folder.

The CSV schema is intentionally kept compatible with the training dataset:

    frame_idx, pt0_x, pt0_y, ..., pt31_x, pt31_y

Detected landmark coordinates are rounded pixel positions. Frames without a
face detection are written as NaN for every point.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from tqdm import tqdm


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}
DEFAULT_FPS = 30.0

# CUSTOM 32 Landmark Index Mapping
CUSTOM_MAP_32 = [
    # Right Eyebrow (3 points) - viewer's left
    70,
    105,
    107,
    # Left Eyebrow (3 points) - viewer's right
    336,
    334,
    300,
    # Nose Bridge (3 points)
    168,
    197,
    4,
    # Nose Bottom (4 points)
    98,
    97,
    326,
    327,
    # Right Eye (6 points)
    33,
    160,
    158,
    133,
    153,
    144,
    # Left Eye (6 points)
    362,
    385,
    387,
    263,
    373,
    380,
    # Mouth (7 points)
    61,   # Outer left corner
    291,  # Outer right corner
    0,    # Top outer midpoint
    17,   # Bottom outer midpoint
    312,
    82,
    14,   # Bottom inner midpoint
]


def normalize_extensions(video_extensions: Sequence[str]) -> set[str]:
    """Normalize extension strings to lower-case values with a leading dot."""
    return {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in video_extensions
    }


def iter_video_paths(input_path: str | Path, video_extensions: Sequence[str] = tuple(VIDEO_EXTENSIONS)) -> list[Path]:
    """Return one video path, or every supported video directly inside a folder."""
    input_path = Path(input_path)
    extensions = normalize_extensions(video_extensions)

    if input_path.is_file():
        if input_path.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported video extension: {input_path}")
        return [input_path]

    if input_path.is_dir():
        return sorted(path for path in input_path.iterdir() if path.suffix.lower() in extensions)

    raise ValueError(f"Input path does not exist: {input_path}")


def safe_fps(cap: cv2.VideoCapture) -> float:
    """Return capture FPS, falling back when metadata is missing or invalid."""
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0:
        return DEFAULT_FPS
    return fps


def open_video_writer(output_path: str | Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """Open a video writer with an H.264 attempt and an mp4v fallback."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {output_path}")
    return writer


def with_suffix(path: str | Path, suffix: str) -> Path:
    """Add suffix if the path does not already have it."""
    path = Path(path)
    if path.suffix.lower() == suffix:
        return path
    return path.with_suffix(suffix)


def make_landmarker_options(
    model_path: str | Path,
    min_face_detection_confidence: float = 0.5,
) -> FaceLandmarkerOptions:
    """Create MediaPipe FaceLandmarker options for video mode."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"MediaPipe model file not found: {model_path}")

    return FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=min_face_detection_confidence,
    )


def landmark_csv_header(num_points: int = len(CUSTOM_MAP_32)) -> list[str]:
    header = ["frame_idx"]
    for point_idx in range(num_points):
        header.extend([f"pt{point_idx}_x", f"pt{point_idx}_y"])
    return header


def format_csv_coord(value: float) -> float | int:
    """Keep detected pixel coordinates integer-like while preserving NaN rows."""
    if np.isnan(value):
        return np.nan
    if float(value).is_integer():
        return int(value)
    return float(value)


def save_landmark_outputs(coords_array: np.ndarray, output_npy_path: str | Path, output_csv_path: str | Path) -> None:
    """Save landmarks as .npy and as the training-compatible CSV schema."""
    output_npy_path = with_suffix(output_npy_path, ".npy")
    output_csv_path = with_suffix(output_csv_path, ".csv")
    output_npy_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_npy_path, coords_array)

    with output_csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(landmark_csv_header(coords_array.shape[1] if coords_array.ndim == 3 else len(CUSTOM_MAP_32)))
        for frame_idx, frame_data in enumerate(coords_array):
            row = [frame_idx]
            for x, y in frame_data:
                row.extend([format_csv_coord(float(x)), format_csv_coord(float(y))])
            writer.writerow(row)


def extract_frame_landmarks(result, width: int, height: int) -> np.ndarray:
    """Return one frame of 32 landmark coordinates in pixel space."""
    coords = np.full((len(CUSTOM_MAP_32), 2), np.nan, dtype=np.float32)
    if not result.face_landmarks:
        return coords

    face_landmarks = result.face_landmarks[0]
    for output_idx, mediapipe_idx in enumerate(CUSTOM_MAP_32):
        if mediapipe_idx >= len(face_landmarks):
            continue
        point = face_landmarks[mediapipe_idx]
        coords[output_idx] = [round(point.x * width), round(point.y * height)]
    return coords


def draw_landmarks(frame: np.ndarray, frame_coords: np.ndarray, radius: int = 2) -> None:
    """Draw finite landmark coordinates onto a BGR frame in-place."""
    for x, y in frame_coords:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(frame, (int(x), int(y)), radius, (0, 255, 0), -1)


def progress_write(progress, message: str) -> None:
    if progress is not None and hasattr(progress, "write"):
        progress.write(message)
    else:
        print(message)


def process_single_video(
    input_path: str | Path,
    output_video_path: str | Path | None,
    output_npy_path: str | Path,
    output_csv_path: str | Path | FaceLandmarker | None = None,
    landmarker: FaceLandmarker | None = None,
    progress: tqdm | None = None,
    write_video: bool = True,
) -> dict:
    """Process one video and save annotated video, .npy landmarks, and CSV landmarks.

    The old script passed ``landmarker`` as the fourth positional argument and
    wrote CSV next to the NPY output. That call shape is still accepted; new code
    should pass an explicit ``output_csv_path``.
    """
    if hasattr(output_csv_path, "detect_for_video"):
        old_progress = landmarker
        landmarker = output_csv_path
        output_csv_path = None
        if progress is None:
            progress = old_progress

    if landmarker is None:
        raise ValueError("process_single_video requires a FaceLandmarker instance")

    input_path = Path(input_path)
    output_npy_path = with_suffix(output_npy_path, ".npy")
    output_csv_path = with_suffix(output_csv_path or output_npy_path.with_suffix(".csv"), ".csv")
    output_video_path = Path(output_video_path) if output_video_path is not None else None

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        message = f"cannot open video: {input_path}"
        progress_write(progress, f"  [SKIP] {message}")
        return {"name": input_path.name, "status": f"error: {message}", "total_frames": 0, "missing_frames": 0}

    fps = safe_fps(cap)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    all_frame_coords: list[np.ndarray] = []
    total_frames = 0
    missing_frames = 0

    try:
        if write_video and output_video_path is not None:
            writer = open_video_writer(output_video_path, width, height, fps)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(total_frames * 1000.0 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_coords = extract_frame_landmarks(result, width, height)
            if not np.isfinite(frame_coords).any():
                missing_frames += 1

            all_frame_coords.append(frame_coords)
            if writer is not None:
                draw_landmarks(frame, frame_coords)
                writer.write(frame)

            total_frames += 1
    except Exception as exc:  # MediaPipe and OpenCV both raise runtime-specific exceptions.
        return {
            "name": input_path.name,
            "status": f"error: {exc}",
            "total_frames": total_frames,
            "missing_frames": missing_frames,
        }
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if all_frame_coords:
        coords_array = np.stack(all_frame_coords).astype(np.float32)
    else:
        coords_array = np.empty((0, len(CUSTOM_MAP_32), 2), dtype=np.float32)

    save_landmark_outputs(coords_array, output_npy_path, output_csv_path)

    summary = {
        "name": input_path.name,
        "status": "ok",
        "total_frames": total_frames,
        "missing_frames": missing_frames,
        "npy_path": str(output_npy_path),
        "csv_path": str(output_csv_path),
    }
    if writer is not None and output_video_path is not None:
        summary["video_path"] = str(output_video_path)

    progress_write(progress, f"  [OK] {input_path.name} ({total_frames} frames, {missing_frames} no-face frames)")
    return summary


def build_output_paths(
    video_path: Path,
    output_video_folder: str | Path | None,
    output_npy_folder: str | Path,
    output_csv_folder: str | Path,
    write_video: bool = True,
) -> tuple[Path | None, Path, Path]:
    output_video_path = None
    if write_video:
        if output_video_folder is None:
            raise ValueError("output_video_folder is required when write_video=True")
        output_video_path = Path(output_video_folder) / video_path.name

    output_npy_path = Path(output_npy_folder) / f"{video_path.stem}.npy"
    output_csv_path = Path(output_csv_folder) / f"{video_path.stem}.csv"
    return output_video_path, output_npy_path, output_csv_path


def outputs_exist(output_video_path: Path | None, output_npy_path: Path, output_csv_path: Path, write_video: bool) -> bool:
    required_paths = [output_npy_path, output_csv_path]
    if write_video and output_video_path is not None:
        required_paths.append(output_video_path)
    return all(path.exists() for path in required_paths)


def process_videos(
    input_path: str | Path,
    output_video_folder: str | Path | None,
    output_npy_folder: str | Path,
    output_csv_folder: str | Path,
    model_path: str | Path,
    video_extensions: Sequence[str] = tuple(VIDEO_EXTENSIONS),
    skip_existing: bool = False,
    write_video: bool = True,
    min_face_detection_confidence: float = 0.5,
) -> list[dict]:
    """Process one video or every supported video in a folder."""
    video_paths = iter_video_paths(input_path, video_extensions)
    if not video_paths:
        print(f"No video files found in: {input_path}")
        return []

    if write_video and output_video_folder is not None:
        Path(output_video_folder).mkdir(parents=True, exist_ok=True)
    Path(output_npy_folder).mkdir(parents=True, exist_ok=True)
    Path(output_csv_folder).mkdir(parents=True, exist_ok=True)

    options = make_landmarker_options(model_path, min_face_detection_confidence)
    summaries: list[dict] = []

    print(f"Found {len(video_paths)} video(s) to process from '{input_path}'\n")
    with tqdm(total=len(video_paths), desc="Videos", unit="vid", dynamic_ncols=True) as progress:
        for video_path in video_paths:
            progress.set_postfix_str(video_path.name, refresh=True)
            output_video_path, output_npy_path, output_csv_path = build_output_paths(
                video_path,
                output_video_folder,
                output_npy_folder,
                output_csv_folder,
                write_video,
            )

            if skip_existing and outputs_exist(output_video_path, output_npy_path, output_csv_path, write_video):
                summary = {
                    "name": video_path.name,
                    "status": "skipped: outputs exist",
                    "total_frames": 0,
                    "missing_frames": 0,
                }
                progress.write(f"  [SKIP] {video_path.name} (outputs exist)")
            else:
                # VIDEO mode requires monotonically increasing timestamps per landmarker session.
                with FaceLandmarker.create_from_options(options) as landmarker:
                    summary = process_single_video(
                        video_path,
                        output_video_path,
                        output_npy_path,
                        output_csv_path,
                        landmarker,
                        progress,
                        write_video,
                    )

            summaries.append(summary)
            progress.update(1)

    print_process_summary(summaries)
    return summaries


def process_folder(
    input_folder: str | Path,
    output_video_folder: str | Path,
    output_npy_folder: str | Path,
    output_csv_folder: str | Path,
    model_path: str | Path,
    video_extensions: Sequence[str] = tuple(VIDEO_EXTENSIONS),
    skip_existing: bool = False,
    write_video: bool = True,
    min_face_detection_confidence: float = 0.5,
) -> list[dict]:
    """Backward-compatible wrapper for the old folder-processing entry point.

    Despite the historical name, ``input_folder`` may now be either a folder or a
    single video file.
    """
    return process_videos(
        input_folder,
        output_video_folder,
        output_npy_folder,
        output_csv_folder,
        model_path,
        video_extensions=video_extensions,
        skip_existing=skip_existing,
        write_video=write_video,
        min_face_detection_confidence=min_face_detection_confidence,
    )


def default_output_folders(input_path: str | Path, output_root: str | Path | None = None) -> tuple[Path, Path, Path]:
    """Derive output folders from input path unless an output root is provided."""
    input_path = Path(input_path)
    if output_root is not None:
        output_root = Path(output_root)
        return output_root / "video", output_root / "npy", output_root / "csv"

    if input_path.is_dir():
        parent = input_path.parent
        prefix = input_path.name
    else:
        parent = input_path.parent
        prefix = input_path.stem

    return (
        parent / f"{prefix}_32landmark_video",
        parent / f"{prefix}_32landmark_npy",
        parent / f"{prefix}_32landmark_csv",
    )


def resolve_output_folders(args: argparse.Namespace) -> tuple[Path | None, Path, Path]:
    default_video, default_npy, default_csv = default_output_folders(args.input_path, args.output_root)
    output_video_folder = Path(args.output_video_folder) if args.output_video_folder else default_video
    output_npy_folder = Path(args.output_npy_folder) if args.output_npy_folder else default_npy
    output_csv_folder = Path(args.output_csv_folder) if args.output_csv_folder else default_csv

    if not args.write_video:
        output_video_folder = None
    return output_video_folder, output_npy_folder, output_csv_folder


def print_process_summary(summaries: Sequence[dict]) -> None:
    ok_count = sum(1 for item in summaries if item["status"] == "ok")
    skipped_count = sum(1 for item in summaries if item["status"].startswith("skipped"))
    failed_count = len(summaries) - ok_count - skipped_count
    total_frames = sum(int(item.get("total_frames", 0)) for item in summaries if item["status"] == "ok")
    missing_frames = sum(int(item.get("missing_frames", 0)) for item in summaries if item["status"] == "ok")

    print(
        f"\nDone. {ok_count} succeeded, {skipped_count} skipped, {failed_count} failed. "
        f"Frames: {total_frames}, no-face frames: {missing_frames}."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the custom 32 MediaPipe face landmarks from one video or a folder of videos.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", "--input-path", dest="input_path", help="Video file or folder to process.")
    input_group.add_argument("--input-folder", "-i", dest="input_path", help="Folder of videos to process.")
    input_group.add_argument("--input-video", dest="input_path", help="Single video file to process.")

    parser.add_argument("--model-path", default="./checkpoints/mediapipe/face_landmarker.task", help="Path to MediaPipe face_landmarker.task.")
    parser.add_argument("--output-root", help="Optional root folder. Defaults become root/video, root/npy, root/csv.")
    parser.add_argument("--output-video-folder", help="Folder for annotated videos.")
    parser.add_argument("--output-npy-folder", help="Folder for .npy landmark arrays.")
    parser.add_argument("--output-csv-folder", help="Folder for training-compatible CSV landmarks.")
    parser.add_argument(
        "--video-extensions",
        nargs="+",
        default=sorted(VIDEO_EXTENSIONS),
        help="Video extensions to include when the input is a folder.",
    )
    parser.add_argument(
        "--min-face-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe face detection confidence threshold.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos whose expected outputs already exist.")
    parser.add_argument("--no-video", dest="write_video", action="store_false", help="Only save .npy and .csv landmarks.")
    parser.set_defaults(write_video=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_video_folder, output_npy_folder, output_csv_folder = resolve_output_folders(args)

    summaries = process_videos(
        input_path=args.input_path,
        output_video_folder=output_video_folder,
        output_npy_folder=output_npy_folder,
        output_csv_folder=output_csv_folder,
        model_path=args.model_path,
        video_extensions=args.video_extensions,
        skip_existing=args.skip_existing,
        write_video=args.write_video,
        min_face_detection_confidence=args.min_face_detection_confidence,
    )
    return 1 if any(item["status"].startswith("error") for item in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
- Example usage:

python3 tools/preprocessing/mediapipe_landmarks.py \
  --input-folder /path/to/videos \
  --model-path ./checkpoints/mediapipe/face_landmarker.task

- Single video:

python3 tools/preprocessing/mediapipe_landmarks.py \
  --input-video /path/to/video.mp4 \
  --model-path ./checkpoints/mediapipe/face_landmarker.task \
  --output-root /path/to/output

  
Default outputs for /data/my_videos become:
- /data/my_videos_32landmark_video
- /data/my_videos_32landmark_npy
- /data/my_videos_32landmark_csv

"""