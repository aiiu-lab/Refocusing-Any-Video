"""Face keypoint control-video generation.

This is the canonical implementation for both previous scripts:

  * process_5_face_kp_video_batch_new.py
  * process_5_face_kp_video_batch_new_multi.py

Modes
-----
single:
    Detect the largest face in each frame and write one control video per input
    video, preserving the input filename. Frames without a detected face are
    written as black frames.

multi:
    Detect all faces in each frame, assign simple IoU-based track IDs, and write
    one control video per face track. Missing frames for a track are black.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import onnxruntime as ort

# PyTorch's CUDA wheels keep CUDA/cuDNN shared libraries under the Python
# environment's ``nvidia/*/lib`` directories, which are not necessarily on
# LD_LIBRARY_PATH. ONNX Runtime 1.21+ can preload them from those site-package
# directories before InsightFace creates its CUDA inference sessions.
if "CUDAExecutionProvider" in ort.get_available_providers() and hasattr(ort, "preload_dlls"):
    ort.preload_dlls(directory="")

from insightface.app import FaceAnalysis
from PIL import Image
from tqdm import tqdm


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSIGHTFACE_ROOT = REPO_ROOT / "checkpoints" / "insightface"
KEYPOINT_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
]
LIMB_SEQUENCE = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])


def iter_video_paths(input_path: str | Path) -> list[Path]:
    """Return one video path or every supported video in a directory."""
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {input_path}")
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    raise ValueError(f"Input path does not exist: {input_path}")


def load_face_encoders(insightface_root: str, providers: Sequence[str] | None = None) -> list[FaceAnalysis]:
    """Load multi-scale InsightFace detectors.

    Detectors are prepared at 640, 320, and 160 pixels. Inference tries them in
    that order and uses the first detector that returns any face.
    """
    providers = list(providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
    apps = []
    for det_size in (640, 320, 160):
        app = FaceAnalysis(name="antelopev2", root=insightface_root, providers=providers)
        app.prepare(ctx_id=0, det_size=(det_size, det_size))
        apps.append(app)
    return apps


def detect_face_multiscale(bgr_image: np.ndarray, apps: Sequence[FaceAnalysis]) -> list:
    """Return faces from the first detector scale that finds at least one face."""
    for app in apps:
        faces = app.get(bgr_image)
        if faces:
            return faces
    return []


# Backward-compatible plural name used by the old multi-face script.
def detect_faces_multiscale(bgr_image: np.ndarray, apps: Sequence[FaceAnalysis]) -> list:
    return detect_face_multiscale(bgr_image, apps)


def face_bbox(face) -> np.ndarray:
    return np.asarray(face["bbox"], dtype=np.float32)


def face_area(face) -> float:
    bbox = face_bbox(face)
    return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))


def select_largest_face(faces: Sequence) -> object | None:
    if not faces:
        return None
    return max(faces, key=face_area)


def draw_kps(
    image_pil: Image.Image,
    kps: np.ndarray,
    color_list: Sequence[tuple[int, int, int]] = KEYPOINT_COLORS,
) -> Image.Image:
    """Draw 5 facial keypoints and connecting limbs on a black canvas."""
    stick_width = 4
    keypoints = np.asarray(kps)
    width, height = image_pil.size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for index in LIMB_SEQUENCE:
        color = color_list[index[0]]
        x = keypoints[index][:, 0]
        y = keypoints[index][:, 1]
        length = float(((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5)
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))),
            (int(length / 2), stick_width),
            int(angle),
            0,
            360,
            1,
        )
        canvas = cv2.fillConvexPoly(canvas.copy(), polygon, color)

    canvas = (canvas * 0.6).astype(np.uint8)
    for idx, kp in enumerate(keypoints):
        canvas = cv2.circle(canvas.copy(), (int(kp[0]), int(kp[1])), 10, color_list[idx], -1)

    return Image.fromarray(canvas)


def render_face_keypoints(face, width: int, height: int) -> np.ndarray:
    """Render one detected face's keypoints as a BGR image."""
    blank = Image.new("RGB", (width, height), (0, 0, 0))
    kps_pil = draw_kps(blank, face["kps"])
    return cv2.cvtColor(np.asarray(kps_pil), cv2.COLOR_RGB2BGR)


def generate_control_image(
    source_image: Image.Image,
    face_apps: Sequence[FaceAnalysis],
    target_width: int,
    target_height: int,
) -> Image.Image | None:
    """Generate a single-face keypoint image from the largest detected face.

    This function keeps the old single-face helper API used by dataset creation
    code. It returns None if no face is detected.
    """
    resized = source_image.convert("RGB").resize((target_width, target_height), Image.LANCZOS)
    bgr = cv2.cvtColor(np.asarray(resized), cv2.COLOR_RGB2BGR)
    face = select_largest_face(detect_face_multiscale(bgr, face_apps))
    if face is None:
        return None
    return draw_kps(resized, face["kps"])


def open_video_writer(output_path: str | Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """Open an MP4 VideoWriter with H.264 codec and an mp4v fallback."""
    output_path = str(output_path)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    if not writer.isOpened():
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {output_path}")
    return writer


def output_size_from_capture(cap: cv2.VideoCapture, target_width: int | None, target_height: int | None) -> tuple[int, int]:
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return target_width or src_w, target_height or src_h


def process_single_video(
    video_path: Path,
    face_apps: Sequence[FaceAnalysis],
    output_path: Path,
    target_width: int | None = None,
    target_height: int | None = None,
) -> dict:
    """Write one largest-face keypoint control video for one input video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"name": video_path.name, "status": "error: cannot open", "total_frames": 0, "skipped_frames": 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = output_size_from_capture(cap, target_width, target_height)
    writer = open_video_writer(output_path, width, height, fps)
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    total_frames = 0
    skipped_frames = 0

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            break
        total_frames += 1

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        control = generate_control_image(Image.fromarray(rgb_frame), face_apps, width, height)
        if control is None:
            skipped_frames += 1
            writer.write(black_frame)
        else:
            writer.write(cv2.cvtColor(np.asarray(control), cv2.COLOR_RGB2BGR))

    cap.release()
    writer.release()

    return {
        "name": video_path.name,
        "status": "ok",
        "total_frames": total_frames,
        "skipped_frames": skipped_frames,
    }


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def track_faces(all_frame_detections: list[list[dict]], iou_threshold: float = 0.3) -> list[list[int]]:
    """Assign simple greedy IoU track IDs to frame detections."""
    active_tracks: dict[int, np.ndarray] = {}
    next_id = 0
    assignments: list[list[int]] = []

    for frame_detections in all_frame_detections:
        if not frame_detections:
            assignments.append([])
            continue

        det_bboxes = [det["bbox"] for det in frame_detections]
        track_ids = list(active_tracks.keys())

        if not track_ids:
            frame_ids = []
            for bbox in det_bboxes:
                active_tracks[next_id] = bbox
                frame_ids.append(next_id)
                next_id += 1
            assignments.append(frame_ids)
            for det, track_id in zip(frame_detections, frame_ids):
                det["track_id"] = track_id
            continue

        iou_matrix = np.zeros((len(det_bboxes), len(track_ids)), dtype=np.float32)
        for det_idx, det_bbox in enumerate(det_bboxes):
            for track_idx, track_id in enumerate(track_ids):
                iou_matrix[det_idx, track_idx] = _iou(det_bbox, active_tracks[track_id])

        assigned_dets = set()
        assigned_tracks = set()
        frame_ids: list[int | None] = [None] * len(det_bboxes)

        while iou_matrix.size:
            det_idx, track_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[det_idx, track_idx] < iou_threshold:
                break
            if det_idx in assigned_dets or track_idx in assigned_tracks:
                iou_matrix[det_idx, track_idx] = -1.0
                continue

            track_id = track_ids[track_idx]
            frame_ids[det_idx] = track_id
            active_tracks[track_id] = det_bboxes[det_idx]
            assigned_dets.add(det_idx)
            assigned_tracks.add(track_idx)
            iou_matrix[det_idx, :] = -1.0
            iou_matrix[:, track_idx] = -1.0

        for det_idx, det_bbox in enumerate(det_bboxes):
            if frame_ids[det_idx] is None:
                active_tracks[next_id] = det_bbox
                frame_ids[det_idx] = next_id
                next_id += 1

        final_frame_ids = [int(track_id) for track_id in frame_ids]
        assignments.append(final_frame_ids)
        for det, track_id in zip(frame_detections, final_frame_ids):
            det["track_id"] = track_id

    return assignments


def process_multi_video(
    video_path: Path,
    face_apps: Sequence[FaceAnalysis],
    output_dir: Path,
    target_width: int | None = None,
    target_height: int | None = None,
    iou_threshold: float = 0.3,
) -> dict:
    """Write one keypoint control video per detected face track."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"name": video_path.name, "status": "error: cannot open", "total_frames": 0, "n_tracks": 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = output_size_from_capture(cap, target_width, target_height)
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    all_frame_detections: list[list[dict]] = []

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            break

        resized_bgr = cv2.resize(bgr_frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        frame_detections = []
        for face in detect_face_multiscale(resized_bgr, face_apps):
            frame_detections.append({
                "bbox": face_bbox(face),
                "kps_img": render_face_keypoints(face, width, height),
            })
        all_frame_detections.append(frame_detections)

    cap.release()
    total_frames = len(all_frame_detections)
    if total_frames == 0:
        return {"name": video_path.name, "status": "error: no frames read", "total_frames": 0, "n_tracks": 0}

    track_assignments = track_faces(all_frame_detections, iou_threshold)
    track_ids = sorted({track_id for frame_ids in track_assignments for track_id in frame_ids})
    writers: dict[int, cv2.VideoWriter] = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = video_path.suffix or ".mp4"
    for output_index, track_id in enumerate(track_ids):
        out_path = output_dir / f"{video_path.stem}_face{output_index:02d}{suffix}"
        writers[track_id] = open_video_writer(out_path, width, height, fps)

    for frame_detections, frame_ids in zip(all_frame_detections, track_assignments):
        track_to_image = {track_id: det["kps_img"] for det, track_id in zip(frame_detections, frame_ids)}
        for track_id, writer in writers.items():
            writer.write(track_to_image.get(track_id, black_frame))

    for writer in writers.values():
        writer.release()

    return {
        "name": video_path.name,
        "status": "ok",
        "total_frames": total_frames,
        "n_tracks": len(track_ids),
    }


def batch_process(
    input_folder: str,
    output_folder: str,
    face_apps: Sequence[FaceAnalysis],
    target_width: int | None = None,
    target_height: int | None = None,
    mode: str = "single",
    iou_threshold: float = 0.3,
) -> list[dict]:
    """Process a single video file or all supported videos in a folder.

    The parameter name `input_folder` is kept for backward compatibility with
    the old scripts, but it can now be either a directory or one video file.
    """
    input_path = Path(input_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = iter_video_paths(input_path)
    if not video_paths:
        print(f"No supported video files found in: {input_path}")
        sys.exit(1)

    source_kind = "file" if input_path.is_file() else "folder"
    tqdm.write(f"Found {len(video_paths)} video(s) from input {source_kind}: '{input_path}'")
    tqdm.write(f"Output folder: '{output_dir}'")
    tqdm.write(f"Mode: {mode}\n")

    summaries: list[dict] = []
    no_face_names: list[str] = []

    with tqdm(video_paths, desc="Videos", unit="vid", dynamic_ncols=True) as progress:
        for video_path in progress:
            progress.set_description(f"Videos [{video_path.name}]")
            if mode == "single":
                summary = process_single_video(
                    video_path=video_path,
                    face_apps=face_apps,
                    output_path=output_dir / video_path.name,
                    target_width=target_width,
                    target_height=target_height,
                )
                if summary["status"] == "ok" and summary["skipped_frames"] > 0:
                    no_face_names.append(video_path.name)
            elif mode == "multi":
                summary = process_multi_video(
                    video_path=video_path,
                    face_apps=face_apps,
                    output_dir=output_dir,
                    target_width=target_width,
                    target_height=target_height,
                    iou_threshold=iou_threshold,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            summaries.append(summary)
            write_progress_line(summary, mode)

    if mode == "single":
        write_no_face_log(output_dir, no_face_names)
    print_summary_table(summaries, mode)
    return summaries


def write_progress_line(summary: dict, mode: str) -> None:
    if summary["status"] != "ok":
        tqdm.write(f"  ERR {summary['name']} - {summary['status']}")
        return
    if mode == "single":
        extra = f", {summary['skipped_frames']} black frame(s)" if summary["skipped_frames"] else ""
        tqdm.write(f"  OK  {summary['name']} - {summary['total_frames']} frames{extra}")
    else:
        tqdm.write(f"  OK  {summary['name']} - {summary['total_frames']} frames, {summary['n_tracks']} face track(s)")


def write_no_face_log(output_dir: Path, no_face_names: Iterable[str]) -> None:
    names = list(no_face_names)
    if not names:
        tqdm.write("\nAll processed frames had a detected face; no no-face log written.")
        return

    no_face_path = output_dir / "no_face_videos.txt"
    with open(no_face_path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")
    tqdm.write(f"\nVideos with no-face frames logged to: {no_face_path}")


def print_summary_table(summaries: Sequence[dict], mode: str) -> None:
    print()
    if mode == "single":
        print("=" * 68)
        print(f"{'File':<36} {'Status':<8} {'Frames':>8} {'Black':>8}")
        print("-" * 68)
        for summary in summaries:
            if summary["status"] == "ok":
                print(f"{summary['name']:<36} {'ok':<8} {summary['total_frames']:>8} {summary['skipped_frames']:>8}")
            else:
                print(f"{summary['name']:<36} {summary['status']}")
        print("=" * 68)
    else:
        print("=" * 68)
        print(f"{'File':<36} {'Status':<8} {'Frames':>8} {'Tracks':>8}")
        print("-" * 68)
        for summary in summaries:
            if summary["status"] == "ok":
                print(f"{summary['name']:<36} {'ok':<8} {summary['total_frames']:>8} {summary['n_tracks']:>8}")
            else:
                print(f"{summary['name']:<36} {summary['status']}")
        print("=" * 68)

    ok = sum(1 for summary in summaries if summary["status"] == "ok")
    err = len(summaries) - ok
    print(f"\n{ok} succeeded, {err} failed.")


def parse_args(default_mode: str = "single") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face-keypoint control-video generator for one video or a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", "--input-path", dest="input_path", metavar="PATH", help="Input video file or folder.")
    input_group.add_argument("--input-folder", "-i", dest="input_path", metavar="DIR", help="Input folder of videos (backward-compatible alias).")
    input_group.add_argument("--input-video", dest="input_path", metavar="VIDEO", help="Single input video file.")
    parser.add_argument("--output-folder", "-o", required=True, metavar="DIR")
    parser.add_argument("--mode", choices=("single", "multi"), default=default_mode)
    parser.add_argument("--width", type=int, default=None, help="Output frame width. Defaults to source width.")
    parser.add_argument("--height", type=int, default=None, help="Output frame height. Defaults to source height.")
    parser.add_argument("--insightface-root", default=str(DEFAULT_INSIGHTFACE_ROOT), help="InsightFace model root.")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold for multi-face tracking.")
    return parser.parse_args()


def main(default_mode: str = "single") -> None:
    args = parse_args(default_mode=default_mode)
    print("Loading face encoders ...\n")
    face_apps = load_face_encoders(args.insightface_root)
    batch_process(
        input_folder=args.input_path,
        output_folder=args.output_folder,
        face_apps=face_apps,
        target_width=args.width,
        target_height=args.height,
        mode=args.mode,
        iou_threshold=args.iou_threshold,
    )


if __name__ == "__main__":
    main()


"""
python tools/preprocessing/face_keypoints.py \
  --input-video ./long_video_example/american_psycho.mp4 \
  --output-folder ./long_video_example/kp_video \
  --mode single \
  --width 512 --height 512

python tools/preprocessing/face_keypoints.py \
  --input-video ./long_video_example/breaking_bad3_K13_tc.mp4 \
  --output-folder ./long_video_example/kp_video \
  --mode multi \
  --width 512 --height 512

python tools/preprocessing/face_keypoints.py \
  --input-folder ./videos_in \
  --output-folder ./videos_out \
  --mode multi
"""
