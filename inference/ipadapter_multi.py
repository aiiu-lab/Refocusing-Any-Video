#!/usr/bin/env python
"""
Multi-Subject Video Refocusing Inference — IP-Adapter + KP + FaRL  (Two-Stage)

Supports 2 or more subjects simultaneously, with the SAME multi-subject
decoupled-cross-attention recipe as the Track4Gen-Refiner multi variant —
just WITHOUT the Track4Gen Refiner module. Use this when your IP-Adapter
checkpoint was trained without the Refiner.

For each subject s, the decoupled cross-attention becomes:
    final_attn = attn(q,k,v)
               + scale * attn(q, k_1', v_1')   ← subject 1
               + scale * attn(q, k_2', v_2')   ← subject 2
               + ...
where k_i'/v_i' are projected from subject i's ip_tokens using the SAME shared
to_k_ip / to_v_ip weights learned during single-subject training.

New CLI flags vs. the single-subject script
-------------------------------------------
  --reference_images  face1.jpg face2.jpg ...  (replaces --reference_image)
  --keypoint_videos   kp1.mp4   kp2.mp4   ...  (replaces --keypoint_video)
  --face_mask_videos  mask1.mp4 mask2.mp4 ...  (optional; one per subject)

The number of reference images and keypoint videos must be equal.
--reference_image / --keypoint_video are still accepted for backward compatibility
(they are treated as subject 0 when the list flags are absent).

Usage:
    python inference/ipadapter_multi.py \\
        --input ./blurred_video.mp4 \\
        --reference_images ./face_A.jpg ./face_B.jpg \\
        --keypoint_videos ./kp_A.mp4 ./kp_B.mp4 \\
        --checkpoint ./outputs/checkpoint-90000 \\
        --output_dir ./inference_output_multi \\
        --chunk_frames --fp16
"""

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from peft import PeftModel
from simple_pipeline import IPAdapterKPStableVideoDiffusionPipeline as StableVideoDiffusionPipeline

# NOTE: imports the NO-Track4Gen wrapper. Use the `_track4genRefiner` variant
# of this script if your checkpoint includes the Refiner module.
NO_REFINER_IMPORT_ERROR = None
try:
    from rav.models.ipadapter_kp import (
        SVDIPAdapterWrapper,
        IPMaskedSpatioTemporalAttnProcessor,
    )
except Exception as exc:  # Keep --help usable when optional model deps are absent.
    NO_REFINER_IMPORT_ERROR = exc

    class _MissingModelDependency:
        pass

    SVDIPAdapterWrapper = _MissingModelDependency
    IPMaskedSpatioTemporalAttnProcessor = _MissingModelDependency


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# The overlapping Stage-2 scheduler is written for SVD's 14-frame window.
# New chunks contribute 8 fresh frames and reuse 6 context frames from the
# previous chunk. Keep these together so the indexing below is auditable.
SVD_CHUNK_FRAMES = 14
SVD_CHUNK_NEW_FRAMES = 8
SVD_CONTEXT_HEAD_INDEX = 0
SVD_CONTEXT_MID_INDEX = 5
SVD_CONTEXT_TAIL_SLICE = slice(10, 14)
SVD_CONTEXT_OUTPUT_SLICE = slice(2, 6)
SVD_NEW_FRAME_SLICE = slice(6, 14)
SVD_BLEND_RULES = (
    (-4, 2, 0.75, 0.25),
    (-3, 3, 0.50, 0.50),
    (-2, 4, 0.50, 0.50),
    (-1, 5, 0.25, 0.75),
)


# ---------------------------------------------------------------------------
# Multi-subject attention processor
# ---------------------------------------------------------------------------

class MultiSubjectIPAttnProcessor(IPMaskedSpatioTemporalAttnProcessor):
    """
    Extends IPMaskedSpatioTemporalAttnProcessor to sum IP-Adapter cross-attention
    contributions from multiple subjects.

    ip_adapter_image_embeds may be:
      None            — IP-Adapter disabled (unchanged behaviour)
      Tensor          — single subject (unchanged behaviour, backward-compatible)
      list[Tensor]    — N subjects; one decoupled cross-attn term added per subject

    Shared projection weights (to_k_ip / to_v_ip) are reused for every subject.
    """

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):

        input_hidden_states = hidden_states

        # ── 1. Standard base path (identical to parent) ──────────────────────
        hidden_states = self.base_processor(
            attn, hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )

        # ── 2. IP-Adapter path — one term per subject ────────────────────────
        if self.ip_adapter_image_embeds is None:
            return hidden_states

        embeds_list = (
            self.ip_adapter_image_embeds
            if isinstance(self.ip_adapter_image_embeds, list)
            else [self.ip_adapter_image_embeds]
        )

        # Compute query once (shared across all subjects)
        query = attn.to_q(input_hidden_states)
        query = attn.head_to_batch_dim(query)

        for ip_embeds in embeds_list:
            ip_key   = self.to_k_ip(ip_embeds)
            ip_value = self.to_v_ip(ip_embeds)
            ip_key   = attn.head_to_batch_dim(ip_key)
            ip_value = attn.head_to_batch_dim(ip_value)

            ip_attn_probs = attn.get_attention_scores(query, ip_key, None)
            ip_hidden     = torch.bmm(ip_attn_probs, ip_value)
            ip_hidden     = attn.batch_to_head_dim(ip_hidden)
            ip_hidden     = attn.to_out[0](ip_hidden)
            ip_hidden     = attn.to_out[1](ip_hidden)

            hidden_states = hidden_states + self.scale * ip_hidden

        return hidden_states


# ---------------------------------------------------------------------------
# Multi-subject UNet wrapper
# ---------------------------------------------------------------------------

class SVDIPAdapterWrapperMulti(SVDIPAdapterWrapper):
    """
    Extends SVDIPAdapterWrapper for multi-subject IP-Adapter inference.

    forward() accepts:
      reference_image : list of [B, C, H, W] tensors (one per subject)
                        — or a single tensor for backward compatibility.
      cross_attention_kwargs["keypoint_frames"] : list of [B, T, 3, H, W] tensors
                        (one per subject) — or a single tensor / None.
      cross_attention_kwargs["blur_frames"]     : [B, T, 3, H, W]  (shared)

    Per-subject ip_tokens are injected as a list into every MultiSubjectIPAttnProcessor.
    """

    def set_ip_adapter_processors(self):
        """Replace all spatial cross-attn processors with MultiSubjectIPAttnProcessor."""
        attn_procs = {}
        for name, _ in self.unet.attn_processors.items():
            if "attn2" in name and "temporal" not in name:
                cross_attention_dim = 1024
                hidden_size = self.unet.config.block_out_channels[0]

                if   "down_blocks.0" in name: hidden_size = 320
                elif "down_blocks.1" in name: hidden_size = 640
                elif "down_blocks.2" in name: hidden_size = 1280
                elif "mid_block"     in name: hidden_size = 1280
                elif "up_blocks.1"   in name: hidden_size = 1280
                elif "up_blocks.2"   in name: hidden_size = 640
                elif "up_blocks.3"   in name: hidden_size = 320

                attn_procs[name] = MultiSubjectIPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                )
            else:
                attn_procs[name] = self.unet.attn_processors[name]

        self.unet.set_attn_processor(attn_procs)

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        added_time_ids,
        reference_image=None,
        cross_attention_kwargs=None,
        return_dict=True,
    ):
        if reference_image is not None and cross_attention_kwargs is not None:
            # Normalise to list
            reference_images = (
                reference_image if isinstance(reference_image, list)
                else [reference_image]
            )

            keypoint_frames_raw = cross_attention_kwargs.get("keypoint_frames", None)
            blur_frames         = cross_attention_kwargs.get("blur_frames", None)

            # Normalise keypoint list — one entry per subject
            if keypoint_frames_raw is None or not isinstance(keypoint_frames_raw, list):
                keypoint_frames_list = [keypoint_frames_raw] * len(reference_images)
            else:
                keypoint_frames_list = keypoint_frames_raw

            num_frames = sample.shape[1]
            all_ip_tokens = []

            for ref_img, kp_frames in zip(reference_images, keypoint_frames_list):
                # ── Feature extraction (frozen) ──────────────────────────────
                with torch.no_grad():
                    face_id_embeds   = self.arcface(ref_img.float()).to(sample.dtype)   # [B, 512, 7, 7]
                    face_repr_embeds = self.farl(ref_img.float()).to(sample.dtype)      # [B, 197, 768]

                # ── Project ──────────────────────────────────────────────────
                id_tokens   = self.image_proj1(face_id_embeds)   # [B, 49,  1024]
                repr_tokens = self.image_proj2(face_repr_embeds) # [B, 197, 1024]

                # Repeat for every frame in the video
                id_tokens   = id_tokens.repeat_interleave(num_frames, dim=0)   # [B*T, 49,  1024]
                repr_tokens = repr_tokens.repeat_interleave(num_frames, dim=0) # [B*T, 197, 1024]

                # ── Keypoint encoder ─────────────────────────────────────────
                if kp_frames is not None and blur_frames is not None:
                    kp_flat   = kp_frames.flatten(0, 1).to(sample.dtype)   # [B*T, 3, H, W]
                    blur_flat = blur_frames.flatten(0, 1).to(sample.dtype) # [B*T, 3, H, W]
                    kp_tokens = self.keypoint_encoder(
                        torch.cat([kp_flat, blur_flat], dim=1)
                    )  # [B*T, 256, 1024]
                    ip_tokens = torch.cat([id_tokens, repr_tokens, kp_tokens], dim=1)  # [B*T, 502, 1024]
                else:
                    # No keypoint frames for this subject — identity+appearance tokens only
                    ip_tokens = torch.cat([id_tokens, repr_tokens], dim=1)  # [B*T, 246, 1024]

                ip_tokens = self.id_attn(ip_tokens)
                all_ip_tokens.append(ip_tokens)

            # ── Inject list into all processors ──────────────────────────────
            for proc in self.unet.attn_processors.values():
                if isinstance(proc, MultiSubjectIPAttnProcessor):
                    proc.ip_adapter_image_embeds = all_ip_tokens

        # ── UNet forward ─────────────────────────────────────────────────────
        out = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
            return_dict=return_dict,
        )

        # ── Cleanup ───────────────────────────────────────────────────────────
        for proc in self.unet.attn_processors.values():
            if isinstance(proc, MultiSubjectIPAttnProcessor):
                proc.ip_adapter_image_embeds = None

        return out


# ---------------------------------------------------------------------------
# Saving / visualization helpers  (unchanged from single-subject script)
# ---------------------------------------------------------------------------

def save_face_masks_visualization(face_masks: torch.Tensor, output_dir: str, fps: int = 10):
    os.makedirs(output_dir, exist_ok=True)
    T, _, H, W = face_masks.shape
    masks_np = (face_masks[:, 0].float().cpu().numpy() * 255).astype(np.uint8)

    video_path = os.path.join(output_dir, "face_masks.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
    for i in range(T):
        writer.write(cv2.cvtColor(masks_np[i], cv2.COLOR_GRAY2BGR))
    writer.release()
    print(f"Saved mask video : {video_path}")

    cols = int(np.ceil(np.sqrt(T)))
    rows = int(np.ceil(T / cols))
    grid = np.zeros((rows * H, cols * W), dtype=np.uint8)
    for i in range(T):
        r, c = divmod(i, cols)
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = masks_np[i]
    grid_path = os.path.join(output_dir, "face_masks_grid.png")
    Image.fromarray(grid, mode="L").save(grid_path)
    print(f"Saved mask grid  : {grid_path}  ({rows}x{cols} tiles of {H}x{W})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_face_point(value: str) -> tuple[int, int]:
    try:
        fx, fy = value.split(",")
        return int(fx.strip()), int(fy.strip())
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "expected two comma-separated integers, e.g. 320,240"
        ) from exc


def normalize_subject_args(args: argparse.Namespace) -> None:
    """Fold backward-compatible single-subject flags into list flags."""
    if args.reference_images is None and args.reference_image is not None:
        args.reference_images = [args.reference_image]
    if args.keypoint_videos is None and args.keypoint_video is not None:
        args.keypoint_videos = [args.keypoint_video]
    if args.face_mask_videos is None and args.face_mask_video is not None:
        args.face_mask_videos = [args.face_mask_video]

    # A provided mask video means the user is asking for mask visualization.
    if args.face_mask_videos:
        args.save_face_mask_debug = True


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    if args.num_frames <= 0:
        parser.error("--num_frames must be positive.")
    if args.output_fps <= 0:
        parser.error("--output_fps must be positive.")
    if args.decode_chunk_size <= 0:
        parser.error("--decode_chunk_size must be positive.")
    if args.keypoint_blur_sigma < 0:
        parser.error("--keypoint_blur_sigma must be non-negative.")
    if args.height is not None and args.height <= 0:
        parser.error("--height must be positive when provided.")
    if args.width is not None and args.width <= 0:
        parser.error("--width must be positive when provided.")

    if args.chunk_frames and args.num_frames != SVD_CHUNK_FRAMES:
        parser.error(
            f"--chunk_frames currently supports --num_frames {SVD_CHUNK_FRAMES} only. "
            "The Stage-2 overlap schedule is tied to the 14-frame SVD window."
        )

    if not args.disable_ipadapter:
        if not args.reference_images:
            parser.error(
                "Provide --reference_images (or --reference_image), "
                "unless --disable_identity_adapter/--disable_ipadapter is set."
            )
        if args.keypoint_videos and len(args.keypoint_videos) != len(args.reference_images):
            parser.error(
                f"--keypoint_videos has {len(args.keypoint_videos)} entries, "
                f"but --reference_images has {len(args.reference_images)}."
            )
        if args.face_mask_videos and len(args.face_mask_videos) != len(args.reference_images):
            parser.error(
                f"--face_mask_videos has {len(args.face_mask_videos)} entries, "
                f"but --reference_images has {len(args.reference_images)}."
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Multi-subject video refocusing inference: IP-Adapter + KP + FaRL, no Track4Gen Refiner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("Input / output")
    io_group.add_argument("--input", type=str, required=True, help="Input video file or image-frame folder.")
    io_group.add_argument("--output_dir", type=str, default="./inference_output_multi")
    io_group.add_argument("--output_fps", type=int, default=24)
    io_group.add_argument("--dry_run", action="store_true", help="Validate inputs and print the resolved setup without loading models.")

    subject_group = parser.add_argument_group("Subject conditioning")
    subject_group.add_argument(
        "--reference_images", type=str, nargs="+", default=None, metavar="IMG",
        help="Reference face images, one per subject. Takes precedence over --reference_image.",
    )
    subject_group.add_argument(
        "--reference_image", type=str, default=None,
        help="Single reference image for backward-compatible single-subject runs.",
    )
    subject_group.add_argument(
        "--keypoint_videos", type=str, nargs="+", default=None, metavar="KP",
        help="Optional keypoint/control videos, one per subject. Length must match --reference_images when provided.",
    )
    subject_group.add_argument(
        "--keypoint_video", type=str, default=None,
        help="Single keypoint/control video for backward-compatible single-subject runs.",
    )
    subject_group.add_argument("--keypoint_blur_sigma", type=float, default=2.0)
    subject_group.add_argument(
        "--disable_ipadapter",
        "--disable_identity_adapter",
        dest="disable_ipadapter",
        action="store_true",
        help="Run VDD without Identity Adapter reference images or keypoint conditioning.",
    )

    debug_group = parser.add_argument_group("Optional face-mask debug visualization")
    debug_group.add_argument(
        "--save_face_mask_debug", action="store_true",
        help=(
            "Save YOLO/SAM face-mask visualizations. These masks are not fed to "
            "this no-Refiner model, so this is off by default."
        ),
    )
    debug_group.add_argument(
        "--face_mask_videos", type=str, nargs="+", default=None, metavar="MASK",
        help="Precomputed mask videos to visualize, one per subject. Enables --save_face_mask_debug.",
    )
    debug_group.add_argument("--face_mask_video", type=str, default=None, help="Single mask video for backward compatibility.")
    debug_group.add_argument("--skip_face_mask_gen", action="store_true", help="When saving mask debug, use full-frame masks instead of YOLO/SAM generation.")
    debug_group.add_argument("--yolo_model", type=str, default="checkpoints/facemask/yolov8n-face.pt")
    debug_group.add_argument("--sam_model", type=str, default="checkpoints/facemask/mobile_sam.pt")
    debug_group.add_argument("--face_point", type=parse_face_point, default=None, metavar="X,Y")

    model_group = parser.add_argument_group("Model checkpoints")
    model_group.add_argument("--checkpoint", type=str, required=True, help="Base/LoRA checkpoint folder.")
    model_group.add_argument("--checkpoint2", type=str, default=None, help="Optional second LoRA checkpoint folder.")
    model_group.add_argument("--ip_adapter_path", type=str, default=None, help="Optional explicit ip_adapter.bin path.")
    model_group.add_argument("--pretrained_model", type=str, default="./checkpoints/svdh")

    size_group = parser.add_argument_group("Preprocessing")
    size_group.add_argument("--height", type=int, default=None, help="Output height. Rounded down to a multiple of 8.")
    size_group.add_argument("--width", type=int, default=None, help="Output width. Rounded down to a multiple of 8.")

    infer_group = parser.add_argument_group("Inference")
    infer_group.add_argument("--batch_size", type=int, default=1, help="Number of Stage-2 chunks per diffusion batch.")
    infer_group.add_argument("--num_frames", type=int, default=SVD_CHUNK_FRAMES, help="SVD temporal window size.")
    infer_group.add_argument("--chunk_frames", action="store_true", help="Use the two-stage overlapping scheduler for long videos.")
    infer_group.add_argument("--skip_stage1", action="store_true", help="Skip anchor-frame Stage 1 when --chunk_frames is enabled.")
    infer_group.add_argument("--num_inference_steps", type=int, default=25)
    infer_group.add_argument("--guidance_scale", type=float, default=1.5)
    infer_group.add_argument("--reconstruction_guidance", type=float, default=0.0)
    infer_group.add_argument("--noise_aug_strength", type=float, default=0.02)
    infer_group.add_argument("--motion_bucket_id", type=int, default=0)
    infer_group.add_argument("--zero_clip_emb", action="store_true")
    infer_group.add_argument("--seed", type=int, default=42)
    infer_group.add_argument("--device", type=str, default="cuda")
    infer_group.add_argument("--fp16", action="store_true")
    infer_group.add_argument("--decode_chunk_size", type=int, default=8)

    args = parser.parse_args(argv)
    normalize_subject_args(args)
    validate_args(args, parser)
    return args


# ---------------------------------------------------------------------------
# Input loading  (unchanged)
# ---------------------------------------------------------------------------

def load_frames_from_video(video_path: str) -> list:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from: {video_path}")
    print(f"Loaded {len(frames)} frames from: {video_path}")
    return frames


def natural_sort_key(path: Path):
    """Sort frame folders naturally: frame_2.png before frame_10.png."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


def load_frames_from_folder(folder_path: str) -> list:
    folder = Path(folder_path)
    paths = sorted(
        [path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_sort_key,
    )
    if not paths:
        raise ValueError(f"No images found in: {folder_path}")

    frames = [np.array(Image.open(path).convert("RGB")) for path in paths]
    print(f"Loaded {len(frames)} frames from folder: {folder_path}")
    return frames


def load_input(input_path: str) -> list:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            return load_frames_from_video(str(path))
        raise ValueError(f"Unsupported input file type '{path.suffix}'.")
    if path.is_dir():
        return load_frames_from_folder(str(path))
    raise ValueError(f"Input path does not exist: {input_path}")


# ---------------------------------------------------------------------------
# Face mask generation  (unchanged)
# ---------------------------------------------------------------------------

def _init_yolo_sam(yolo_model_path, sam_model_path):
    try:
        from ultralytics import YOLO, SAM
    except ImportError:
        raise ImportError(
            "ultralytics required for face mask auto-generation.\n"
            "Install: pip install ultralytics"
        )
    if not os.path.exists(yolo_model_path):
        raise FileNotFoundError(f"YOLO weights not found: {yolo_model_path}")
    if not os.path.exists(sam_model_path):
        raise FileNotFoundError(f"SAM weights not found: {sam_model_path}")
    return YOLO(yolo_model_path), SAM(sam_model_path)


def _select_bbox_by_point(bboxes, point_xy):
    px, py = point_xy
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    dists = np.sqrt((cx - px)**2 + (cy - py)**2)
    best  = int(np.argmin(dists))
    return bboxes[best: best + 1]


def _detect_and_segment_frame(frame_bgr, face_model, sam_model, point_xy=None):
    height, width = frame_bgr.shape[:2]
    full_mask = np.zeros((height, width), dtype=np.uint8)
    results = face_model(frame_bgr, verbose=False)[0]
    bboxes  = results.boxes.xyxy.cpu().numpy()
    if len(bboxes) == 0:
        return full_mask
    if point_xy is not None:
        bboxes = _select_bbox_by_point(bboxes, point_xy)
    sam_results = sam_model(frame_bgr, bboxes=bboxes, verbose=False)[0]
    if sam_results.masks is not None:
        for mask in sam_results.masks.data:
            m = mask.cpu().numpy().astype(np.uint8)
            if m.shape[0] != height or m.shape[1] != width:
                m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
            full_mask = cv2.bitwise_or(full_mask, m)
    return full_mask


def generate_face_masks_for_frames(frames_rgb, yolo_model_path, sam_model_path, point_xy=None):
    face_model, sam_model = _init_yolo_sam(yolo_model_path, sam_model_path)
    masks, n_no_face = [], 0
    for frame_rgb in tqdm(frames_rgb, desc="  Generating face masks", unit="frame"):
        m = _detect_and_segment_frame(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                                       face_model, sam_model, point_xy=point_xy)
        masks.append(m)
        if m.max() == 0:
            n_no_face += 1
    if n_no_face > 0:
        print(f"  [WARNING] {n_no_face}/{len(frames_rgb)} frames had no face detected.")
    print(f"  Face mask done. Mean coverage: {np.mean([m.mean() for m in masks])*100:.1f}%")
    return masks


def load_face_masks_from_video(mask_video_path, height, width):
    cap = cv2.VideoCapture(mask_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open mask video: {mask_video_path}")
    masks = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] != height or gray.shape[1] != width:
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
        masks.append((gray >= 128).astype(np.uint8))
    cap.release()
    print(f"Loaded {len(masks)} mask frames from: {mask_video_path}")
    return masks


def _masks_to_tensor(masks, height, width):
    stacked = np.stack(masks, axis=0).astype(np.float32)
    tensor  = torch.from_numpy(stacked).unsqueeze(1)
    tensor  = F.interpolate(tensor, size=(height // 8, width // 8), mode="nearest")
    return tensor  # [T, 1, H/8, W/8]


def prepare_face_masks(args, frames_rgb, height, width, subject_idx=0):
    """
    Prepare face masks for one subject.  subject_idx selects which entry in
    args.face_mask_videos to use (if provided).
    """
    T = len(frames_rgb)

    mask_video = None
    if args.face_mask_videos and subject_idx < len(args.face_mask_videos):
        mask_video = args.face_mask_videos[subject_idx]

    if mask_video is not None:
        if not os.path.exists(mask_video):
            print(f"  [WARNING] face_mask_video[{subject_idx}] not found: {mask_video}. "
                  f"Using full-frame masks.")
        else:
            print(f"  Loading face mask video (subject {subject_idx}): {mask_video}")
            raw = load_face_masks_from_video(mask_video, height, width)
            while len(raw) < T:
                raw.append(np.ones((height, width), dtype=np.uint8))
            return _masks_to_tensor(raw[:T], height, width)

    if args.skip_face_mask_gen:
        print(f"  --skip_face_mask_gen: full-frame masks for subject {subject_idx}.")
        return torch.ones(T, 1, height // 8, width // 8, dtype=torch.float32)

    print(f"  Auto-generating per-frame face masks (subject {subject_idx}) ...")
    raw = generate_face_masks_for_frames(
        frames_rgb,
        yolo_model_path=args.yolo_model,
        sam_model_path=args.sam_model,
        point_xy=getattr(args, "face_point", None),
    )
    resized = []
    for m in raw:
        if m.shape[0] != height or m.shape[1] != width:
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
        if m.max() == 0:
            m = np.ones((height, width), dtype=np.uint8)
        resized.append(m)
    return _masks_to_tensor(resized, height, width)


# ---------------------------------------------------------------------------
# Keypoint heatmap loading  (unchanged)
# ---------------------------------------------------------------------------

def load_keypoint_frames_from_video(kp_video_path, height, width, blur_sigma=2.0):
    cap = cv2.VideoCapture(kp_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open keypoint video: {kp_video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[0] != height or frame_rgb.shape[1] != width:
            frame_rgb = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(frame_rgb)
    cap.release()
    if not frames:
        raise ValueError(f"No frames from keypoint video: {kp_video_path}")
    print(f"Loaded {len(frames)} keypoint frames from: {kp_video_path}")

    kernel_size = int(6 * blur_sigma + 1) | 1
    frames_f = [f.astype(np.float32) / 255.0 for f in frames]
    frames_blurred = [
        np.clip(cv2.GaussianBlur(f, (kernel_size, kernel_size),
                                 sigmaX=blur_sigma, sigmaY=blur_sigma), 0, 1)
        for f in frames_f
    ]
    arr    = np.stack(frames_blurred, axis=0)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float()
    return tensor  # [T, 3, H, W]


def prepare_keypoint_frames(kp_video_path, total_frames, height, width, blur_sigma):
    """Load and validate one keypoint video. Returns [T, 3, H, W] or None."""
    if kp_video_path is None:
        print("  No keypoint video — keypoint conditioning disabled for this subject.")
        return None
    if not os.path.exists(kp_video_path):
        print(f"  [WARNING] Keypoint video not found: {kp_video_path}. Disabled.")
        return None
    kp = load_keypoint_frames_from_video(kp_video_path, height, width, blur_sigma)
    if kp.shape[0] != total_frames:
        print(f"  [WARNING] KP video {kp.shape[0]} frames vs input {total_frames}. Adjusting.")
        if kp.shape[0] < total_frames:
            pad = kp[-1:].expand(total_frames - kp.shape[0], -1, -1, -1)
            kp = torch.cat([kp, pad], dim=0)
        kp = kp[:total_frames]
    print(f"  Keypoint frames: {kp.shape}  [{kp.min():.3f}, {kp.max():.3f}]")
    return kp


def _slice_kp_list(kp_list, frame_indices):
    """Slice a list of [T, 3, H, W] keypoint tensors → list of [1, T_chunk, 3, H, W]."""
    if kp_list is None:
        return None
    result = []
    for kp in kp_list:
        if kp is None:
            result.append(None)
        else:
            result.append(kp[frame_indices].unsqueeze(0))
    return result


# ---------------------------------------------------------------------------
# Preprocessing  (unchanged)
# ---------------------------------------------------------------------------

def preprocess_frames(frames, height, width):
    orig_h, orig_w = frames[0].shape[:2]
    if height is None:
        height = (orig_h // 8) * 8
    if width is None:
        width = (orig_w // 8) * 8
    height = (height // 8) * 8
    width = (width // 8) * 8

    print(f"Preprocessing {len(frames)} frames: {orig_h}x{orig_w} -> {height}x{width}")
    processed = []
    for frame in frames:
        img = Image.fromarray(frame).resize((width, height), Image.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0 * 2.0 - 1.0
        processed.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(processed, dim=0).unsqueeze(0)  # [1, T, C, H, W]


def preprocess_reference_image(image_path, height, width):
    """Load one reference image -> [1, C, H, W] in [-1, 1]."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Reference image not found: {image_path}")
    img = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args):
    """Load SVD + IP-Adapter (NO Track4Gen Refiner) using SVDIPAdapterWrapperMulti."""
    if NO_REFINER_IMPORT_ERROR is not None:
        raise ImportError(
            "Could not import the no-Track4Gen IP-Adapter/FaRL UNet wrapper. "
            "Install the model dependencies first; for FaRL this usually includes "
            "the `clip` Python package."
        ) from NO_REFINER_IMPORT_ERROR

    device       = torch.device(args.device)
    weight_dtype = torch.float16 if args.fp16 else torch.float32

    print(f"Loading base model from: {args.pretrained_model}")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model, subfolder="image_encoder"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model, subfolder="vae", variant="fp16"
    )

    lora_path = os.path.join(args.checkpoint, "unet_lora")
    if os.path.exists(lora_path):
        print(f"  Loading LoRA from: {lora_path}")
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            args.pretrained_model, subfolder="unet", low_cpu_mem_usage=True, variant="fp16"
        )
        unet = PeftModel.from_pretrained(unet, lora_path).merge_and_unload()
    else:
        print(f"  Loading full UNet from: {args.checkpoint}")
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            args.checkpoint, subfolder="unet", low_cpu_mem_usage=True
        )

    if args.checkpoint2 is not None:
        lora_path2 = os.path.join(args.checkpoint2, "unet_lora")
        if os.path.exists(lora_path2):
            print(f"  Loading second LoRA from: {lora_path2}")
            unet = PeftModel.from_pretrained(unet, lora_path2).merge_and_unload()
        else:
            print(f"  [WARNING] No unet_lora/ in --checkpoint2. Skipping.")

    # ── Multi-subject wrapper ─────────────────────────────────────────────────
    print("Wrapping UNet with SVDIPAdapterWrapperMulti ...")
    unet = SVDIPAdapterWrapperMulti(unet=unet)

    ip_adapter_path = args.ip_adapter_path
    if ip_adapter_path is None:
        candidate = os.path.join(args.checkpoint, "ip_adapter.bin")
        if os.path.exists(candidate):
            ip_adapter_path = candidate

    if ip_adapter_path and os.path.exists(ip_adapter_path):
        print(f"  Loading IP-Adapter weights from: {ip_adapter_path}")
        ip_state = torch.load(ip_adapter_path, map_location="cpu")

        if "image_proj1" in ip_state:
            unet.image_proj1.load_state_dict(ip_state["image_proj1"])
            print("  image_proj1 (ArcFace) loaded.")
        elif "image_proj" in ip_state:
            unet.image_proj1.load_state_dict(ip_state["image_proj"])
            print("  image_proj1 loaded from legacy key.")
        else:
            print("  [WARNING] No image_proj1 — ArcFace projection randomly initialised.")

        if "image_proj2" in ip_state:
            unet.image_proj2.load_state_dict(ip_state["image_proj2"])
            print("  image_proj2 (FaRL) loaded.")
        else:
            print("  [WARNING] No image_proj2 — FaRL projection randomly initialised.")

        if "keypoint_encoder" in ip_state:
            unet.keypoint_encoder.load_state_dict(ip_state["keypoint_encoder"])
            print("  KeypointEncoder loaded.")
        else:
            print("  [WARNING] No keypoint_encoder — randomly initialised.")

        if "id_attn" in ip_state:
            unet.id_attn.load_state_dict(ip_state["id_attn"])
            print("  id_attn loaded.")
        else:
            print("  [WARNING] No id_attn.")

        # NO Track4Gen Refiner in this variant — silently ignore "refiner" key
        # if a Track4Gen-trained checkpoint is loaded by mistake.
        if "refiner" in ip_state:
            print("  [INFO] checkpoint has 'refiner' weights but this variant "
                  "has no Refiner module — ignoring.")

        # Load per-attention-layer k/v weights.
        # MultiSubjectIPAttnProcessor is a subclass of IPMaskedSpatioTemporalAttnProcessor,
        # so isinstance checks work transparently.
        for name, proc in unet.unet.attn_processors.items():
            if isinstance(proc, IPMaskedSpatioTemporalAttnProcessor):
                if name in ip_state.get("ip_adapter", {}):
                    proc.load_state_dict(ip_state["ip_adapter"][name])
        print("  IP-Adapter attention processor weights loaded.")
    elif args.disable_ipadapter:
        print("  [INFO] Identity Adapter disabled; adapter weights are unused for baseline VDD.")
    else:
        print("  [WARNING] No ip_adapter.bin found — weights randomly initialised.")

    unet.eval()
    unet.arcface.eval()
    image_encoder.eval()
    vae.eval()
    unet.requires_grad_(False)
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    unet.to(device, dtype=weight_dtype)
    image_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)

    # ArcFace explicitly operates in fp32 (its implementation notes that half
    # precision is unsupported), and both face extractors receive float inputs
    # in the identity-conditioning path. Moving the whole wrapper to fp16 above
    # would otherwise leave their convolution weights in half precision and
    # fail with FloatTensor/HalfTensor input-weight mismatches.
    if not args.disable_ipadapter:
        unet.arcface.to(device=device, dtype=torch.float32)
        unet.farl.to(device=device, dtype=torch.float32)
        print("Keeping ArcFace and FaRL feature extractors in fp32.")

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        unet=unet.unet,
        image_encoder=image_encoder,
        vae=vae,
        revision=None,
        torch_dtype=weight_dtype,
    )
    pipeline.register_modules(unet=unet)
    pipeline.vae           = pipeline.vae.to(device)
    pipeline.image_encoder = pipeline.image_encoder.to(device)

    print(f"Pipeline ready on {device} ({weight_dtype})")
    return pipeline, device, weight_dtype


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_inference_chunk(
    pipeline,
    video_chunk: torch.Tensor,
    reference_images: "list[torch.Tensor] | None",  # one [B, C, H, W] per subject
    args,
    device: torch.device,
    keypoint_frames_chunks: "list[torch.Tensor | None] | None" = None,  # one [B,T,3,H,W] per subject
    blur_frames_chunk: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """
    Run inference on one batch of chunks.

    reference_images       : list of [B, C, H, W] tensors, one per subject
    keypoint_frames_chunks : list of [B, T, 3, H, W] tensors, one per subject (or None entries)
    blur_frames_chunk      : [B, T, 3, H, W]  (shared across subjects)
    """
    B          = video_chunk.shape[0]
    H, W       = video_chunk.shape[3], video_chunk.shape[4]
    num_frames = video_chunk.shape[1]

    # Batch each per-subject reference image
    ref_imgs_batched = None
    if reference_images is not None:
        ref_imgs_batched = [img.expand(B, -1, -1, -1).to(device) for img in reference_images]

    cross_attention_kwargs = None
    if keypoint_frames_chunks is not None or blur_frames_chunk is not None:
        kp_on_device = None
        if keypoint_frames_chunks is not None:
            kp_on_device = [kp.to(device) if kp is not None else None
                            for kp in keypoint_frames_chunks]
        cross_attention_kwargs = {
            "keypoint_frames": kp_on_device,
            "blur_frames": blur_frames_chunk.to(device) if blur_frames_chunk is not None else None,
        }

    with torch.no_grad():
        output = pipeline(
            video_chunk.to(device),
            height=H,
            width=W,
            num_frames=num_frames,
            decode_chunk_size=args.decode_chunk_size,
            motion_bucket_id=args.motion_bucket_id,
            min_guidance_scale=args.guidance_scale,
            max_guidance_scale=args.guidance_scale,
            reconstruction_guidance_scale=args.reconstruction_guidance,
            fps=7,
            noise_aug_strength=args.noise_aug_strength,
            num_inference_steps=args.num_inference_steps,
            focal_stack_num=None,
            zero_clip_emb=args.zero_clip_emb,
            reference_image=ref_imgs_batched,  # list of tensors
            cross_attention_kwargs=cross_attention_kwargs,
        )

    frames_out = output.frames
    frames_out = torch.clamp(frames_out * 0.5 + 0.5, 0, 1)
    frames_out = frames_out.permute(0, 2, 1, 3, 4).cpu()  # [B, T, C, H, W]
    return frames_out


def run_stage1_ipadapter(
    pipeline,
    video_tensor: torch.Tensor,
    reference_images: "list[torch.Tensor] | None",
    args,
    device: torch.device,
    keypoint_frames_list: "list[torch.Tensor | None] | None" = None,
    blur_frames: "torch.Tensor | None" = None,
) -> tuple:
    """
    Stage 1: Anchor-frame inference with IP-Adapter + keypoint conditioning
    (multi-subject variant).
    """
    N = video_tensor.shape[1]
    T = args.num_frames

    anchor_indices = np.linspace(0, N - 1, T, endpoint=True).round().astype(int)
    anchor_indices = np.clip(anchor_indices, 0, N - 1)

    print(f"\n{'='*60}")
    print(f"  STAGE 1 — anchor-frame inference (multi-subject, {len(reference_images or [])} subject(s))")
    print(f"{'='*60}")
    print(f"  Video length  : {N} frames")
    print(f"  Anchor count  : {T}")
    print(f"  Anchor indices: {anchor_indices.tolist()}")

    anchor_tensor = video_tensor[0, anchor_indices].unsqueeze(0)  # [1, T, C, H, W]

    kp_anchor = _slice_kp_list(keypoint_frames_list, anchor_indices)

    bl_anchor = None
    if blur_frames is not None:
        bl_anchor = blur_frames[anchor_indices].unsqueeze(0)  # [1, T, 3, H, W]

    batch_out = run_inference_chunk(
        pipeline, anchor_tensor, reference_images, args, device,
        keypoint_frames_chunks=kp_anchor,
        blur_frames_chunk=bl_anchor,
    )  # [1, T, C, H, W] in [0, 1]
    enhanced_frames = batch_out[0]  # [T, C, H, W]

    stage1_normalized = enhanced_frames * 2.0 - 1.0
    for i, idx in enumerate(anchor_indices):
        video_tensor[0, idx] = stage1_normalized[i]

    print(f"  Stage 1 complete. Injected {T} anchor frames (hard blend).")
    return anchor_indices, enhanced_frames


def stage2_padding_frames(total_frames: int) -> int:
    """Return how many repeated tail frames Stage 2 needs for 8-frame strides."""
    remainder = (total_frames - SVD_CHUNK_FRAMES) % SVD_CHUNK_NEW_FRAMES
    return 0 if remainder == 0 else SVD_CHUNK_NEW_FRAMES - remainder


def pad_for_stage2(
    video_tensor: torch.Tensor,
    blur_frames: torch.Tensor,
    keypoint_frames_list: "list[torch.Tensor | None] | None",
    pad_frames_num: int,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor | None] | None]:
    """Pad input, blur frames, and optional keypoint tensors by repeating the last frame."""
    if pad_frames_num <= 0:
        return video_tensor, blur_frames, keypoint_frames_list

    video_pad = video_tensor[:, -1:].expand(-1, pad_frames_num, -1, -1, -1)
    video_tensor = torch.cat([video_tensor, video_pad], dim=1)

    blur_pad = blur_frames[-1:].expand(pad_frames_num, -1, -1, -1)
    blur_frames = torch.cat([blur_frames, blur_pad], dim=0)

    if keypoint_frames_list is not None:
        padded_keypoints = []
        for kp in keypoint_frames_list:
            if kp is not None:
                kp_pad = kp[-1:].expand(pad_frames_num, -1, -1, -1)
                kp = torch.cat([kp, kp_pad], dim=0)
            padded_keypoints.append(kp)
        keypoint_frames_list = padded_keypoints

    print(f"Padded {pad_frames_num} frame(s). New total: {video_tensor.shape[1]}")
    return video_tensor, blur_frames, keypoint_frames_list


def build_stage2_chunks(video_tensor: torch.Tensor) -> list[tuple[torch.Tensor, list[int]]]:
    """Build the 14-frame overlapping chunks used by Stage 2.

    Each item is ``(chunk_tensor, source_frame_indices)``. The source indices are
    also used to align keypoint and blur-frame conditioning with each chunk.
    """
    _, _, channels, height, width = video_tensor.shape

    first_frame_indices = list(range(SVD_CHUNK_FRAMES))
    first_chunk = video_tensor[0, first_frame_indices].cpu()
    chunks: list[tuple[torch.Tensor, list[int]]] = [(first_chunk, first_frame_indices)]

    end_frame = SVD_CHUNK_FRAMES
    prev_chunk = first_chunk.clone()
    prev_indices = first_frame_indices.copy()

    while end_frame < video_tensor.shape[1]:
        new_indices = list(range(end_frame, min(end_frame + SVD_CHUNK_NEW_FRAMES, video_tensor.shape[1])))
        while len(new_indices) < SVD_CHUNK_NEW_FRAMES:
            new_indices.append(new_indices[-1])

        chunk = torch.zeros(SVD_CHUNK_FRAMES, channels, height, width, dtype=first_chunk.dtype)
        chunk[SVD_NEW_FRAME_SLICE] = video_tensor[0, new_indices].cpu()
        chunk[SVD_CONTEXT_OUTPUT_SLICE] = prev_chunk[SVD_CONTEXT_TAIL_SLICE]
        chunk[1] = prev_chunk[SVD_CONTEXT_MID_INDEX]
        chunk[0] = prev_chunk[SVD_CONTEXT_HEAD_INDEX]

        frame_indices = (
            [prev_indices[SVD_CONTEXT_HEAD_INDEX], prev_indices[SVD_CONTEXT_MID_INDEX]]
            + prev_indices[SVD_CONTEXT_TAIL_SLICE]
            + new_indices
        )
        chunks.append((chunk, frame_indices))
        prev_chunk = chunk.clone()
        prev_indices = frame_indices
        end_frame += SVD_CHUNK_NEW_FRAMES

    return chunks


def run_stage2_batches(
    pipeline,
    chunks: list[tuple[torch.Tensor, list[int]]],
    reference_images: "list[torch.Tensor] | None",
    args,
    device: torch.device,
    blur_frames: torch.Tensor,
    keypoint_frames_list: "list[torch.Tensor | None] | None" = None,
) -> list[torch.Tensor]:
    """Run diffusion over already-built Stage-2 chunks."""
    print(f"Chunking {len(blur_frames)} frames -> {len(chunks)} chunks, batch size {args.batch_size} ...")
    all_chunk_outputs: list[torch.Tensor] = []

    for batch_start in range(0, len(chunks), args.batch_size):
        batch = chunks[batch_start: batch_start + args.batch_size]
        current_bs = len(batch)
        batch_video = torch.stack([chunk for chunk, _ in batch], dim=0)

        batch_kp_chunks = None
        if keypoint_frames_list is not None:
            batch_kp_chunks = []
            for kp in keypoint_frames_list:
                if kp is None:
                    batch_kp_chunks.append(None)
                else:
                    batch_kp_chunks.append(torch.stack([kp[indices] for _, indices in batch], dim=0))

        batch_blur = torch.stack([blur_frames[indices] for _, indices in batch], dim=0)

        print(f"  Chunks {batch_start}..{batch_start + current_bs - 1} ({current_bs} in batch) ...")
        frames_out = run_inference_chunk(
            pipeline,
            batch_video,
            reference_images,
            args,
            device,
            keypoint_frames_chunks=batch_kp_chunks,
            blur_frames_chunk=batch_blur,
        )
        all_chunk_outputs.extend(frames_out[b] for b in range(current_bs))

    return all_chunk_outputs


def blend_stage2_outputs(chunk_outputs: list[torch.Tensor], padded_frame_count: int, channels: int, height: int, width: int) -> torch.Tensor:
    """Blend Stage-2 chunk outputs back into one continuous video."""
    output_frames = torch.zeros(padded_frame_count, channels, height, width)
    output_frames[:SVD_CHUNK_FRAMES] = chunk_outputs[0]

    end_frame = SVD_CHUNK_FRAMES
    for chunk_out in chunk_outputs[1:]:
        output_frames[end_frame:end_frame + SVD_CHUNK_NEW_FRAMES] = chunk_out[SVD_NEW_FRAME_SLICE]
        for target_offset, source_idx, old_weight, new_weight in SVD_BLEND_RULES:
            target_idx = end_frame + target_offset
            output_frames[target_idx] = old_weight * output_frames[target_idx] + new_weight * chunk_out[source_idx]
        end_frame += SVD_CHUNK_NEW_FRAMES

    return output_frames


def run_inference(
    pipeline,
    video_tensor: torch.Tensor,
    reference_images: "list[torch.Tensor] | None",
    args,
    device: torch.device,
    keypoint_frames_list: "list[torch.Tensor | None] | None" = None,
) -> torch.Tensor:
    """Run either a single SVD pass or the two-stage overlapping scheduler."""
    total_frames = video_tensor.shape[1]
    channels, height, width = video_tensor.shape[2], video_tensor.shape[3], video_tensor.shape[4]
    blur_frames = (video_tensor[0].float() * 0.5 + 0.5).clamp(0.0, 1.0)  # [N, C, H, W]

    if not args.chunk_frames or total_frames <= args.num_frames:
        print(f"Running inference on {total_frames} frames (single pass) ...")
        kp_chunks = [kp.unsqueeze(0) if kp is not None else None for kp in keypoint_frames_list] if keypoint_frames_list else None
        out = run_inference_chunk(
            pipeline,
            video_tensor,
            reference_images,
            args,
            device,
            keypoint_frames_chunks=kp_chunks,
            blur_frames_chunk=blur_frames.unsqueeze(0),
        )
        return out[0]

    pad_frames_num = stage2_padding_frames(total_frames)
    video_tensor, blur_frames, keypoint_frames_list = pad_for_stage2(
        video_tensor,
        blur_frames,
        keypoint_frames_list,
        pad_frames_num,
    )

    if not args.skip_stage1:
        keypoints_unpadded = (
            [kp[:total_frames] if kp is not None else None for kp in keypoint_frames_list]
            if keypoint_frames_list else None
        )
        run_stage1_ipadapter(
            pipeline,
            video_tensor[:, :total_frames],
            reference_images,
            args,
            device,
            keypoint_frames_list=keypoints_unpadded,
            blur_frames=blur_frames[:total_frames],
        )
        blur_frames[:total_frames] = (video_tensor[0, :total_frames].float() * 0.5 + 0.5).clamp(0.0, 1.0)
    else:
        print("\n  --skip_stage1: bypassing Stage 1.")

    print(f"\n{'='*60}")
    print("  STAGE 2 - overlapping-chunk inference")
    print(f"{'='*60}")

    chunks = build_stage2_chunks(video_tensor)
    chunk_outputs = run_stage2_batches(
        pipeline,
        chunks,
        reference_images,
        args,
        device,
        blur_frames,
        keypoint_frames_list=keypoint_frames_list,
    )
    output_frames = blend_stage2_outputs(chunk_outputs, video_tensor.shape[1], channels, height, width)
    return output_frames[:total_frames]


# ---------------------------------------------------------------------------
# Output saving  (unchanged)
# ---------------------------------------------------------------------------

def save_output(output_frames: torch.Tensor, output_dir: str, fps: int = 24):
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    T    = output_frames.shape[0]
    H, W = output_frames.shape[2], output_frames.shape[3]

    print(f"Saving {T} frames ...")
    for i in range(T):
        frame_np = (output_frames[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(frame_np).save(os.path.join(frames_dir, f"frame_{i:04d}.png"))

    video_path = os.path.join(output_dir, "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
    for i in range(T):
        frame_np = (output_frames[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Done. Output saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Run preparation helpers
# ---------------------------------------------------------------------------

def print_config_summary(args: argparse.Namespace, num_frames: int | None = None, height: int | None = None, width: int | None = None) -> None:
    subjects = len(args.reference_images) if args.reference_images else 0
    print("\n=== Inference setup ===")
    print(f"  Variant        : IP-Adapter + KP + FaRL, no Track4Gen Refiner")
    print(f"  Input          : {args.input}")
    print(f"  Output dir     : {args.output_dir}")
    if num_frames is not None and height is not None and width is not None:
        print(f"  Frames/res     : {num_frames} frames at {height}x{width}")
    print(f"  Mode           : {'VDD baseline' if args.disable_ipadapter else f'{subjects} subject(s)'}")
    print(f"  Chunking       : {'on' if args.chunk_frames else 'off'}")
    print(f"  Stage 1        : {'skip' if args.skip_stage1 else 'run'}")
    print(f"  Precision      : {'fp16' if args.fp16 else 'fp32'} on {args.device}")
    print(f"  Mask debug     : {'on' if args.save_face_mask_debug else 'off'}")


def print_subject_summary(args: argparse.Namespace) -> None:
    if args.disable_ipadapter:
        return
    print("\nSubjects:")
    for idx, ref_path in enumerate(args.reference_images):
        kp_path = args.keypoint_videos[idx] if args.keypoint_videos and idx < len(args.keypoint_videos) else None
        mask_path = args.face_mask_videos[idx] if args.face_mask_videos and idx < len(args.face_mask_videos) else None
        print(f"  [{idx}] reference : {ref_path}")
        print(f"      keypoints : {kp_path or 'disabled'}")
        if args.save_face_mask_debug:
            print(f"      mask debug: {mask_path or 'auto/full-frame'}")


def prepare_reference_images(reference_paths: list[str], height: int, width: int) -> list[torch.Tensor]:
    reference_images = []
    for idx, ref_path in enumerate(reference_paths):
        ref = preprocess_reference_image(ref_path, height, width)
        reference_images.append(ref)
        print(f"Reference image [{idx}]: {ref_path}  shape {ref.shape}  [{ref.min():.2f}, {ref.max():.2f}]")
    return reference_images


def maybe_save_face_mask_debug(args, frames, height: int, width: int, num_subjects: int) -> None:
    if not args.save_face_mask_debug:
        print("Skipping face-mask debug generation. Pass --save_face_mask_debug to save visualizations.")
        return

    for idx in range(num_subjects):
        print(f"Preparing face masks (subject {idx}) for visualization ...")
        masks = prepare_face_masks(args, frames, height, width, subject_idx=idx)
        print(f"  Face masks [{idx}]: {masks.shape}  (mean coverage {masks.mean()*100:.1f}%)")
        save_face_masks_visualization(
            masks,
            output_dir=f"{args.output_dir}/debug_facemasks_subject{idx}",
            fps=10,
        )


def prepare_keypoint_frame_list(args, total_frames: int, height: int, width: int, num_subjects: int) -> list[torch.Tensor | None]:
    keypoint_paths = args.keypoint_videos or [None] * num_subjects
    keypoint_frames_list = []
    for idx, kp_path in enumerate(keypoint_paths):
        print(f"Loading keypoint frames (subject {idx}) ...")
        kp = prepare_keypoint_frames(
            kp_path,
            total_frames=total_frames,
            height=height,
            width=width,
            blur_sigma=args.keypoint_blur_sigma,
        )
        keypoint_frames_list.append(kp)
    return keypoint_frames_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    num_subjects = len(args.reference_images) if args.reference_images else 0
    print_subject_summary(args)

    # Load and resize input before dry-run so frame count/resolution are visible.
    frames = load_input(args.input)
    video_tensor = preprocess_frames(frames, args.height, args.width)
    _, total_frames, _, height, width = video_tensor.shape
    print_config_summary(args, num_frames=total_frames, height=height, width=width)
    print(f"Input tensor: {video_tensor.shape}  (B, T, C, H, W)")

    if args.dry_run:
        print("Dry run complete. Model loading and diffusion were skipped.")
        return

    reference_images = None
    keypoint_frames_list = None

    if args.disable_ipadapter:
        print("[INFO] --disable_identity_adapter/--disable_ipadapter: skipping reference images, masks, and keypoints.")
    else:
        reference_images = prepare_reference_images(args.reference_images, height, width)
        maybe_save_face_mask_debug(args, frames, height, width, num_subjects)
        keypoint_frames_list = prepare_keypoint_frame_list(
            args,
            total_frames=total_frames,
            height=height,
            width=width,
            num_subjects=num_subjects,
        )

    pipeline, device, weight_dtype = load_model(args)

    if reference_images is not None:
        reference_images = [img.to(device, dtype=weight_dtype) for img in reference_images]

    output_frames = run_inference(
        pipeline,
        video_tensor,
        reference_images,
        args,
        device,
        keypoint_frames_list=keypoint_frames_list,
    )
    print(f"Output: {output_frames.shape}  (T, C, H, W)")

    save_output(output_frames, args.output_dir, fps=args.output_fps)


if __name__ == "__main__":
    main()
