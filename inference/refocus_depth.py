#!/usr/bin/env python
"""
Video Refocusing Inference Script with Depth/Defocus Map

Inference script for the trained SVD refocus model, capable of depth conditioning.
Accepts a .mp4 video or a folder of .png/.jpg images as input for both RGB and depth,
runs the refocusing pipeline, and saves the output.
"""

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from peft import PeftModel
from simple_pipeline import RefocusDepthStableVideoDiffusionPipeline as StableVideoDiffusionPipeline


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_COC = 100.0

# The overlapping long-video scheduler is written for SVD's 14-frame window.
# Each later chunk contributes 8 new frames and reuses 6 context frames.
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


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    if args.num_frames <= 0:
        parser.error("--num_frames must be positive.")
    if args.output_fps <= 0:
        parser.error("--output_fps must be positive.")
    if args.decode_chunk_size <= 0:
        parser.error("--decode_chunk_size must be positive.")
    if args.coc < 0:
        parser.error("--coc must be non-negative.")
    if args.height is not None and args.height <= 0:
        parser.error("--height must be positive when provided.")
    if args.width is not None and args.width <= 0:
        parser.error("--width must be positive when provided.")
    if args.chunk_frames and args.num_frames != SVD_CHUNK_FRAMES:
        parser.error(
            f"--chunk_frames currently supports --num_frames {SVD_CHUNK_FRAMES} only. "
            "The overlap scheduler is tied to the 14-frame SVD window."
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Video refocusing inference with depth/defocus conditioning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("Input / output")
    io_group.add_argument("--input", type=str, required=True, help="Input RGB video file or image-frame folder.")
    io_group.add_argument("--depth_input", type=str, required=True, help="Depth video file or image-frame folder.")
    io_group.add_argument("--output_dir", type=str, default="./inference_output", help="Directory for output frames/video/debug files.")
    io_group.add_argument("--output_fps", type=int, default=24, help="FPS for the output .mp4 video.")
    io_group.add_argument("--dry_run", action="store_true", help="Load/preprocess inputs and print the resolved setup without loading models.")

    depth_group = parser.add_argument_group("Depth / focal plane")
    depth_group.add_argument(
        "--focal_plane",
        type=str,
        default="0.5",
        help=(
            "Either a scalar focal-plane value, e.g. 0.5, or a txt file with one "
            "value per frame. Comma-separated values in the file are also accepted."
        ),
    )
    depth_group.add_argument("--coc", type=float, default=10.0, help="Circle-of-confusion multiplier for defocus conditioning.")
    depth_group.add_argument(
        "--allow_focal_plane_out_of_range",
        action="store_true",
        help="Allow focal-plane values outside [0, 1] for experiments on another scale.",
    )
    depth_group.add_argument(
        "--skip_debug_outputs",
        action="store_true",
        help="Skip defocus heatmap and focal-plane plot generation.",
    )

    model_group = parser.add_argument_group("Model checkpoints")
    model_group.add_argument("--checkpoint", type=str, required=True, help="Checkpoint dir containing unet_lora/ or full unet/.")
    model_group.add_argument("--checkpoint2", type=str, default=None, help="Optional second checkpoint dir containing unet_lora/.")
    model_group.add_argument("--pretrained_model", type=str, default="./checkpoints/svdh", help="Base pretrained SVD model path.")

    size_group = parser.add_argument_group("Preprocessing")
    size_group.add_argument("--height", type=int, default=None, help="Output height. Rounded down to a multiple of 8.")
    size_group.add_argument("--width", type=int, default=None, help="Output width. Rounded down to a multiple of 8.")

    infer_group = parser.add_argument_group("Inference")
    infer_group.add_argument("--batch_size", type=int, default=1, help="Number of Stage-2 chunks per diffusion batch.")
    infer_group.add_argument("--num_frames", type=int, default=SVD_CHUNK_FRAMES, help="SVD temporal window size.")
    infer_group.add_argument("--chunk_frames", action="store_true", help="Use the overlapping long-video scheduler.")
    infer_group.add_argument("--num_inference_steps", type=int, default=25, help="Number of denoising steps.")
    infer_group.add_argument("--guidance_scale", type=float, default=1.5, help="Classifier-free guidance scale.")
    infer_group.add_argument("--reconstruction_guidance", type=float, default=0.0)
    infer_group.add_argument("--noise_aug_strength", type=float, default=0.02, help="Noise augmentation strength.")
    infer_group.add_argument("--motion_bucket_id", type=int, default=0, help="Motion bucket ID for SVD conditioning.")
    infer_group.add_argument("--zero_clip_emb", action="store_true", help="Zero out CLIP image embedding.")
    infer_group.add_argument("--seed", type=int, default=42, help="Random seed.")
    infer_group.add_argument("--device", type=str, default="cuda", help="Inference device.")
    infer_group.add_argument("--fp16", action="store_true", help="Use float16 precision.")
    infer_group.add_argument("--decode_chunk_size", type=int, default=8, help="VAE decode chunk size.")

    args = parser.parse_args(argv)
    validate_args(args, parser)
    return args


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_frames_from_video(video_path: str) -> list[np.ndarray]:
    """Load all frames from a video as RGB arrays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from video: {video_path}")
    print(f"Loaded {len(frames)} frames from video: {video_path}")
    return frames


def natural_sort_key(path: Path):
    """Sort frame folders naturally: frame_2.png before frame_10.png."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


def load_frames_from_folder(folder_path: str) -> list[np.ndarray]:
    """Load image frames from a folder in natural filename order."""
    folder = Path(folder_path)
    image_paths = sorted(
        [path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_sort_key,
    )
    if not image_paths:
        raise ValueError(f"No .png/.jpg images found in: {folder_path}")

    frames = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
    print(f"Loaded {len(frames)} frames from folder: {folder_path}")
    return frames


def load_input(input_path: str) -> list[np.ndarray]:
    """Auto-detect input type and load frames."""
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            return load_frames_from_video(str(path))
        raise ValueError(f"Unsupported file type '{path.suffix}'. Expected video or image-frame directory.")
    if path.is_dir():
        return load_frames_from_folder(str(path))
    raise ValueError(f"Input path does not exist: {input_path}")


# ---------------------------------------------------------------------------
# Focal-plane helpers
# ---------------------------------------------------------------------------

def parse_float_sequence(value: str) -> list[float]:
    tokens: list[str] = []
    for line in value.splitlines():
        tokens.extend(part.strip() for part in line.split(","))
    return [float(token) for token in tokens if token]


def load_focal_plane_file(path: str | Path) -> list[float]:
    with Path(path).open("r", encoding="utf-8") as file:
        return parse_float_sequence(file.read())


def validate_focal_planes(values: list[float], allow_out_of_range: bool = False) -> None:
    if not values:
        raise ValueError("Focal-plane sequence is empty.")
    if allow_out_of_range:
        return
    for value in values:
        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"Focal-plane value {value} is outside [0, 1]. "
                "Pass --allow_focal_plane_out_of_range if this is intentional."
            )


def build_focal_planes(args: argparse.Namespace, num_frames: int) -> list[float]:
    """Return one focal-plane value per frame.

    ``--focal_plane`` accepts either a scalar or a text file. A one-value file is
    treated like a scalar; otherwise the file must already match the frame count.
    """
    focal_arg = Path(args.focal_plane)
    if focal_arg.is_file():
        values = load_focal_plane_file(focal_arg)
        if len(values) == 1:
            values = values * num_frames
        elif len(values) != num_frames:
            raise ValueError(f"Focal-plane file has {len(values)} values, expected {num_frames} frames.")
    else:
        try:
            values = [float(args.focal_plane)] * num_frames
        except ValueError as exc:
            raise ValueError(
                f"Invalid --focal_plane argument: {args.focal_plane}. "
                "Use a scalar or a valid txt file path."
            ) from exc

    validate_focal_planes(values, args.allow_focal_plane_out_of_range)
    return values


def print_run_summary(args: argparse.Namespace, num_frames: int, height: int, width: int, focal_planes: list[float]) -> None:
    print("\n=== Refocus-depth inference setup ===")
    print(f"  Input       : {args.input}")
    print(f"  Depth input : {args.depth_input}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Frames/res  : {num_frames} frames at {height}x{width}")
    print(f"  Focal plane : {min(focal_planes):.4f} to {max(focal_planes):.4f}")
    print(f"  CoC         : {args.coc}")
    print(f"  Chunking    : {'on' if args.chunk_frames else 'off'}")
    print(f"  Precision   : {'fp16' if args.fp16 else 'fp32'} on {args.device}")
    print(f"  Debug files : {'off' if args.skip_debug_outputs else 'on'}")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_frames(
    frames: list[np.ndarray],
    height: int | None,
    width: int | None,
) -> torch.Tensor:
    """
    Resize and normalize frames to a tensor of shape [1, T, C, H, W] in [-1, 1] range.
    Height/width are rounded down to nearest multiple of 8 if not already.
    """
    # Determine target size
    orig_h, orig_w = frames[0].shape[:2]
    if height is None:
        height = (orig_h // 8) * 8
    if width is None:
        width = (orig_w // 8) * 8

    # Ensure divisible by 8
    height = (height // 8) * 8
    width = (width // 8) * 8

    if height == 0 or width == 0:
        raise ValueError(f"Computed height={height}, width={width} is invalid. Input frames are too small.")

    print(f"Preprocessing: {len(frames)} frames, original {orig_h}x{orig_w} -> target {height}x{width}")

    processed = []
    for frame in frames:
        # Resize using PIL for quality
        img = Image.fromarray(frame)
        img = img.resize((width, height), Image.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0  # [0, 1]
        arr = arr * 2.0 - 1.0  # [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [C, H, W]
        processed.append(tensor)

    # Stack: [T, C, H, W] -> [1, T, C, H, W]
    video_tensor = torch.stack(processed, dim=0).unsqueeze(0)
    return video_tensor

def preprocess_depth_frames(
    frames: list[np.ndarray],
    height: int,
    width: int,
    focal_planes: list[float],
    coc: float,
) -> torch.Tensor:
    """
    Resize and normalize depth frames.
    Input range is [0, 255]. We scale to [0, 1].
    Then we apply:
       depth = depth - focal_plane
       depth = depth * coc
       defocus_map = depth / 100.0 (where MAX_COC = 100.0)
    Returns: tensor of shape [1, T, 3, H, W]
    """
    processed = []
    for i, frame in enumerate(frames):
        # Convert to grayscale if it's RGB
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        img = Image.fromarray(frame).convert("L")
        img = img.resize((width, height), Image.LANCZOS)
        
        arr = np.array(img).astype(np.float32) / 255.0  # [0, 1]
        arr = arr - focal_planes[i]
        arr = arr * coc
        arr = arr / MAX_COC
        
        # We need 3 channels for the pipeline to process it like an image
        tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
        processed.append(tensor)

    # Stack: [T, 3, H, W] -> [1, T, 3, H, W]
    video_tensor = torch.stack(processed, dim=0).unsqueeze(0)
    return video_tensor

# ---------------------------------------------------------------------------
# UNet Extender
# ---------------------------------------------------------------------------

def _extend_conv_in(unet, new_in_channels: int = 12) -> None:
    old_conv = unet.conv_in
    old_in_ch = old_conv.in_channels
    
    new_conv = nn.Conv2d(
        new_in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )
    with torch.no_grad():
        new_conv.weight.data[:, :old_in_ch, :, :] = old_conv.weight.data
        new_conv.weight.data[:, old_in_ch:, :, :] = 0.0
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data

    unet.conv_in = new_conv
    unet.register_to_config(in_channels=new_in_channels)

def _load_conv_in(model, path: str) -> None:
    base_model = model.base_model.model if hasattr(model, "base_model") else model
    conv_in_state = torch.load(path, map_location="cpu")
    base_model.conv_in.load_state_dict(conv_in_state)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args):
    """Load the SVD pipeline with UNet from a training checkpoint."""
    device = torch.device(args.device)
    weight_dtype = torch.float16 if args.fp16 else torch.float32

    print(f"Loading base model components from: {args.pretrained_model}")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model, subfolder="image_encoder"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model, subfolder="vae", variant="fp16"
    )

    # Define the base model path (always needed for LoRA)
    base_model_path = args.pretrained_model
    
    # Check if the path provided is a LoRA adapter or a full fine-tuned UNet
    # Based on our training script, LoRA is saved in "unet_lora"
    lora_path = os.path.join(args.checkpoint, "unet_lora")
    conv_in_path = os.path.join(args.checkpoint, "unet_conv_in_refocus.pt")
    
    if os.path.exists(lora_path):
        # 1. Load the base UNet first
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
            low_cpu_mem_usage=True,
            variant="fp16",
            in_channels=8, # Explicitly load as 8 channels first
            ignore_mismatched_sizes=False
        )
        
        # 1.5. Extend to 12 channels for defocus map concat and load the learned conv_in
        _extend_conv_in(unet, 12)
        if os.path.exists(conv_in_path):
            print(f"Loading extended conv_in weights from: {conv_in_path}")
            _load_conv_in(unet, conv_in_path)

        # 2. Wrap it with the LoRA weights
        unet = PeftModel.from_pretrained(unet, lora_path)
        unet = unet.merge_and_unload()
        print('loading lora...')
    else:
        # Fallback for full fine-tuned weights (original logic)
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            args.checkpoint,
            subfolder="unet",
            low_cpu_mem_usage=True,
        )
        print('loading full unet...')

    if args.checkpoint2 is not None:
        lora_path2 = os.path.join(args.checkpoint2, "unet_lora")
        if os.path.exists(lora_path2):
            print(f"  Loading second LoRA adapter from: {lora_path2}")
            unet = PeftModel.from_pretrained(unet, lora_path2)
            unet = unet.merge_and_unload()

    # Set to eval and correct dtype
    unet.eval()
    image_encoder.eval()
    vae.eval()

    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    image_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    # Build pipeline
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        unet=unet,
        image_encoder=image_encoder,
        vae=vae,
        revision=None,
        torch_dtype=weight_dtype,
    )

    print(f"Pipeline loaded on {device} with dtype {weight_dtype}")
    return pipeline, device, weight_dtype


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference_chunk(
    pipeline,
    video_chunk: torch.Tensor,
    defocus_chunk: torch.Tensor,
    args,
    device: torch.device,
) -> torch.Tensor:
    """
    Run inference on a single video chunk of shape [1, T, C, H, W] in [-1, 1].
    Returns output frames tensor [T, C, H, W] in [0, 1].
    """
    num_frames = video_chunk.shape[1]

    with torch.no_grad():
        output = pipeline(
            video_chunk.to(device),
            height=video_chunk.shape[3],
            width=video_chunk.shape[4],
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
            defocus_map=defocus_chunk.to(device) if defocus_chunk is not None else None,
        )

    # output.frames is [B, C, T, H, W] — denormalize from [-1, 1] to [0, 1]
    frames = output.frames[0]  # [C, T, H, W]
    frames = frames * 0.5 + 0.5
    frames = torch.clamp(frames, 0, 1)
    frames = frames.permute(1, 0, 2, 3)  # [T, C, H, W]

    return frames.cpu()


def stage2_padding_frames(total_frames: int) -> int:
    """Return how many repeated tail frames Stage 2 needs for 8-frame strides."""
    remainder = (total_frames - SVD_CHUNK_FRAMES) % SVD_CHUNK_NEW_FRAMES
    return 0 if remainder == 0 else SVD_CHUNK_NEW_FRAMES - remainder


def pad_for_stage2(
    video_tensor: torch.Tensor,
    defocus_tensor: torch.Tensor,
    pad_frames_num: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad RGB and defocus tensors by repeating their final frame."""
    if pad_frames_num <= 0:
        return video_tensor, defocus_tensor

    video_pad = video_tensor[:, -1:].expand(-1, pad_frames_num, -1, -1, -1)
    defocus_pad = defocus_tensor[:, -1:].expand(-1, pad_frames_num, -1, -1, -1)
    video_tensor = torch.cat([video_tensor, video_pad], dim=1)
    defocus_tensor = torch.cat([defocus_tensor, defocus_pad], dim=1)
    print(f"Padded {pad_frames_num} frame(s). New total: {video_tensor.shape[1]}")
    return video_tensor, defocus_tensor


def build_stage2_chunks(
    video_tensor: torch.Tensor,
    defocus_tensor: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build paired RGB/defocus 14-frame chunks for Stage 2."""
    _, _, channels, height, width = video_tensor.shape
    defocus_channels = defocus_tensor.shape[2]

    first_video_chunk = video_tensor[0, :SVD_CHUNK_FRAMES].cpu()
    first_defocus_chunk = defocus_tensor[0, :SVD_CHUNK_FRAMES].cpu()
    chunks: list[tuple[torch.Tensor, torch.Tensor]] = [(first_video_chunk, first_defocus_chunk)]

    end_frame = SVD_CHUNK_FRAMES
    prev_video = first_video_chunk.clone()
    prev_defocus = first_defocus_chunk.clone()

    while end_frame < video_tensor.shape[1]:
        new_slice = slice(end_frame, end_frame + SVD_CHUNK_NEW_FRAMES)

        video_chunk = torch.zeros(SVD_CHUNK_FRAMES, channels, height, width, dtype=first_video_chunk.dtype)
        video_chunk[SVD_NEW_FRAME_SLICE] = video_tensor[0, new_slice].cpu()
        video_chunk[SVD_CONTEXT_OUTPUT_SLICE] = prev_video[SVD_CONTEXT_TAIL_SLICE]
        video_chunk[1] = prev_video[SVD_CONTEXT_MID_INDEX]
        video_chunk[0] = prev_video[SVD_CONTEXT_HEAD_INDEX]

        defocus_chunk = torch.zeros(SVD_CHUNK_FRAMES, defocus_channels, height, width, dtype=first_defocus_chunk.dtype)
        defocus_chunk[SVD_NEW_FRAME_SLICE] = defocus_tensor[0, new_slice].cpu()
        defocus_chunk[SVD_CONTEXT_OUTPUT_SLICE] = prev_defocus[SVD_CONTEXT_TAIL_SLICE]
        defocus_chunk[1] = prev_defocus[SVD_CONTEXT_MID_INDEX]
        defocus_chunk[0] = prev_defocus[SVD_CONTEXT_HEAD_INDEX]

        chunks.append((video_chunk, defocus_chunk))
        prev_video = video_chunk.clone()
        prev_defocus = defocus_chunk.clone()
        end_frame += SVD_CHUNK_NEW_FRAMES

    return chunks


def run_inference_batch(
    pipeline,
    batch_tensor: torch.Tensor,
    batch_defocus_tensor: torch.Tensor,
    args,
    device: torch.device,
) -> torch.Tensor:
    """Run a batch of SVD chunks and return [B, T, C, H, W] in [0, 1]."""
    batch_tensor = batch_tensor.to(device)
    batch_defocus_tensor = batch_defocus_tensor.to(device)
    with torch.no_grad():
        output = pipeline(
            batch_tensor,
            height=batch_tensor.shape[3],
            width=batch_tensor.shape[4],
            num_frames=batch_tensor.shape[1],
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
            defocus_map=batch_defocus_tensor,
        )

    frames_out = output.frames * 0.5 + 0.5  # [B, C, T, H, W]
    frames_out = torch.clamp(frames_out, 0, 1)
    return frames_out.permute(0, 2, 1, 3, 4).cpu()


def run_stage2_batches(
    pipeline,
    chunks: list[tuple[torch.Tensor, torch.Tensor]],
    args,
    device: torch.device,
) -> list[torch.Tensor]:
    print(f"Chunking into {len(chunks)} chunks, batch size {args.batch_size} ...")
    all_chunk_outputs: list[torch.Tensor] = []

    for batch_start in range(0, len(chunks), args.batch_size):
        batch = chunks[batch_start: batch_start + args.batch_size]
        batch_video = torch.stack([video for video, _ in batch], dim=0)
        batch_defocus = torch.stack([defocus for _, defocus in batch], dim=0)
        current_batch_size = len(batch)
        print(f"  Chunks {batch_start}..{batch_start + current_batch_size - 1} ({current_batch_size} in batch) ...")

        frames_out = run_inference_batch(pipeline, batch_video, batch_defocus, args, device)
        all_chunk_outputs.extend(frames_out[b] for b in range(current_batch_size))

    return all_chunk_outputs


def blend_stage2_outputs(chunk_outputs: list[torch.Tensor], padded_frame_count: int, channels: int, height: int, width: int) -> torch.Tensor:
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


def run_inference(pipeline, video_tensor: torch.Tensor, defocus_tensor: torch.Tensor, args, device: torch.device) -> torch.Tensor:
    total_frames = video_tensor.shape[1]
    channels, height, width = video_tensor.shape[2], video_tensor.shape[3], video_tensor.shape[4]

    if not args.chunk_frames or total_frames <= args.num_frames:
        print(f"Running inference on {total_frames} frames (single pass) ...")
        return run_inference_chunk(pipeline, video_tensor, defocus_tensor, args, device)

    pad_frames_num = stage2_padding_frames(total_frames)
    video_tensor, defocus_tensor = pad_for_stage2(video_tensor, defocus_tensor, pad_frames_num)
    chunks = build_stage2_chunks(video_tensor, defocus_tensor)
    chunk_outputs = run_stage2_batches(pipeline, chunks, args, device)
    output_frames = blend_stage2_outputs(chunk_outputs, video_tensor.shape[1], channels, height, width)
    return output_frames[:total_frames]


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_output(output_frames: torch.Tensor, output_dir: str, fps: int = 24):
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    T = output_frames.shape[0]
    H, W = output_frames.shape[2], output_frames.shape[3]

    print(f"Saving {T} frames to {frames_dir}/")
    for i in range(T):
        frame_np = (output_frames[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(frame_np)
        img.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))

    video_path = os.path.join(output_dir, "output.mp4")
    print(f"Saving video to {video_path} ({T} frames at {fps} fps)")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
    for i in range(T):
        frame_np = (output_frames[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"Done! Output saved to {output_dir}")

def get_cinematic_defocus_palette():
    """
    Creates a 256-color BGR lookup table with the following mapping:
    - Index 0   (Dist 1.0/Furthest): Black
    - Index 128 (Dist 0.5/Far):      Purple
    - Index 200 (Dist 0.2/Near):     Orange
    - Index 255 (Dist 0.0/Focus):    Bright Orange
    """
    # BGR Colors
    black = np.array([0, 0, 0])
    purple = np.array([128, 0, 128])
    orange = np.array([0, 140, 255])
    bright_orange = np.array([0, 200, 255])
    
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    
    for i in range(256):
        if i < 128: # Black to Purple
            alpha = i / 128.0
            color = (1 - alpha) * black + alpha * purple
        elif i < 200: # Purple to Orange
            alpha = (i - 128) / (200 - 128)
            color = (1 - alpha) * purple + alpha * orange
        else: # Orange to Bright Orange
            alpha = (i - 200) / (255 - 200)
            color = (1 - alpha) * orange + alpha * bright_orange
        lut[i, 0] = color
    return lut

def save_defocus_heatmap(depth_frames, focal_planes, output_dir, fps, height, width):
    """
    Visualizes the defocus map based on distance from focal plane.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "defocus_heatmap_cinematic.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    custom_lut = get_cinematic_defocus_palette()

    print(f"Generating Cinematic defocus heatmap (Focus=Bright Orange, Furthest=Black)...")
    
    for i, frame in enumerate(depth_frames):
        # 1. Load depth and ensure grayscale
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        depth = frame.astype(np.float32) / 255.0
        
        # 2. Calculate absolute distance from focal plane
        # Dist 0.0 = On focal plane, Dist 1.0 = Max distance
        dist = np.abs(depth - focal_planes[i])
        
        # 3. Map to LUT index: 0.0 distance -> 255 (Bright Orange), 1.0 distance -> 0 (Black)
        heatmap_intensity = (1.0 - np.clip(dist, 0, 1)) * 255
        heatmap_intensity = heatmap_intensity.astype(np.uint8)
        
        # 4. Apply custom LUT and resize
        color_heatmap = cv2.applyColorMap(heatmap_intensity, custom_lut)
        color_heatmap = cv2.resize(color_heatmap, (width, height))
        
        writer.write(color_heatmap)

    writer.release()
    print(f"Heatmap video saved to: {video_path}")
    

def save_focal_plane_plot(focal_planes: list[float], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(focal_planes) + 1), focal_planes, marker="o")
    plt.title("Focal Plane Values Across Frames")
    plt.xlabel("Frame Number")
    plt.ylabel("Focal Plane Value")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    out_path = output_dir / "focal_plane_plot.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Focal-plane plot saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    frames = load_input(args.input)
    depth_frames = load_input(args.depth_input)
    if len(depth_frames) != len(frames):
        raise ValueError(f"Depth frames ({len(depth_frames)}) != RGB frames ({len(frames)})")

    video_tensor = preprocess_frames(frames, args.height, args.width)
    _, total_frames, _, height, width = video_tensor.shape
    focal_planes = build_focal_planes(args, total_frames)
    defocus_tensor = preprocess_depth_frames(depth_frames, height, width, focal_planes=focal_planes, coc=args.coc)

    print_run_summary(args, total_frames, height, width, focal_planes)
    print(f"Input tensor shape  : {video_tensor.shape}  (B, T, C, H, W)")
    print(f"Defocus tensor shape: {defocus_tensor.shape}  (B, T, C, H, W)")

    if args.dry_run:
        print("Dry run complete. Model loading, diffusion, and debug file writes were skipped.")
        return

    if not args.skip_debug_outputs:
        save_defocus_heatmap(depth_frames, focal_planes, args.output_dir, args.output_fps, height, width)
        save_focal_plane_plot(focal_planes, args.output_dir)

    pipeline, device, _ = load_model(args)
    output_frames = run_inference(pipeline, video_tensor, defocus_tensor, args, device)
    print(f"Output tensor shape: {output_frames.shape}  (T, C, H, W)")

    save_output(output_frames, args.output_dir, fps=args.output_fps)


if __name__ == "__main__":
    main()
