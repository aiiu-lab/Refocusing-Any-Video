import argparse
import os
import tempfile
import cv2
import gradio as gr
import numpy as np
import torch
import spaces

# ---------------------------------------------------------------------------
# Imports from Stage 1: VDD deblur (ipadapter_multi)
# ---------------------------------------------------------------------------
from inference.ipadapter_multi import (
    load_model        as s1_load_model,
    run_inference     as s1_run_inference,
    preprocess_frames as s1_preprocess_frames,
)

# ---------------------------------------------------------------------------
# Imports from Stage 2: video bokeh rendering (refocus_depth)
# ---------------------------------------------------------------------------
from inference.refocus_depth import (
    load_model              as s2_load_model,
    run_inference           as s2_run_inference,
    preprocess_frames       as s2_preprocess_frames,
    preprocess_depth_frames as s2_preprocess_depth_frames,
)

# ---------------------------------------------------------------------------
# Demo constraints
# ---------------------------------------------------------------------------
MAX_FRAMES    = 14    # truncate input to this many frames
MAX_LONG_EDGE = 512   # downscale so the longer edge is at most this many pixels

# Fixed inference parameters (not exposed in the UI)
S1_GUIDANCE_SCALE      = 1.5
S1_NOISE_AUG_STRENGTH  = 0.02
S2_GUIDANCE_SCALE      = 1.5
S2_NOISE_AUG_STRENGTH  = 0.02

# ---------------------------------------------------------------------------
# Model configuration — edit these paths (or set env vars) before launching
# ---------------------------------------------------------------------------
# Stage 1: VDD deblur checkpoint
S1_CHECKPOINT    = os.environ.get("S1_CHECKPOINT", "./checkpoints/vdd_weights/checkpoint-90000")
# Stage 2: refocus-depth checkpoint
S2_CHECKPOINT    = os.environ.get("S2_CHECKPOINT", "./checkpoints/vbr_weights/checkpoint-100000")
# Shared SVD base model
PRETRAINED_MODEL = os.environ.get("PRETRAINED_MODEL", "./checkpoints/svdh")
DEVICE           = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
FP16             = os.environ.get("FP16", "0") == "1"

# Video Depth Anything (VDA) checkpoint and config
VDA_ENCODER    = os.environ.get("VDA_ENCODER", "vitl")   # vits | vitb | vitl
VDA_CHECKPOINT = os.environ.get("VDA_CHECKPOINT", f"./checkpoints/video_depth_anything_{VDA_ENCODER}.pth")
VDA_INPUT_SIZE = 518   # fixed input resolution used during VDA training
VDA_USE_AIF_INPUT = False # False to use the original input video for VDA instead of the Stage 1 AIF output
_VDA_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192,  384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384,  768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

# ---------------------------------------------------------------------------
# Minimal args namespaces mirroring the shell scripts
# ---------------------------------------------------------------------------
def _make_s1_args(**overrides) -> argparse.Namespace:
    """Args for Stage 1 (VDD / ipadapter_multi), matching infer_vdd.sh defaults."""
    defaults = dict(
        checkpoint            = S1_CHECKPOINT,
        checkpoint2           = None,
        pretrained_model      = PRETRAINED_MODEL,
        ip_adapter_path       = None,
        input                 = None,
        output_dir            = "./inference_output_demo_s1",
        output_fps            = 24,
        # VDD mode — no identity adapter
        disable_ipadapter     = True,
        reference_images      = None,
        reference_image       = None,
        keypoint_videos       = None,
        keypoint_video        = None,
        face_mask_videos      = None,
        face_mask_video       = None,
        save_face_mask_debug  = False,
        skip_face_mask_gen    = True,
        yolo_model            = "checkpoints/facemask/yolov8n-face.pt",
        sam_model             = "checkpoints/facemask/mobile_sam.pt",
        # inference knobs
        num_frames            = MAX_FRAMES,
        batch_size            = 1,
        chunk_frames          = False,
        skip_stage1           = False,
        num_inference_steps   = 25,
        guidance_scale        = 1.5,
        reconstruction_guidance = 0.0,
        noise_aug_strength    = 0.02,
        motion_bucket_id      = 0,
        zero_clip_emb         = False,
        decode_chunk_size     = 8,
        seed                  = 42,
        device                = DEVICE,
        fp16                  = FP16,
        height                = None,
        width                 = None,
        keypoint_blur_sigma   = 2.0,
        dry_run               = False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_s2_args(**overrides) -> argparse.Namespace:
    """Args for Stage 2 (refocus-depth), matching infer_vbr_defocus_map.sh defaults."""
    defaults = dict(
        checkpoint            = S2_CHECKPOINT,
        checkpoint2           = None,
        pretrained_model      = PRETRAINED_MODEL,
        input                 = None,
        depth_input           = None,
        output_dir            = "./inference_output_demo_s2",
        output_fps            = 24,
        dry_run               = False,
        # depth / focal plane
        focal_plane           = "0.5",   # overridden per call
        coc                   = 27.0,    # from infer_vbr_defocus_map.sh default
        allow_focal_plane_out_of_range = False,
        skip_debug_outputs    = True,    # no debug files in the demo
        # inference knobs
        num_frames            = MAX_FRAMES,
        batch_size            = 1,
        chunk_frames          = False,
        num_inference_steps   = 25,
        guidance_scale        = 1.5,
        reconstruction_guidance = 0.0,
        noise_aug_strength    = 0.02,
        motion_bucket_id      = 0,
        zero_clip_emb         = False,
        decode_chunk_size     = 8,
        seed                  = 42,
        device                = DEVICE,
        fp16                  = FP16,
        height                = None,
        width                 = None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _load_vda_model():
    """
    Load and return a VideoDepthAnything model ready for eval.
    Mirrors the model-loading block in process_ultravideo960.py main().
    """
    from VideoDepthAnything.video_depth_anything.video_depth import VideoDepthAnything

    if not os.path.exists(VDA_CHECKPOINT):
        raise FileNotFoundError(
            f"VDA checkpoint not found: {VDA_CHECKPOINT}"
            "Set the VDA_CHECKPOINT env var or place the .pth file at the default path."
        )

    print(f"Loading VDA model ({VDA_ENCODER}) from {VDA_CHECKPOINT} ...")
    model = VideoDepthAnything(**_VDA_MODEL_CONFIGS[VDA_ENCODER])
    state_dict = torch.load(VDA_CHECKPOINT, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE).eval()
    print("VDA model ready.")
    return model


# ---------------------------------------------------------------------------
# Load both models once at startup
# ---------------------------------------------------------------------------
print("=" * 60)
print("Loading Stage 1 model (VDD deblur) …")
_s1_base_args = _make_s1_args()
S1_PIPELINE, S1_DEVICE, S1_DTYPE = s1_load_model(_s1_base_args)
print("Stage 1 model ready.")

print("=" * 60)
print("Loading Stage 2 model (refocus-depth / bokeh) …")
_s2_base_args = _make_s2_args()
S2_PIPELINE, S2_DEVICE, S2_DTYPE = s2_load_model(_s2_base_args)
print("Stage 2 model ready.")
print("=" * 60)

# Load VDA once at startup alongside the other two models
print("=" * 60)
print("Loading VDA model (Video Depth Anything) ...")
VDA_MODEL = _load_vda_model()
print("=" * 60)


# ---------------------------------------------------------------------------
# Demo preprocessing helpers
# ---------------------------------------------------------------------------

def _truncate(frames: list) -> list:
    """Keep only the first MAX_FRAMES frames."""
    return frames[:MAX_FRAMES]


def _downscale(frame: np.ndarray) -> np.ndarray:
    """Resize a single H×W×3 uint8 frame so its longer edge <= MAX_LONG_EDGE."""
    H, W = frame.shape[:2]
    long_edge = max(H, W)
    if long_edge <= MAX_LONG_EDGE:
        return frame
    scale = MAX_LONG_EDGE / long_edge
    return cv2.resize(frame, (int(round(W * scale)), int(round(H * scale))),
                      interpolation=cv2.INTER_AREA)


def _apply_demo_constraints(frames: list) -> list:
    """Truncate to MAX_FRAMES then downscale every frame."""
    return [_downscale(f) for f in _truncate(frames)]


def _read_video(path: str) -> tuple[list, float]:
    """Return (list of RGB uint8 frames, fps)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise gr.Error(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise gr.Error("The uploaded video appears to be empty.")
    return frames, fps


def _write_video(frames_uint8: np.ndarray, fps: float) -> str:
    """Write (T, H, W, 3) uint8 array to a temp .mp4 and return its path."""
    T, H, W, _ = frames_uint8.shape
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, fps, (W, H))
    for frame_rgb in frames_uint8:
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    return tmp.name


# ---------------------------------------------------------------------------
# Video Depth Anything (VDA) — model loading + inference
# ---------------------------------------------------------------------------
def run_vda(frames_rgb: list) -> np.ndarray:
    """
    Run Video Depth Anything on a list of RGB frames to produce a disparity map.

    Adapted from process_ultravideo960.py :: process_video(), with the
    file-I/O parts (read_video_frames / VideoWriter) removed — we operate
    entirely in memory.

    Args:
        frames_rgb : list of T x (H, W, 3) uint8 RGB frames.
                     These are the AIF frames produced by Stage 1.

    Returns:
        depth_uint8 : (T, H, W) uint8 disparity map.
                      Convention (inherited from VDA + process_ultravideo960):
                        255 = near (high disparity, close to camera)
                          0 = far  (low disparity, far from camera)
    """
    # VDA infer_video_depth expects (T, H, W, 3) uint8 RGB — same as our list.
    frames_arr = np.stack(frames_rgb, axis=0)   # (T, H, W, 3) uint8

    # A dummy fps; VDA uses it only for temporal-smoothing heuristics.
    dummy_fps = 24.0

    # --- infer (mirrors process_video step 2) --------------------------------
    with torch.no_grad():
        depths, _ = VDA_MODEL.infer_video_depth(
            frames_arr,
            dummy_fps,
            input_size = VDA_INPUT_SIZE,
            device     = DEVICE,
            fp32       = False,
        )
    # depths: (T, H, W) float32, raw relative depth from VDA

    # --- min-max normalise across the whole clip (mirrors process_video step 3)
    d_min, d_max = depths.min(), depths.max()
    if d_max - d_min > 1e-6:
        depths_norm = (depths - d_min) / (d_max - d_min)   # float [0, 1]
    else:
        depths_norm = np.zeros_like(depths)

    # --- scale to uint8 (mirrors process_video step 4, without VideoWriter)
    depth_uint8 = (depths_norm * 255).astype(np.uint8)   # (T, H, W)

    print(f"[VDA] Depth estimated: {depth_uint8.shape}, "
          f"range [{depth_uint8.min()}, {depth_uint8.max()}]")
    return depth_uint8


# ---------------------------------------------------------------------------
# Stage 1: VDD deblur  →  AIF (all-in-focus) video
# ---------------------------------------------------------------------------

def run_stage1(frames_rgb: list, guidance_scale: float, noise_aug_strength: float) -> np.ndarray:
    """
    Run Stage 1 VDD deblur on preprocessed RGB frames.

    Args:
        frames_rgb        : list of T × (H, W, 3) uint8 frames (already constrained).
        guidance_scale    : diffusion guidance scale.
        noise_aug_strength: noise augmentation strength.

    Returns:
        AIF video as (T, H, W, 3) uint8 ndarray.
    """
    torch.manual_seed(_s1_base_args.seed)
    np.random.seed(_s1_base_args.seed)

    args = _make_s1_args(
        guidance_scale      = guidance_scale,
        noise_aug_strength  = noise_aug_strength,
    )

    # list[H×W×3 uint8] → [1, T, C, H, W] float in [-1, 1]
    video_tensor = s1_preprocess_frames(frames_rgb, args.height, args.width)

    # → (T, C, H, W) float in [0, 1]
    output_frames = s1_run_inference(
        pipeline             = S1_PIPELINE,
        video_tensor         = video_tensor,
        reference_images     = None,
        args                 = args,
        device               = S1_DEVICE,
        keypoint_frames_list = None,
    )

    # → (T, H, W, 3) uint8
    return (output_frames.permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 2: refocus-depth  →  bokeh-rendered video
# ---------------------------------------------------------------------------

def run_stage2(
    aif_frames_rgb: np.ndarray,
    depth_frames: np.ndarray,
    focal_plane: float,
    coc: float,
    guidance_scale: float,
    noise_aug_strength: float,
) -> np.ndarray:
    """
    Run Stage 2 refocus-depth bokeh rendering.

    Args:
        aif_frames_rgb    : (T, H, W, 3) uint8 AIF video from Stage 1.
        depth_frames      : (T, H, W)    uint8 disparity from VDA.
                            255=near, 0=far.
        focal_plane       : float in [0, 1] — which disparity depth to keep sharp.
        coc               : circle-of-confusion multiplier (default 27.0).
        guidance_scale    : diffusion guidance scale.
        noise_aug_strength: noise augmentation strength.

    Returns:
        Bokeh-rendered video as (T, H, W, 3) uint8 ndarray.
    """
    torch.manual_seed(_s2_base_args.seed)
    np.random.seed(_s2_base_args.seed)

    args = _make_s2_args(
        focal_plane        = str(focal_plane),   # refocus_depth.build_focal_planes expects str
        coc                = coc,
        guidance_scale     = guidance_scale,
        noise_aug_strength = noise_aug_strength,
    )

    T = aif_frames_rgb.shape[0]
    focal_planes_list = [focal_plane] * T   # fixed focal plane for all frames

    # AIF frames: (T, H, W, 3) uint8 list → [1, T, C, H, W] float in [-1, 1]
    aif_list = [aif_frames_rgb[i] for i in range(T)]
    video_tensor = s2_preprocess_frames(aif_list, args.height, args.width)
    _, _, _, H, W = video_tensor.shape

    # Depth frames: (T, H, W) uint8 → list of H×W uint8 arrays (grayscale)
    depth_list = [depth_frames[i] for i in range(T)]

    # → [1, T, 3, H, W] defocus map tensor
    defocus_tensor = s2_preprocess_depth_frames(
        depth_list, H, W,
        focal_planes=focal_planes_list,
        coc=coc,
    )

    # → (T, C, H, W) float in [0, 1]
    output_frames = s2_run_inference(
        pipeline       = S2_PIPELINE,
        video_tensor   = video_tensor,
        defocus_tensor = defocus_tensor,
        args           = args,
        device         = S2_DEVICE,
    )

    # → (T, H, W, 3) uint8
    return (output_frames.permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Full pipeline: input video → AIF → bokeh-rendered
# ---------------------------------------------------------------------------
@spaces.GPU
def predict(
    video_path: str,
    focal_plane: float,
    coc: float,
) -> tuple[str, str]:
    """
    Gradio entry point for the two-stage pipeline.

    Returns:
        (aif_video_path, refocused_video_path) — both as temp .mp4 file paths.
    """
    # ── Read & constrain input ───────────────────────────────────────────────
    frames, fps = _read_video(video_path)
    frames = _apply_demo_constraints(frames)   # truncate + downscale

    # ── Stage 1: VDD deblur → AIF ────────────────────────────────────────────
    print("\n[Stage 1] Running VDD deblur …")
    aif_frames = run_stage1(frames, S1_GUIDANCE_SCALE, S1_NOISE_AUG_STRENGTH)  # (T, H, W, 3) uint8
    aif_path   = _write_video(aif_frames, fps)
    print(f"[Stage 1] Done. AIF video: {aif_path}")

    # ── VDA: estimate disparity from AIF ─────────────────────────────────────
    vda_source = "AIF" if VDA_USE_AIF_INPUT else "original input"
    print(f"\n[VDA] Estimating depth from {vda_source} video ...")
    vda_input_frames = aif_frames if VDA_USE_AIF_INPUT else np.stack(frames, axis=0)
    depth_frames = run_vda([vda_input_frames[i] for i in range(vda_input_frames.shape[0])])
    print("[VDA] Done.")

    # ── Stage 2: refocus-depth → bokeh ───────────────────────────────────────
    print("\n[Stage 2] Running refocus-depth bokeh rendering …")
    refocused_frames = run_stage2(
        aif_frames_rgb     = aif_frames,
        depth_frames       = depth_frames,
        focal_plane        = focal_plane,
        coc                = coc,
        guidance_scale     = S2_GUIDANCE_SCALE,
        noise_aug_strength = S2_NOISE_AUG_STRENGTH,
    )
    refocused_path = _write_video(refocused_frames, fps)
    print(f"[Stage 2] Done. Refocused video: {refocused_path}")

    return aif_path, refocused_path


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Refocusing Any Video (RAV)") as demo:
    gr.Markdown(
        """
        # 🎬 Refocusing Any Video (RAV)
        Given an in-the-wild video and a target focal plane, RAV refocuses the video according to
        the specified focal plane, including dynamic (time-varying) focal planes.
        > **Note**: This demo only supports a single fixed focal plane. For dynamic focal planes,
        see our [GitHub repository](https://github.com/aiiu-lab/Refocusing-Any-Video).

        ### Pipeline
        **Stage 1**:Video Defocus Deblurring: input video → all-in-focus (AIF) video  
        **Auxiliary - Depth estimation**: We use off-the-shelf Video Depth Anything: AIF video → per-frame disparity maps  
        **Stage 2**: Video Bokeh Rendering: AIF + conditioning depth + conditioning focal plane → refocused video

        ### Instructions
        Upload a video, set the focal plane and bokeh controls, then click **Run**. (inference should take around 30~60 seconds)  
        **Note:** Due to resource constraints, the input is automatically:
        - **Downscaled** — longer edge resized to 512 px.
        - **Truncated** — only the first 14 frames are used.
        """
    )

    with gr.Row():
        # ── Left column: inputs ───────────────────────────────────────────────
        with gr.Column():
            input_video = gr.Video(label="Upload input video here.", sources=["upload"])

            gr.Markdown("#### Bokeh Controls")
            focal_plane_slider = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                step=0.01,
                value=0.5,
                label="Target Focal Plane",
                info="Disparity value to keep sharp. 1.0 = near, 0.0 = far.",
            )
            coc_slider = gr.Slider(
                minimum=1.0, maximum=50.0, step=1.0, value=25.0,
                label="Circle of Confusion (CoC)",
                info="Strength of the bokeh blur. Higher = more blur.",
            )

            run_btn = gr.Button("Run", variant="primary")

        # ── Right column: outputs ─────────────────────────────────────────────
        with gr.Column():
            aif_video_out = gr.Video(
                label="Stage 1 Output — All-in-Focus (AIF) Video",
                interactive=False,
            )
            refocused_video_out = gr.Video(
                label="Stage 2 Output — Refocused Video",
                interactive=False,
            )

    run_btn.click(
        fn=predict,
        inputs=[
            input_video,
            focal_plane_slider,
            coc_slider,
        ],
        outputs=[aif_video_out, refocused_video_out],
    )

    gr.Examples(
        examples=[],   # add ["path/to/sample.mp4", 0.5, 27.0, 1.5, 0.02, 1.5, 0.02] later
        inputs=[
            input_video,
            focal_plane_slider,
            coc_slider,
        ],
    )


if __name__ == "__main__":
    demo.launch()