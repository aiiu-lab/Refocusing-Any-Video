# Inference guide

Prepare the [environment](dependencies.md) and [model assets](../checkpoints/README.md) first. Run Python commands from the repository root. The shell launchers select their repository directory automatically; relative input/output overrides are resolved there.

## VDD: all-in-focus recovery

```bash
INPUT_VIDEO=./inputs/input.mp4 OUTPUT_DIR=./results/vdd \
  FP16=1 bash infer_vdd.sh
```

Use `DRY_RUN=1` first to read/preprocess inputs and print the resolved setup before loading models. This is an input-validation run, and still imports the inference packages.

The implementation also requires the ArcFace/FaRL/MTCNN dependencies in baseline mode because its shared model wrapper constructs them unconditionally. Full VDD expects CUDA.

## Identity Adapter

For one subject:

```bash
MODE=single INPUT_VIDEO=./inputs/input.mp4 \
  REFERENCE_IMAGE=./inputs/reference_face.jpg \
  IP_ADAPTER_PATH=./checkpoints/identity_adapter/ip_adapter.bin \
  HEIGHT=512 WIDTH=896 FP16=1 bash infer_vdd_identity.sh
```

Optional keypoint control:

```bash
python tools/preprocessing/face_keypoints.py \
  --input-video ./inputs/input.mp4 --output-folder ./inputs/kp_video \
  --mode single --height 512 --width 896

MODE=single REFERENCE_IMAGE=./inputs/reference_face.jpg \
  KEYPOINT_VIDEO=./inputs/kp_video/input.mp4 HEIGHT=512 WIDTH=896 \
  FP16=1 bash infer_vdd_identity.sh
```

The launcher uses `./inputs/keypoints.mp4` automatically if that file exists. `KEYPOINT_VIDEO=""` disables this default. With multiple subjects, pass ordered arrays on the command line:

```bash
MODE=multi HEIGHT=512 WIDTH=896 FP16=1 bash infer_vdd_identity.sh \
  --reference_images ./inputs/person_a.jpg ./inputs/person_b.jpg \
  --keypoint_videos ./inputs/kp_a.mp4 ./inputs/kp_b.mp4
```

Keypoint videos are optional, but when supplied their count and subject order must match the reference images. All input/reference paths are examples supplied by the user; no sample videos or faces are bundled.

Use `infer_vdd_identity_track4gen.sh` with `checkpoints/identity_adapter_track4gen/ip_adapter.bin` for a Track4Gen-trained adapter. Its controls are the same. Its checkpoint must contain the trained refiner weights.

`MODE=vdd` (historical alias `MODE=deblur`) disables identity conditioning on either identity launcher. `infer_vdd.sh` is a convenience wrapper for that baseline mode.

## VBR: refocusing

Estimate depth with `VideoDepthAnything/infer_video_depth.py`, then provide aligned RGB and depth with identical frame counts:

```bash
INPUT_VIDEO=./results/vdd/output.mp4 DEPTH_INPUT=./inputs/depth.mp4 \
  FOCAL_PLANE=0.5 COC=27 FP16=1 bash infer_vbr_defocus_map.sh
```

Depth is read as an 8-bit map normalized to `[0, 1]`. Focal-plane values use the same scale. A scalar holds the plane constant; a text file supplies one value per frame (a one-value file is also repeated). Generate a smooth sequence with:

```bash
python tools/preprocessing/focal_plane.py \
  --frames 64 --x 1 62 --y 0.78 0.05 --output ./inputs/focal_plane.txt
```

Then use `FOCAL_PLANE=./inputs/focal_plane.txt`. The default expected interval is `[0, 1]`. Higher `COC` increases defocus strength. `SKIP_DEBUG_OUTPUTS=1` suppresses the heatmap video and focal-plane plot.

## Common options

| Environment variable | Purpose |
| --- | --- |
| `CHECKPOINT` | VDD/VBR model checkpoint directory |
| `IP_ADAPTER_PATH` | Trained identity/refiner adapter file |
| `PRETRAINED_MODEL` | SVD base directory, default `./checkpoints/svdh` |
| `INPUT_VIDEO`, `OUTPUT_DIR` | Input video/frame folder and output directory |
| `HEIGHT`, `WIDTH` | Inference dimensions, rounded down to multiples of eight |
| `FP16=1` | Lower model memory use |
| `DRY_RUN=1` | Validate inputs before model loading |
| `NUM_INFERENCE_STEPS` | Diffusion steps, default 25 |
| `OUTPUT_FPS` | Saved video FPS, default 24 |
| `NUM_FRAMES` | Temporal window; keep 14 with long-video chunking |
| `CHUNK_FRAMES=1` | Long-video scheduling, default on |
| `CHUNK_FRAMES=0` | Disable chunking for short clips/debugging |
| `SKIP_STAGE1=1` | Skip the anchor pass in VDD |
| `BATCH_SIZE` | Number of scheduled chunks per batch |
| `DEVICE` | Runtime device, usually CUDA |
| `REFERENCE_IMAGE`, `KEYPOINT_VIDEO` | Single-subject input overrides |
| `DEPTH_INPUT`, `FOCAL_PLANE`, `COC` | VBR conditioning |

The scheduler uses a 14-frame window and an 8-frame stride. Baseline VDD preserves input dimensions by default (rounded down to multiples of eight). Identity launcher image-size defaults differ between variants; set `HEIGHT` and `WIDTH` explicitly for comparable runs. Set either dimension to an empty string for automatic sizing on an identity launcher. Additional Python arguments are forwarded:

```bash
FP16=1 bash infer_vdd.sh --decode_chunk_size 4
```

Direct Python entrypoints are `inference/vdd_identity_multi.py`, `inference/vdd_identity_track4gen_multi.py`, and `inference/vbr_defocus_map.py`; run with `--help` for their full option sets.

## Outputs and troubleshooting

Each inference launcher writes `OUTPUT_DIR/output.mp4` and individual PNG frames under `OUTPUT_DIR/frames/`. VBR optionally adds `defocus_heatmap_cinematic.mp4` and `focal_plane_plot.png`. Default outputs are under `results/`, which Git ignores.

For a memory error, enable fp16, reduce image size/batch size, or pass `--decode_chunk_size 4`. For a frame-count mismatch, align the RGB/depth streams and focal-plane sequence. Missing CLIP or face checkpoints can affect baseline VDD too; see [the dependency guide](dependencies.md).
