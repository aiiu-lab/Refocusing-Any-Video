# Refocusing Any Video: Temporally Consistent Video Refocusing for Dynamic Focal Planes

**SIGGRAPH Asia 2026**

Chee Kim Gan, [Pin-Yen Chiu](https://itsnickchiu.github.io/), I-Sheng Fang, Jun-Cheng Chen  
Research Center for Information Technology Innovation, Academia Sinica

Refocusing Any Video (RAV) recovers an all-in-focus video with **Video Defocus Deblurring (VDD)**, then uses **Video Bokeh Rendering (VBR)** to refocus it at a chosen focal plane. Optional identity adapters guide face restoration using reference images.

[Models](https://huggingface.co/nickchiu/Refocusing-Any-Video) · [Inference guide](docs/inference_guide.md)

## Release status

- [x] Inference code and pretrained checkpoints
- [ ] Training code — being prepared for a future release

## Environment

Run all commands from the cloned repository root. Use Linux, an NVIDIA GPU with a CUDA-compatible driver, and Conda.

```bash
conda env create -f setup/environment.yaml
conda activate refocus
```

See [environment checks](docs/dependencies.md) if setup fails.

## Checkpoints

Download the checkpoints from [our Hugging Face repository](https://huggingface.co/nickchiu/Refocusing-Any-Video) into `checkpoints/`, preserving the directory structure:

```bash
python tools/checkpoints/transfer_checkpoints.py download \
  --repo-id nickchiu/Refocusing-Any-Video --local-dir ./checkpoints
```

Also download the SVD base, depth, and face models following the [checkpoint setup guide](checkpoints/README.md). Complete its file check before inference.

## Inference

Place your video at `inputs/input.mp4`, then run:

```bash
# 1. Estimate depth.
python VideoDepthAnything/infer_video_depth.py \
  --input ./inputs/input.mp4 --output ./inputs/depth.mp4

# 2. Recover the all-in-focus video.
INPUT_VIDEO=./inputs/input.mp4 OUTPUT_DIR=./results/vdd \
  FP16=1 bash infer_vdd.sh

# 3. Refocus the video.
INPUT_VIDEO=./results/vdd/output.mp4 DEPTH_INPUT=./inputs/depth.mp4 \
FOCAL_PLANE=0.5 COC=27 OUTPUT_DIR=./results/vbr \
  FP16=1 bash infer_vbr_defocus_map.sh
```

The final video is **`results/vbr/output.mp4`**. Both VDD and VBR also save PNG frames in their output directory's `frames/` folder.

| Option | Purpose |
| --- | --- |
| `FOCAL_PLANE` | Focus disparity: `0` is far, `1` is near. A text file can specify one value per frame. |
| `COC` | Bokeh strength; larger values increase defocus. |
| `HEIGHT`, `WIDTH` | Set inference resolution; preserve the input aspect ratio. Baseline VDD/VBR use the input size by default. |
| `OUTPUT_FPS` | Integer output playback rate, default `24`; frames are not resampled and audio is omitted. |

Keep RGB and depth frame counts, order, and framing aligned. For lower memory use, reduce resolution or append `--decode_chunk_size 4` to the launcher. See the [inference guide](docs/inference_guide.md) for identity guidance, Track4Gen, moving focus, and the optional Gradio demo.

## Acknowledgments

We use Stable Video Diffusion (SVD), Video Depth Anything, FaRL, ArcFace, OpenAI CLIP, and facenet-pytorch. See [third-party notices](THIRD_PARTY_NOTICES.md) and [model licensing](https://huggingface.co/nickchiu/Refocusing-Any-Video#models-and-licenses).

## Citation

If you use this work, please cite:

```bibtex
@misc{gan2026refocusinganyvideo,
  title  = {Refocusing Any Video: Temporally Consistent Video Refocusing for Dynamic Focal Planes},
  author = {Gan, Chee Kim and Chiu, Pin-Yen and Fang, I-Sheng and Chen, Jun-Cheng},
  year   = {2026},
  note   = {SIGGRAPH Asia 2026},
  url    = {https://github.com/aiiu-lab/Refocusing-Any-Video}
}
```
