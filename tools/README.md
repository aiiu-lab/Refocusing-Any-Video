# Tools

Run from the repository root after installing the dependencies for the selected workflow. All inputs and weights are supplied separately; output directories are local and ignored by Git.

| Directory | Purpose |
| --- | --- |
| `preprocessing/` | synthetic bokeh, face control videos, focal planes, facial landmarks, focus-map visualization |
| `evaluation/` | metrics and face/landmark figure helpers |
| `checkpoints/` | separate Hugging Face model transfer and base SVD download |

## Preprocessing

```bash
python tools/preprocessing/focal_plane.py \
  --frames 64 --x 1 62 --y 0.78 0.05 --output ./inputs/focal_plane.txt

python tools/preprocessing/face_keypoints.py \
  --input-video ./inputs/input.mp4 --output-folder ./inputs/kp_video \
  --mode multi --height 512 --width 896 \
  --insightface-root ./checkpoints/insightface

python tools/preprocessing/create_bokeh_video.py \
  --input ./inputs/aif.mp4 --depth-input ./inputs/depth.mp4 \
  --output ./results/blurred.mp4 --focal-plane 0.5

python tools/preprocessing/visualize_focusmap.py \
  --depth-input ./inputs/depth.mp4 --focal-plane 0.5 \
  --output-dir ./results/focusmap

python tools/preprocessing/mediapipe_landmarks.py \
  --input ./inputs/input.mp4 --output-dir ./results/landmarks \
  --model-path ./checkpoints/mediapipe/face_landmarker.task
```

Training and dataset-construction scripts are not included in this inference release.

## Models and evaluation

[checkpoints/README.md](../checkpoints/README.md) describes Hugging Face upload/download and external model paths. `download_svd_weights.py` downloads the base SVD model; it does not download the project-trained VDD/VBR adapters.

`evaluation/compute_metrics.py` retains historical dataset-layout assumptions. Edit its configured result directory and expected input/ground-truth layout before use. The two figure helpers are optional analysis tools; review their CLI arguments for your own videos and landmarks.
