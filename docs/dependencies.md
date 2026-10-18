# Environment setup

Run from the cloned repository root on Linux with Conda and an NVIDIA GPU.
Use an NVIDIA driver compatible with the CUDA 12.4 PyTorch wheels; a separate
CUDA toolkit is not required for the default inference path.

```bash
conda env create -f setup/environment.yaml
conda activate refocus
python -m pip check
```

The environment includes Python 3.10, PyTorch 2.6, Diffusers 0.32, Transformers
4.50, PEFT 0.15, OpenAI CLIP, Video Depth Anything dependencies, and the Gradio
demo. CLIP is pinned to a Git commit. Use the supplied versions: the pipelines
and checkpoint loaders depend on these library APIs. Only the headless OpenCV
package is installed; desktop OpenCV windows are not used.

## Verify the installation

```bash
python - <<'PY'
import torch, cv2, clip, easydict
from rav.models.ipadapter_kp import SVDIPAdapterWrapper
from inference import ipadapter_multi, refocus_depth
assert torch.cuda.is_available(), "CUDA is unavailable; check your NVIDIA driver and PyTorch installation."
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY
ffmpeg -version
```

Then follow the [checkpoint setup guide](../checkpoints/README.md) and
[README inference commands](../README.md#inference).

## Optional workflows

- **Identity guidance:** uses ArcFace, FaRL, MTCNN, and CLIP. Follow the
  [checkpoint guide](../checkpoints/README.md) for their weights. FaRL's first
  use also downloads OpenAI CLIP ViT-B/16 to `~/.cache/clip/`.
- **Face keypoint preprocessing:** requires `insightface` and
  `onnxruntime-gpu` (or `onnxruntime` for CPU), plus the AntelopeV2 models.
- **Track4Gen landmark preprocessing:** requires `mediapipe` and its face
  landmarker model. It is not needed for the standard inference commands.
- **Bokeh synthesis:** requires a CuPy package matching your CUDA version and
  the BokehMe weights. It is not needed for inference on an existing video.
- **Evaluation tools:** require `torchmetrics` and `torch-fidelity` in addition
  to the inference environment.
- **Gradio demo:** dependencies are included; run `FP16=1 python app.py`
  after preparing the checkpoints.

Training code and its dependencies will be released later.
