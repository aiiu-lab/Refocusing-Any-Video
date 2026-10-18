# Checkpoint setup

Run these commands from the cloned repository root with the `refocus` environment active. Model weights are hosted separately from the source code.

## 1. Project checkpoints

Download the model folders from [nickchiu/Refocusing-Any-Video on Hugging Face](https://huggingface.co/nickchiu/Refocusing-Any-Video), preserving their folders inside `checkpoints/`. You can download the complete bundle with:

```bash
python tools/checkpoints/transfer_checkpoints.py download \
  --repo-id nickchiu/Refocusing-Any-Video --local-dir ./checkpoints
```

The download helper preserves this setup guide. The bundle includes VDD, VBR, both identity adapters, and MTCNN face detector weights:

```text
checkpoints/
  vdd_weights/checkpoint-90000/unet_lora/
  vbr_weights/checkpoint-100000/
    unet_lora/
    unet_conv_in_refocus.pt
  identity_adapter/ip_adapter.bin
  identity_adapter_track4gen/ip_adapter.bin
  facenet/{pnet,rnet,onet}.pt
```

Keep `unet_conv_in_refocus.pt` alongside the VBR LoRA folder; VBR needs both. For identity guidance, use the adapter matching your launcher.

## 2. Base and auxiliary models

The following models are downloaded separately under their providers' terms. **Baseline VDD also requires the face models** because its shared wrapper initializes them even when identity guidance is disabled.

| Model | Source | Save under `checkpoints/` |
| --- | --- | --- |
| SVD, 14-frame img2vid | [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) | `svdh/` |
| Video Depth Anything, relative-depth Large | [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) | `video_depth_anything_vitl.pth` |
| FaRL ViT-B/16, ep16 | [FaRL](https://github.com/FacePerceiver/FaRL#pre-trained-backbones) | `farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth` |
| ArcFace IR-SE50 | [pSp pretrained-model instructions](https://github.com/eladrich/pixel2style2pixel#pretrained-models) | `arcface/model_ir_se50.pth` |

Download SVD, depth, and FaRL:

```bash
python tools/checkpoints/download_svd_weights.py

mkdir -p checkpoints/farl checkpoints/arcface
curl -fL \
  https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth \
  -o checkpoints/video_depth_anything_vitl.pth
curl -fL \
  https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep16.pth \
  -o checkpoints/farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth
```

The SVD helper downloads only the required configs, image encoder, fp16 UNet/VAE, and license files (about 5.8 GB). Keep the fp16 UNet/VAE files even when running with `FP16=0`. If a provider requires authentication, run `python -c 'from huggingface_hub import login; login()'` first.

For ArcFace, open the [official IR-SE50 download](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing), complete Google Drive's confirmation if shown, and save the actual PyTorch checkpoint as `checkpoints/arcface/model_ir_se50.pth`. An HTML download page or an InsightFace ONNX file cannot replace it.

FaRL also initializes OpenAI CLIP ViT-B/16. Download its cache before inference:

```bash
python -c 'import clip; clip.load("ViT-B/16", device="cpu")'
```

## 3. Verify files

```bash
python - <<'CHECK'
from pathlib import Path
required = [
    "svdh/model_index.json", "svdh/image_encoder/config.json",
    "svdh/unet/config.json", "svdh/vae/config.json",
    "svdh/feature_extractor/preprocessor_config.json",
    "svdh/scheduler/scheduler_config.json",
    "svdh/image_encoder/model.safetensors",
    "svdh/unet/diffusion_pytorch_model.fp16.safetensors",
    "svdh/vae/diffusion_pytorch_model.fp16.safetensors",
    "vdd_weights/checkpoint-90000/unet_lora/adapter_config.json",
    "vdd_weights/checkpoint-90000/unet_lora/adapter_model.safetensors",
    "vbr_weights/checkpoint-100000/unet_lora/adapter_config.json",
    "vbr_weights/checkpoint-100000/unet_lora/adapter_model.safetensors",
    "vbr_weights/checkpoint-100000/unet_conv_in_refocus.pt",
    "facenet/pnet.pt", "facenet/rnet.pt", "facenet/onet.pt",
    "arcface/model_ir_se50.pth",
    "farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth",
    "video_depth_anything_vitl.pth",
]
missing = [name for name in required if not (Path("checkpoints") / name).is_file()]
if missing:
    raise SystemExit("Missing checkpoints:\n" + "\n".join(missing))
print("Required checkpoint files found.")
CHECK
```

Then follow the [README inference example](../README.md#inference). For custom paths, see the [inference guide](../docs/inference_guide.md).
