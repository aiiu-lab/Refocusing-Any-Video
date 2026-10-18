"""Download only the Stable Video Diffusion files used by RAV inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "stabilityai/stable-video-diffusion-img2vid"
DEFAULT_LOCAL_DIR = Path(__file__).resolve().parents[2] / "checkpoints" / "svdh"
METADATA_FILES = (
    "model_index.json",
    "feature_extractor/preprocessor_config.json",
    "scheduler/scheduler_config.json",
    "image_encoder/config.json",
    "unet/config.json",
    "vae/config.json",
    "LICENSE.md",
    "README.md",
)
WEIGHT_FILES = (
    # The image encoder loads its default variant; the VAE/UNet request fp16.
    "image_encoder/model.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--revision", default="main", help="Hub branch, tag, or commit.")
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Check access and download small configs/license files only; weights are omitted.",
    )
    args = parser.parse_args()
    required = METADATA_FILES if args.metadata_only else METADATA_FILES + WEIGHT_FILES
    destination = args.local_dir.expanduser().resolve()

    snapshot_download(
        repo_id=REPO_ID,
        revision=args.revision,
        local_dir=str(destination),
        allow_patterns=list(required),
    )
    missing = [name for name in required if not (destination / name).is_file()]
    if missing:
        raise SystemExit("Download incomplete; missing files: " + ", ".join(missing))
    if args.metadata_only:
        print(f"SVD metadata downloaded to {destination}; model weights are not included.")
    else:
        print(f"SVD inference files downloaded to {destination} (about 5.8 GB).")


if __name__ == "__main__":
    main()
