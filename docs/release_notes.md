# Source distribution notes

This distribution was prepared from the current working files, including uncommitted implementation changes. The original directory and Git history were preserved in place. A separate local cleanup audit records source paths, omissions, hashes, and checkpoint staging; that machine-specific inventory is outside the GitHub source directory.

## Included

- Inference entrypoints, diffusion pipelines, and model/face/runtime definitions under `rav/`.
- Depth estimation, reusable preprocessing/evaluation tools, optional bokeh renderers, and the optional Gradio demo.
- Compatibility wrappers needed by current imports, vendor runtime code, and existing license notices.
- Restored root Track4Gen launcher, portable launcher defaults, repository-relative dataset/model paths, and configurable Hugging Face checkpoint transfer.

## Excluded

- Training entrypoints, runners, datasets, losses, configurations, split lists, validation/debug helpers, and dataset-construction scripts. The current cleaned versions are preserved in a sibling directory outside this release for a future training release.

- Model weights and optimizer/random-state snapshots, including vendored MTCNN weights.
- Experiment outputs, frame dumps, example videos/images, presentation renders, and generated landmark CSVs.
- Original/nested Git histories, bytecode/cache folders, local editor settings, personal notes, duplicate test launchers, unused archive snapshots, and vendor example/test media.
- The legacy downloader targeting a different historical project checkpoint.

Core numerical model implementations are preserved. Inference-required definitions previously kept under `training/` now live under `rav/`, with import paths updated. Optional bokeh preprocessing imports only the renderer instead of the training dataset. Mechanical fixes include external MTCNN weight lookup, a broken unused VDA streaming utility import, and launcher path/argument handling.

## Validation

On October 18, 2026, the documented baseline workflow passed in a fresh environment
created from `setup/environment.yaml`: dependency checks, CUDA/import checks, project
checkpoint download, SVD setup, CLIP initialization, file preflight, depth estimation,
VDD, and VBR. Downloads preserve the checkpoint setup guide.

The GPU run used an NVIDIA RTX A6000, PyTorch 2.6.0/CUDA 12.4, fp16 diffusion,
and an 18-frame 512×288 video at 24 FPS. Both diffusion stages used the default
25 steps. VDD exercised anchor inference, overlapping windows, tail padding, and
blending; VBR exercised overlapping windows and tail padding. Both produced
18 PNG frames and readable, nonblank 18-frame output videos. Freshly downloaded
project, SVD, depth, and face checkpoints were used; CLIP reused its local cache.

This checks execution for that input and hardware. It does not establish perceptual
quality, a minimum VRAM requirement, or compatibility with every video. Optional
identity conditioning, preprocessing, and Gradio model execution were not included
in this baseline GPU run. Training code is withheld for a future release.

The inference bundle is available at [Hugging Face](https://huggingface.co/nickchiu/Refocusing-Any-Video).
All ten checkpoint/configuration artifacts were downloaded and verified against
the staged SHA-256 checksums. Detailed logs and inventories are stored outside
the source release in the maintainer's validation and cleanup-audit directories.
No code or model-card changes were pushed remotely during this revision.
