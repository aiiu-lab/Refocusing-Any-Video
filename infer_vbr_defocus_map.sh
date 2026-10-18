#!/usr/bin/env bash
# Clean launcher for VBR defocus-map refocusing inference.
#
# Common use:
#   bash infer_vbr_defocus_map.sh
#
# Useful switches:
#   DRY_RUN=1 bash infer_vbr_defocus_map.sh
#   FP16=1 bash infer_vbr_defocus_map.sh
#   SKIP_DEBUG_OUTPUTS=1 bash infer_vbr_defocus_map.sh
#   INPUT_VIDEO=... DEPTH_INPUT=... FOCAL_PLANE=... OUTPUT_DIR=... bash infer_vbr_defocus_map.sh

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$repo_root"

# -----------------------------------------------------------------------------
# Edit this block for the run
# -----------------------------------------------------------------------------

python_bin="${PYTHON_BIN:-python3}"
script="${RAV_INFERENCE_SCRIPT:-inference/vbr_defocus_map.py}"

checkpoint="${CHECKPOINT:-./checkpoints/vbr_weights/checkpoint-100000}"
pretrained_model="${PRETRAINED_MODEL:-./checkpoints/svdh}"

input="${INPUT_VIDEO:-./inputs/aif.mp4}"
depth_input="${DEPTH_INPUT:-./inputs/depth.mp4}"
focal_plane="${FOCAL_PLANE:-0.5}"
output_dir="${OUTPUT_DIR:-./results/vbr}"

batch_size="${BATCH_SIZE:-1}"
num_frames="${NUM_FRAMES:-14}"
height="${HEIGHT:-}"       # leave empty to use input height rounded down to /8
width="${WIDTH:-}"         # leave empty to use input width rounded down to /8
coc="${COC:-27.0}"
output_fps="${OUTPUT_FPS:-24}"
num_inference_steps="${NUM_INFERENCE_STEPS:-25}"
guidance_scale="${GUIDANCE_SCALE:-1.5}"
reconstruction_guidance="${RECONSTRUCTION_GUIDANCE:-0.0}"
noise_aug_strength="${NOISE_AUG_STRENGTH:-0.02}"
seed="${SEED:-42}"
device="${DEVICE:-cuda}"

chunk_frames="${CHUNK_FRAMES:-1}"
fp16="${FP16:-0}"
dry_run="${DRY_RUN:-0}"
skip_debug_outputs="${SKIP_DEBUG_OUTPUTS:-0}"
allow_focal_plane_out_of_range="${ALLOW_FOCAL_PLANE_OUT_OF_RANGE:-0}"

# -----------------------------------------------------------------------------
# Build command
# -----------------------------------------------------------------------------

args=(
  --checkpoint "$checkpoint"
  --pretrained_model "$pretrained_model"
  --input "$input"
  --depth_input "$depth_input"
  --focal_plane "$focal_plane"
  --output_dir "$output_dir"
  --batch_size "$batch_size"
  --num_frames "$num_frames"
  --coc "$coc"
  --output_fps "$output_fps"
  --num_inference_steps "$num_inference_steps"
  --guidance_scale "$guidance_scale"
  --reconstruction_guidance "$reconstruction_guidance"
  --noise_aug_strength "$noise_aug_strength"
  --seed "$seed"
  --device "$device"
)

if [[ -n "$height" ]]; then
  args+=(--height "$height")
fi
if [[ -n "$width" ]]; then
  args+=(--width "$width")
fi
if [[ "$chunk_frames" == "1" ]]; then
  args+=(--chunk_frames)
fi
if [[ "$fp16" == "1" ]]; then
  args+=(--fp16)
fi
if [[ "$dry_run" == "1" ]]; then
  args+=(--dry_run)
fi
if [[ "$skip_debug_outputs" == "1" ]]; then
  args+=(--skip_debug_outputs)
fi
if [[ "$allow_focal_plane_out_of_range" == "1" ]]; then
  args+=(--allow_focal_plane_out_of_range)
fi

cmd=("$python_bin" "$script" "${args[@]}" "$@")

printf 'Running VBR defocus-map command:\n'
printf '  %q' "${cmd[@]}"
printf '\n\n'
exec "${cmd[@]}"
