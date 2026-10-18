#!/usr/bin/env bash
# Clean launcher for VDD + Identity Adapter inference without the Track4Gen Refiner.
#
# Common use:
#   bash infer_vdd_identity.sh
#
# Useful switches:
#   MODE=single bash infer_vdd_identity.sh
#   MODE=vdd bash infer_vdd_identity.sh
#   DRY_RUN=1 bash infer_vdd_identity.sh
#   FP16=1 bash infer_vdd_identity.sh

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$repo_root"

# -----------------------------------------------------------------------------
# Edit this block for the run
# -----------------------------------------------------------------------------

mode="${MODE:-multi}"        # multi | single | vdd (deblur alias)
python_bin="${PYTHON_BIN:-python3}"
script="${RAV_INFERENCE_SCRIPT:-inference/vdd_identity_multi.py}"

checkpoint="${CHECKPOINT:-./checkpoints/vdd_weights/checkpoint-90000}"
ip_adapter_path="${IP_ADAPTER_PATH:-./checkpoints/identity_adapter/ip_adapter.bin}"
pretrained_model="${PRETRAINED_MODEL:-./checkpoints/svdh}"

input="${INPUT_VIDEO:-./inputs/input.mp4}"
output_dir="${OUTPUT_DIR:-./results/vdd_identity}"

# For multiple subjects, pass --reference_images and --keypoint_videos on the CLI.
reference_images=("${REFERENCE_IMAGE:-./inputs/reference_face.jpg}")

# Use the example control video if present; KEYPOINT_VIDEO="" disables it.
keypoint_video="${KEYPOINT_VIDEO-./inputs/keypoints.mp4}"
keypoint_videos=()
if [[ -n "$keypoint_video" && ( -f "$keypoint_video" || -n "${KEYPOINT_VIDEO+x}" ) ]]; then
  keypoint_videos+=("$keypoint_video")
fi

batch_size="${BATCH_SIZE:-1}"
num_frames="${NUM_FRAMES:-14}"
# Unset dimensions use the identity defaults; empty dimensions follow the input.
height="${HEIGHT-512}"
width="${WIDTH-1216}"
output_fps="${OUTPUT_FPS:-24}"
num_inference_steps="${NUM_INFERENCE_STEPS:-25}"
guidance_scale="${GUIDANCE_SCALE:-1.5}"
reconstruction_guidance="${RECONSTRUCTION_GUIDANCE:-0.0}"
noise_aug_strength="${NOISE_AUG_STRENGTH:-0.02}"
seed="${SEED:-42}"
device="${DEVICE:-cuda}"

chunk_frames="${CHUNK_FRAMES:-1}"
skip_stage1="${SKIP_STAGE1:-0}"
fp16="${FP16:-0}"
dry_run="${DRY_RUN:-0}"
save_face_mask_debug="${SAVE_FACE_MASK_DEBUG:-0}"

# -----------------------------------------------------------------------------
# Build command
# -----------------------------------------------------------------------------

common_args=(
  --checkpoint "$checkpoint"
  --pretrained_model "$pretrained_model"
  --output_dir "$output_dir"
  --input "$input"
  --batch_size "$batch_size"
  --num_frames "$num_frames"
  --output_fps "$output_fps"
  --num_inference_steps "$num_inference_steps"
  --guidance_scale "$guidance_scale"
  --reconstruction_guidance "$reconstruction_guidance"
  --noise_aug_strength "$noise_aug_strength"
  --seed "$seed"
  --device "$device"
)

if [[ -n "$height" ]]; then
  common_args+=(--height "$height")
fi
if [[ -n "$width" ]]; then
  common_args+=(--width "$width")
fi

if [[ "$mode" != "deblur" && "$mode" != "vdd" && -n "$ip_adapter_path" ]]; then
  common_args+=(--ip_adapter_path "$ip_adapter_path")
fi
if [[ "$chunk_frames" == "1" ]]; then
  common_args+=(--chunk_frames)
fi
if [[ "$skip_stage1" == "1" ]]; then
  common_args+=(--skip_stage1)
fi
if [[ "$fp16" == "1" ]]; then
  common_args+=(--fp16)
fi
if [[ "$dry_run" == "1" ]]; then
  common_args+=(--dry_run)
fi
if [[ "$save_face_mask_debug" == "1" ]]; then
  common_args+=(--save_face_mask_debug)
fi

case "$mode" in
  multi)
    subject_args=(--reference_images "${reference_images[@]}")
    if (( ${#keypoint_videos[@]} > 0 )); then
      subject_args+=(--keypoint_videos "${keypoint_videos[@]}")
    fi
    ;;
  single)
    subject_args=(--reference_image "${reference_images[0]}")
    if (( ${#keypoint_videos[@]} > 0 )); then
      subject_args+=(--keypoint_video "${keypoint_videos[0]}")
    fi
    ;;
  deblur|vdd)
    subject_args=(--disable_identity_adapter)
    ;;
  *)
    echo "Unknown MODE='$mode'. Use: multi, single, vdd, or deblur." >&2
    exit 2
    ;;
esac

cmd=("$python_bin" "$script" "${common_args[@]}" "${subject_args[@]}" "$@")

printf 'Running VDD command (MODE=%s):\n' "$mode"
printf '  %q' "${cmd[@]}"
printf '\n\n'
exec "${cmd[@]}"
