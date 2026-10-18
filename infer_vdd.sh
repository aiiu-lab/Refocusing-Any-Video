#!/usr/bin/env bash
# Run VDD without identity conditioning.
# Override INPUT_VIDEO, OUTPUT_DIR, CHECKPOINT, PYTHON_BIN, and other options as
# in infer_vdd_identity.sh. Additional command-line arguments are passed through.
# Identity-conditioned runs use infer_vdd_identity.sh instead.

set -euo pipefail

case "${MODE:-vdd}" in
  vdd|deblur) ;;
  *)
    echo "Unsupported MODE='${MODE}' for infer_vdd.sh. Use vdd or deblur; use infer_vdd_identity.sh for identity modes." >&2
    exit 2
    ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$repo_root"
export MODE=vdd
export OUTPUT_DIR="${OUTPUT_DIR:-./results/vdd}"
# Baseline VDD follows the input size unless dimensions are explicitly supplied.
export HEIGHT="${HEIGHT-}"
export WIDTH="${WIDTH-}"
exec bash "$repo_root/infer_vdd_identity.sh" "$@"
