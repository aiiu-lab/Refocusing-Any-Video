"""Generate a per-frame focal-plane text file for refocus inference.

The output format is intentionally simple: one focal-plane value per line.
That file can be passed to ``tools/preprocessing/create_bokeh_video.py`` with ``--zero-depth-file``.

Key-frame convention
--------------------
``--x`` values are 1-based frame indices. ``--y`` values are the normalized
focal-plane depths at those frames, usually in [0, 1]. Values between key frames
are linearly interpolated. Frames before the first key frame hold the first
value; frames after the last key frame hold the last value.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_OUTPUT = "output_focal_plane.txt"

EXAMPLES = """
Examples:
  # Current simple ramp preset from the old script: 64 frames, 0.78 -> 0.05
  python3 tools/preprocessing/focal_plane.py --frames 64 --x 1 62 --y 0.78 0.05 --output output_focal_plane.txt

  # Same inputs, comma-separated
  python3 tools/preprocessing/focal_plane.py -n 64 --x 1,62 --y 0.78,0.05

  # Use the generated txt for refocus inference
  python3 tools/preprocessing/create_bokeh_video.py ... --zero-depth-file output_focal_plane.txt
"""


def parse_number_sequence(values: Sequence[str], value_name: str, cast) -> list:
    """Parse either space-separated or comma-separated CLI values."""
    tokens: list[str] = []
    for value in values:
        tokens.extend(part.strip() for part in value.split(","))
    tokens = [token for token in tokens if token]
    if not tokens:
        raise ValueError(f"No values provided for {value_name}.")

    try:
        return [cast(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"Invalid {value_name} value in: {values}") from exc


def validate_key_frames(
    num_frames: int,
    key_frames: Sequence[int],
    focal_planes: Sequence[float],
    allow_value_out_of_range: bool = False,
) -> None:
    """Validate interpolation inputs before writing an output file."""
    if num_frames <= 0:
        raise ValueError("--frames must be a positive integer.")
    if len(key_frames) != len(focal_planes):
        raise ValueError("--x and --y must contain the same number of values.")
    if len(key_frames) < 1:
        raise ValueError("At least one key frame and focal-plane value is required.")

    for frame_idx in key_frames:
        if frame_idx < 1 or frame_idx > num_frames:
            raise ValueError(f"Key frame {frame_idx} is outside the valid range [1, {num_frames}].")

    for prev_frame, next_frame in zip(key_frames, key_frames[1:]):
        if next_frame <= prev_frame:
            raise ValueError("--x frame indices must be strictly increasing.")

    if not allow_value_out_of_range:
        for value in focal_planes:
            if value < 0.0 or value > 1.0:
                raise ValueError(
                    f"Focal-plane value {value} is outside [0, 1]. "
                    "Pass --allow-value-out-of-range if this is intentional."
                )


def generate_interpolated_values(
    num_frames: int,
    key_frames: Sequence[int],
    focal_planes: Sequence[float],
    allow_value_out_of_range: bool = False,
) -> np.ndarray:
    """Return one focal-plane value per frame using linear interpolation.

    Parameters
    ----------
    num_frames:
        Total number of output frames/lines.
    key_frames:
        1-based frame indices where focal-plane values are specified.
    focal_planes:
        Focal-plane values at ``key_frames``. In normal refocus usage these are
        normalized depth values in [0, 1].
    allow_value_out_of_range:
        Disable the [0, 1] value check for experiments that use another scale.
    """
    validate_key_frames(num_frames, key_frames, focal_planes, allow_value_out_of_range)

    x_full = np.arange(1, num_frames + 1, dtype=np.float32)
    key_frames_array = np.asarray(key_frames, dtype=np.float32)
    focal_planes_array = np.asarray(focal_planes, dtype=np.float32)
    return np.interp(x_full, key_frames_array, focal_planes_array).astype(np.float32)


def write_focal_plane_file(output_path: str | Path, values: Sequence[float], precision: int = 4) -> None:
    """Write one focal-plane value per line."""
    if precision < 0:
        raise ValueError("--precision must be non-negative.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = f"{{:.{precision}f}}\n"
    with output_path.open("w", encoding="utf-8") as file:
        for value in values:
            file.write(fmt.format(float(value)))


def generate_interpolated_file(
    output_txt_name: str | Path,
    N: int,
    x: Sequence[int],
    y: Sequence[float],
    precision: int = 4,
    allow_value_out_of_range: bool = False,
) -> np.ndarray:
    """Backward-compatible helper used by the old script.

    ``N`` is the number of frames, ``x`` is a list of 1-based key-frame indices,
    and ``y`` is the corresponding list of focal-plane values. The generated
    file contains exactly ``N`` lines and can be used as ``--zero-depth-file``.
    """
    values = generate_interpolated_values(N, x, y, allow_value_out_of_range)
    write_focal_plane_file(output_txt_name, values, precision)
    print(f"Successfully wrote {len(values)} lines to {output_txt_name}")
    return values


def print_summary(
    output_path: str | Path,
    values: np.ndarray,
    key_frames: Sequence[int],
    focal_planes: Sequence[float],
) -> None:
    """Print a compact summary that helps catch wrong frame counts or ranges."""
    print(f"Successfully wrote {len(values)} lines to {output_path}")
    print(f"Key frames: {list(key_frames)}")
    print(f"Key values: {[float(value) for value in focal_planes]}")
    print(f"Output range: {float(values.min()):.4f} to {float(values.max()):.4f}")
    print(f"First value: {float(values[0]):.4f}; last value: {float(values[-1]):.4f}")

    if key_frames[0] != 1:
        print(f"Note: frames 1-{key_frames[0] - 1} hold the first value {float(focal_planes[0]):.4f}.")
    if key_frames[-1] != len(values):
        print(
            f"Note: frames {key_frames[-1] + 1}-{len(values)} hold "
            f"the last value {float(focal_planes[-1]):.4f}."
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a per-frame focal-plane txt file for refocus preprocessing.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--frames", "-n", type=int, required=True, help="Number of output frames/lines.")
    parser.add_argument(
        "--x",
        "--key-frames",
        nargs="+",
        required=True,
        help="1-based key-frame indices, e.g. '1 62' or '1,62'.",
    )
    parser.add_argument(
        "--y",
        "--focal-planes",
        nargs="+",
        required=True,
        help="Focal-plane values at each key frame, e.g. '0.78 0.05'.",
    )
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output txt path.")
    parser.add_argument("--precision", type=int, default=4, help="Decimal places to write per value.")
    parser.add_argument(
        "--allow-value-out-of-range",
        action="store_true",
        help="Allow focal-plane values outside [0, 1].",
    )
    parser.add_argument("--quiet", action="store_true", help="Only write the file; do not print the summary.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    key_frames = parse_number_sequence(args.x, "--x", int)
    focal_planes = parse_number_sequence(args.y, "--y", float)

    values = generate_interpolated_values(
        args.frames,
        key_frames,
        focal_planes,
        allow_value_out_of_range=args.allow_value_out_of_range,
    )
    write_focal_plane_file(args.output, values, args.precision)

    if not args.quiet:
        print_summary(args.output, values, key_frames, focal_planes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
