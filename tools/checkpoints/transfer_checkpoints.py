"""Transfer a staged RAV checkpoint tree to or from a Hugging Face model repo.

Stage the desired model folders separately from experiment outputs before
uploading. Authentication uses HF_TOKEN or the local Hugging Face login.
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatchcase
from glob import escape as escape_glob
import os
from pathlib import Path, PurePosixPath
import sys


DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"

# Model tensors, tokenizer data, and model configuration/documentation only.
# Hugging Face matches these patterns against full repository-relative paths.
ALLOW_PATTERNS = (
    "*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.onnx",
    "*.onnx_data", "*.msgpack", "*.h5", "*.hdf5", "*.pb", "*.tflite", "*.task",
    "*.model", "*.tiktoken", "*.json", "*.yaml", "*.yml", "*.txt",
    "README", "README.md", "*/README", "*/README.md",
    "LICENSE", "LICENSE.md", "*/LICENSE", "*/LICENSE.md",
    "LICENCE", "LICENCE.md", "*/LICENCE", "*/LICENCE.md",
    "NOTICE", "NOTICE.md", "*/NOTICE", "*/NOTICE.md",
    "COPYING", "COPYRIGHT", "*/COPYING", "*/COPYRIGHT",
)

_EXCLUDED_DIRECTORIES = (
    "cache", "caches", "__pycache__", "outputs*", "output", "results*",
    "inference_output*", "inference-output*", "infer_out*", "*-outputs",
    "wandb", "logs", "runs", "tmp", "temp", "datasets", "dataset",
    "videos", "video", "images", "image", "samples",
)
_EXCLUDED_FILES = (
    "token", "token.txt", "hf_token", "hf_token.txt", "credentials",
    "credentials.json", "credentials.txt",
)
IGNORE_PATTERNS = (
    ".*", "*/.*",
    *(
        pattern
        for directory in _EXCLUDED_DIRECTORIES
        for pattern in (f"{directory}/*", f"*/{directory}/*")
    ),
    *(
        pattern
        for filename in _EXCLUDED_FILES
        for pattern in (filename, f"*/{filename}")
    ),
)


def _excluded_directory(name: str) -> bool:
    return name.startswith(".") or any(
        fnmatchcase(name.lower(), pattern) for pattern in _EXCLUDED_DIRECTORIES
    )


def _selected_file(relative_path: str) -> bool:
    return (
        any(fnmatchcase(relative_path, pattern) for pattern in ALLOW_PATTERNS)
        and not any(fnmatchcase(relative_path, pattern) for pattern in IGNORE_PATTERNS)
    )


def collect_upload_files(local_dir: Path) -> tuple[list[tuple[str, int]], int]:
    """Select regular files without following any file or directory symlinks."""
    selected = []
    skipped = 0
    for directory, dirnames, filenames in os.walk(local_dir, followlinks=False):
        parent = Path(directory)
        kept_dirs = []
        for name in dirnames:
            if _excluded_directory(name) or (parent / name).is_symlink():
                skipped += 1
            else:
                kept_dirs.append(name)
        dirnames[:] = kept_dirs
        for name in filenames:
            path = parent / name
            relative_path = path.relative_to(local_dir).as_posix()
            if path.is_symlink() or not path.is_file() or not _selected_file(relative_path):
                skipped += 1
                continue
            selected.append((relative_path, path.stat().st_size))
    return sorted(selected), skipped


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _repo_path(value: str) -> str:
    if not value:
        return ""
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise argparse.ArgumentTypeError("use a relative repository folder without '..'")
    if any(_excluded_directory(part) for part in path.parts):
        raise argparse.ArgumentTypeError("the destination must not be a hidden/cache/output folder")
    return path.as_posix() if path.as_posix() != "." else ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("upload", "download"):
        subparser = commands.add_parser(command, help=f"{command.capitalize()} model assets")
        subparser.add_argument("--repo-id", required=True, help="Existing model repo: OWNER/NAME")
        subparser.add_argument(
            "--local-dir", type=Path, required=(command == "upload"),
            default=None if command == "upload" else DEFAULT_CHECKPOINT_DIR,
            help=("Staged directory containing only the desired model tree" if command == "upload"
                  else "Destination directory (default: this repository's checkpoints/ directory)"),
        )
        subparser.add_argument("--revision", help="Model repo branch/tag/commit (upload: existing branch)")
        subparser.add_argument(
            "--dry-run", action="store_true",
            help=("List selected local files and total size without importing the Hub client or networking"
                  if command == "upload" else "Show the destination/filter policy without networking"),
        )
        if command == "upload":
            subparser.add_argument(
                "--path-in-repo", type=_repo_path, default="",
                help="Optional destination subfolder inside the model repo",
            )
    return parser


def _hub_client():
    try:
        import huggingface_hub
    except ImportError as error:
        raise RuntimeError("Install huggingface-hub before transferring checkpoints.") from error
    return huggingface_hub


def upload(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    local_dir = args.local_dir.expanduser().resolve()
    if not local_dir.is_dir():
        parser.error(f"upload directory does not exist or is not a directory: {local_dir}")
    if _excluded_directory(local_dir.name):
        parser.error("stage the model folders separately, or point directly to a checkpoint folder; "
                     "uploading an output/cache/media directory is disabled")
    files, skipped = collect_upload_files(local_dir)
    if not files:
        parser.error("no eligible checkpoint/config/tokenizer files found in the staged directory")
    print(f"Source: {local_dir}")
    print(f"Model repo: {args.repo_id}")
    print(f"Destination: {args.path_in_repo or '/'}")
    if args.revision:
        print(f"Revision: {args.revision}")
    for relative_path, size in files:
        print(f"  {relative_path}  ({size:,} bytes)")
    total = sum(size for _, size in files)
    print(f"Selected {len(files)} files: {_human_size(total)} ({total:,} bytes).")
    print(f"Skipped {skipped} files/directories (media, outputs, caches, hidden files, symlinks, or unsupported types).")
    if args.dry_run:
        print("Dry run complete; no network requests or uploads were made.")
        return 0
    hub = _hub_client()
    # Exact, escaped paths preserve the reviewed inventory even when another
    # eligible file exists in an excluded folder or uses glob characters.
    result = hub.HfApi().upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        path_in_repo=args.path_in_repo or None,
        revision=args.revision,
        allow_patterns=[escape_glob(relative_path) for relative_path, _ in files],
        ignore_patterns=list(IGNORE_PATTERNS),
        commit_message="Upload RAV model checkpoints and configuration",
    )
    print(f"Upload complete: {result.commit_url}")
    return 0


def download(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    local_dir = args.local_dir.expanduser().resolve()
    if local_dir.exists() and not local_dir.is_dir():
        parser.error(f"download destination is not a directory: {local_dir}")
    print(f"Model repo: {args.repo_id}")
    print(f"Destination: {local_dir}")
    if args.revision:
        print(f"Revision: {args.revision}")
    if args.dry_run:
        print("Allowed files: model tensors, configuration, tokenizer data, and license/readme metadata.")
        print("Excluded: media, outputs, caches, hidden files, and unsupported file types.")
        print("Remote file names and total size require a network request and are unavailable in this dry run.")
        print("Dry run complete; no network requests or downloads were made.")
        return 0
    hub = _hub_client()
    destination = hub.snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=str(local_dir),
        allow_patterns=list(ALLOW_PATTERNS),
        # The source repository keeps its setup guide at checkpoints/README.md.
        # Download model assets and licenses without replacing that guide.
        ignore_patterns=[*IGNORE_PATTERNS, "README.md"],
    )
    print(f"Download complete: {destination}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Do not accept tokens on the command line or print authentication state.
    # huggingface_hub reads HF_TOKEN or the cached login only during transfer.
    try:
        if args.command == "upload":
            return upload(args, parser)
        return download(args, parser)
    except (OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
