"""Paper-aligned wrapper for VDD Identity Adapter + Track4Gen inference."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.ipadapter_track4gen_multi import main


if __name__ == "__main__":
    main()
