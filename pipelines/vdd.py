"""Paper-aligned VDD pipeline import surface.

The implementation currently lives in the historical ``pipelines.base`` module.
This wrapper lets new code import the paper name without breaking old imports.
"""

from pipelines.base import *  # noqa: F401,F403
