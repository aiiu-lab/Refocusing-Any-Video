"""Paper-aligned VBR defocus-map pipeline import surface.

The implementation currently lives in the historical ``pipelines.refocus_depth``
module. This wrapper lets new code import the paper name without breaking old
imports.
"""

from pipelines.refocus_depth import *  # noqa: F401,F403
