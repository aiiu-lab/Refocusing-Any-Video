"""Paper-aligned VDD + Identity Adapter pipeline import surface.

The implementation currently lives in the historical ``pipelines.ipadapter_kp``
module. This wrapper lets new code import the paper name without breaking old
imports.
"""

from pipelines.ipadapter_kp import *  # noqa: F401,F403
