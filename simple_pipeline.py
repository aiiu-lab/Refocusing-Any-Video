"""Unified import surface for the project SVD pipelines.

The three implementation modules are kept separate because their conditioning
paths differ:

- ``pipelines/vdd.py``: VDD / base SVD defocus-deblurring pipeline
- ``pipelines/vdd_identity.py``: VDD + Identity Adapter / keypoint pipeline
- ``pipelines/vbr_defocus_map.py``: VBR / defocus-map refocusing pipeline

Import from this module in new code so the callers do not need to know the
historical file names. A later cleanup can move the shared implementation into
this file without touching training/inference scripts again.
"""

from __future__ import annotations

from typing import Literal, Type

PipelineMode = Literal[
    "base",
    "ipadapter_kp",
    "refocus_depth",
    "vdd",
    "vdd_identity",
    "vbr_defocus_map",
]

_MODE_ALIASES = {
    "base": "vdd",
    "ipadapter_kp": "vdd_identity",
    "refocus_depth": "vbr_defocus_map",
}


def get_pipeline_class(mode: PipelineMode = "base") -> Type:
    """Return the pipeline class for a supported conditioning mode."""
    mode = _MODE_ALIASES.get(mode, mode)

    if mode == "vdd":
        from pipelines.vdd import StableVideoDiffusionPipeline

        return StableVideoDiffusionPipeline

    if mode == "vdd_identity":
        from pipelines.vdd_identity import StableVideoDiffusionPipeline

        return StableVideoDiffusionPipeline

    if mode == "vbr_defocus_map":
        from pipelines.vbr_defocus_map import StableVideoDiffusionPipeline

        return StableVideoDiffusionPipeline

    raise ValueError(
        f"Unknown pipeline mode {mode!r}. Expected one of: "
        "'vdd', 'vdd_identity', 'vbr_defocus_map' "
        "(or historical aliases 'base', 'ipadapter_kp', 'refocus_depth')."
    )


BaseStableVideoDiffusionPipeline = get_pipeline_class("base")
IPAdapterKPStableVideoDiffusionPipeline = get_pipeline_class("ipadapter_kp")
RefocusDepthStableVideoDiffusionPipeline = get_pipeline_class("refocus_depth")
VDDStableVideoDiffusionPipeline = BaseStableVideoDiffusionPipeline
VDDIdentityStableVideoDiffusionPipeline = IPAdapterKPStableVideoDiffusionPipeline
VBRDefocusMapStableVideoDiffusionPipeline = RefocusDepthStableVideoDiffusionPipeline

# Backward-friendly default for generic/base SVD code.
StableVideoDiffusionPipeline = BaseStableVideoDiffusionPipeline

__all__ = [
    "BaseStableVideoDiffusionPipeline",
    "IPAdapterKPStableVideoDiffusionPipeline",
    "RefocusDepthStableVideoDiffusionPipeline",
    "StableVideoDiffusionPipeline",
    "VBRDefocusMapStableVideoDiffusionPipeline",
    "VDDIdentityStableVideoDiffusionPipeline",
    "VDDStableVideoDiffusionPipeline",
    "get_pipeline_class",
]
