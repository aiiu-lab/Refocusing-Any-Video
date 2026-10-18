# Paper-to-code map

This release contains inference and preprocessing. Training code and dataset-construction scripts will be released later.

| Concept | Entrypoint | Implementation |
| --- | --- | --- |
| VDD | `infer_vdd.sh` | `inference/ipadapter_multi.py`, `pipelines/base.py` |
| Identity Adapter | `infer_vdd_identity.sh` | `rav/models/ipadapter_kp.py`, `pipelines/ipadapter_kp.py` |
| Track4Gen Refiner | `infer_vdd_identity_track4gen.sh` | `rav/models/ipadapter_kp_track4gen_refiner.py` |
| VBR | `infer_vbr_defocus_map.sh` | `inference/refocus_depth.py`, `pipelines/refocus_depth.py` |
| Depth estimation | `VideoDepthAnything/infer_video_depth.py` | `VideoDepthAnything/video_depth_anything/` |
| Bokeh preprocessing | `tools/preprocessing/create_bokeh_video.py` | `rav/renderers/bokeh.py` |
| Focal-plane sequence | `tools/preprocessing/focal_plane.py` | Per-frame text values |
| Face control videos | `tools/preprocessing/face_keypoints.py` | InsightFace keypoints |
| Facial landmarks | `tools/preprocessing/mediapipe_landmarks.py` | Landmark CSVs |

`rav/` contains model definitions, face encoders, external runtime code, and optional renderers needed for inference. It contains no training entrypoints, datasets, losses, optimizer code, or training configurations. `simple_pipeline.py` exposes the shared diffusion pipeline imports.

Historical parameter names `zero_depth` and `K` correspond to focal plane and CoC/bokeh scale. `MODE=deblur` remains an alias for `MODE=vdd`.
