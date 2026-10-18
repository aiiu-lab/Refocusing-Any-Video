# Third-party notices

Existing copyright headers and model-license texts are retained. This file records evidence found in the supplied source tree; it does not choose a license for the authors' project code or for newly distributed model weights.

| Component | Local evidence |
| --- | --- |
| facenet-pytorch runtime | MIT license in `rav/third_party/facenet_pytorch/LICENSE.md`, copyright Timothy Esler (2019) |
| Video Depth Anything / DINOv2 / motion modules | Existing ByteDance, Meta, and Hugging Face headers in `VideoDepthAnything/`; several refer to Apache 2.0 |
| DepthCrafter utility | Existing MIT attribution in `VideoDepthAnything/utils/dc_utils.py` |
| ArcFace backbone | Existing attribution to InsightFace_Pytorch in `rav/face/arcface.py` |
| Stable Video Diffusion and related model families | License texts preserved in `model_licenses/`; each applies to its corresponding model |

The facenet runtime was copied from upstream commit `787da06156087cd6b616fe6608213722bddc30cd`, with the local `detect_face.py` changes preserved. Its checkpoint loader now resolves separately downloaded MTCNN weights under `checkpoints/facenet/` or `FACENET_CHECKPOINT_DIR`.

Before publishing, the authors should supply the top-level project license and complete upstream license/notice files for vendored components where only headers were present. No VideoDepthAnything root license file was included in the supplied directory. Identify the provenance of the neural bokeh renderer and retain its applicable notices. Publish third-party checkpoints through their original sources unless their distribution terms allow inclusion in the model release.
