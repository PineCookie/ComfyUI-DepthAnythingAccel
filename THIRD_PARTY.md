# Third-party code and model licenses

This document separates the license of source code shipped in this repository
from the licenses of model weights loaded by users. It is not legal advice.

## Vendored Depth Anything V2 inference architecture

Files under `depthaccel/models/da2/` are derived from the official
`DepthAnything/Depth-Anything-V2` repository at commit:

```text
a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
```

That implementation incorporates DINOv2 architecture code from Meta. The
official source repository distributes this code under Apache License 2.0. A
copy is included at
`third_party_licenses/Depth-Anything-V2-Apache-2.0.txt`.

DepthAnythingAccel modifications remove the OpenCV/NumPy image path, make patch-grid
shape computation exportable with dynamic H/W, and integrate the architecture
with ComfyUI tensor preprocessing and ONNX export.

Sources:

- https://github.com/DepthAnything/Depth-Anything-V2
- https://github.com/facebookresearch/dinov2

## Depth Anything V2 model weights

Code licensing does not grant a license for arbitrary checkpoints. The official
Depth Anything V2 project currently identifies these weight licenses:

| Encoder/model | Upstream weight license |
| --- | --- |
| DA2 Small (`vits`) | Apache-2.0 |
| DA2 Base (`vitb`) | CC-BY-NC-4.0 |
| DA2 Large (`vitl`) | CC-BY-NC-4.0 |
| DA2 Giant (`vitg`) | CC-BY-NC-4.0 |

The current performance artifact was exported from DA2 Large (`vitl`), so it
must be treated as non-commercial under the upstream CC-BY-NC-4.0 terms unless
the user has obtained separate rights.

Kijai's safetensors repository labels the converted collection CC-BY-4.0, but
format conversion does not clearly supersede the original model owner's more
restrictive license. DepthAnythingAccel therefore applies the conservative upstream
DA2 Large classification and does not redistribute checkpoints or generated
ONNX weight artifacts in Git.

Users are responsible for checking the exact source and license of any model
they load. DepthAnythingAccel should not infer commercial-use permission merely from a
filename or serialization format.

## Distill Any Depth

DepthAnythingAccel can load architecture-compatible Distill Any Depth checkpoints
without vendoring DAD source code. Compatibility was validated against the
official DAD-Small checkpoint: all 239 state-dict keys and tensor shapes match
the shared DA2-Small inference architecture. The official implementation's
forward method additionally exposes a backbone feature tensor; DepthAnythingAccel only
uses its depth output.

The official repository labels its sample code MIT, but this does not by
itself establish a license for every published checkpoint or erase the license
of DA2 initialization weights. DepthAnythingAccel therefore does not redistribute DAD
weights or generated ONNX artifacts. Users must verify the terms attached to
their selected checkpoint.

Source: https://github.com/Westlake-AGI-Lab/Distill-Any-Depth
Hugging Face mirror: https://huggingface.co/xingyang1/Distill-Any-Depth
(model card labeled Apache-2.0; checkpoints remain user-verified).
