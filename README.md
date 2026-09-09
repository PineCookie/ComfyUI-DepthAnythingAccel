# ComfyUI DepthAnythingAccel

Native-resolution **Depth Anything V2 / Distill-Any-Depth** depth estimation nodes for
ComfyUI — one pipeline that runs the same model on **PyTorch, ONNX Runtime, or TensorRT**.

中文文档见 [README_zh.md](README_zh.md)。

## What it does

- **Native resolution.** The input is not resized to 518×518: it is only
  aligned to the 14-pixel patch grid and the depth map is resized back to the
  original image size afterwards.
- **Three backends, one pipeline.** The same model can run on PyTorch
  (reference), ONNX Runtime (CUDA), or TensorRT. Nodes are chained directly in
  the graph — `Load → Export to ONNX → (Fuse) → (Build TensorRT) → Estimate` —
  with progress printed to the console. No command line, no refresh.
- **DA2 and DAD in one node set.** Distill-Any-Depth uses the same architecture
  as Depth Anything V2, so both families share the same nodes. The encoder
  (small/base/large/giant) and the family (DA2 vs DAD) are detected from the
  checkpoint automatically.
- **Dynamic input size.** One exported model or engine accepts any
  batch/height/width (14-pixel aligned).
- **Auto-download.** If a supported checkpoint is missing, the Load node
  downloads it into `ComfyUI/models/depthanything/`.

Measured on an RTX 5070 Ti (FP16, 784×1176):

| Encoder | PyTorch | ONNX Runtime (fused) | TensorRT |
| --- | ---: | ---: | ---: |
| Small (`vits`) | 29.1 ms | 20.5 ms | 7.5 ms |
| Base (`vitb`) | 51.0 ms | 37.4 ms | 16.5 ms |
| Large (`vitl`) | 127.3 ms | 93.5 ms | 47.3 ms |

TensorRT is about 2–2.7× faster than ONNX Runtime, which is about 1.4× faster
than PyTorch. All backends match the FP32 reference output closely
(correlation ≈ 1.0, normalized error ≈ 1e-3).

## Installation

### Option 1: ComfyUI Manager (recommended)

Search for `DepthAnythingAccel` in ComfyUI Manager and install it.

### Option 2: manual

```powershell
git clone <repository-url> "$env:ComfyUI\custom_nodes\ComfyUI-DepthAnythingAccel"
& "$env:ComfyUI\.venv\Scripts\python.exe" -m pip install safetensors onnxruntime-gpu
```

> Install `onnx` and `tensorrt` only if you want to export ONNX models or build TensorRT engines.

ComfyUI-Manager installs dependencies from `requirements.txt` automatically;
the optional conversion/engine packages (`onnx`, `onnxslim`, `tensorrt`) are
listed there commented out.

## Download models

Weights are not bundled with this repository. You can download them manually, or pick a supported checkpoint in the Load node's dropdown — models that are not present are downloaded automatically into `ComfyUI/models/depthanything/`:

| Model | Source | Notes |
| --- | --- | --- |
| DA2 Small / Base / Large | Kijai safetensors (default) or official `.pth` | Base / Large / Giant are non-commercial |
| Distill-Any-Depth | `xingyang1/Distill-Any-Depth` | Small / Base / Large (Multi-Teacher); family is auto-detected from the name |

> ⚠️ Official DA2 Base / Large / Giant weights are **non-commercial** (CC-BY-NC). Read [THIRD_PARTY.md](THIRD_PARTY.md) before using or redistributing them.

## Quick start

1. Find the `DepthAnythingAccel` category in the node panel.
2. Connect an image:

```text
[Load Depth Model (PyTorch)] -> [Estimate Depth (PyTorch)] -> Save Image
```

The encoder and the DA2/DAD family are detected automatically from the
checkpoint; you only pick the model file and (optionally) the precision.

## Want more speed? Use ONNX / TensorRT

Conversion nodes are **chainable**: each one takes the previous model and
returns a ready-to-use model handle, so nothing needs a ComfyUI refresh. The
generated files are also written into `ComfyUI/models/depthanything/` so the
`Load ...` nodes can reuse them later.

**ONNX (~1.4× faster)**

```text
[Load Depth Model (PyTorch)] -> [Export to ONNX] -> [Estimate Depth (ONNX)]
```

`Export to ONNX` returns an explicit-graph model; `Fuse ONNX Attention`
(optional, ~faster) rewrites it into ORT's fused-attention graph. Prefer the
fused handle when you do not plan to build a TensorRT engine:

```text
[Load Depth Model (PyTorch)] -> [Export to ONNX] -> [Fuse ONNX Attention] -> [Estimate Depth (ONNX)]
```

**TensorRT (~2–2.7×, fastest)**

```text
[Load Depth Model (PyTorch)] -> [Export to ONNX] -> [Build TensorRT Engine] -> [Estimate Depth (TensorRT)]
```

> A TensorRT engine must be built from the explicit (non-fused) ONNX graph; its
> `Max H` / `Max W` profile bounds the largest input resolution it accepts.

## Nodes

**PyTorch**

- `Load Depth Model (PyTorch)`
- `Estimate Depth (PyTorch)`

**ONNX Runtime**

- `Load Depth Model (ONNX)`
- `Estimate Depth (ONNX)`

**TensorRT**

- `Load Depth Model (TensorRT)`
- `Estimate Depth (TensorRT)`

**Conversion**

- `Export to ONNX` — PyTorch model → explicit ONNX model handle
- `Fuse ONNX Attention` — explicit ONNX → fused ONNX model handle
- `Build TensorRT Engine` — explicit ONNX → TensorRT engine handle

## License

Source code is licensed under the Apache License 2.0; see [LICENSE](LICENSE). Model weights are licensed separately by their respective owners; see [THIRD_PARTY.md](THIRD_PARTY.md).

## For developers

CLI tools, tests, and development notes are in [DEVELOPMENT.md](DEVELOPMENT.md).
