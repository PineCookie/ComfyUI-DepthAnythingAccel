# ComfyUI DepthAnythingAccel

Native-resolution **Depth Anything V2 / Distill-Any-Depth** depth estimation nodes for
ComfyUI — one pipeline that runs the same model on **PyTorch, ONNX Runtime, or TensorRT**.

中文文档见 [README_zh.md](README_zh.md)。

## What it does

- **Native resolution.** The input is not resized to 518×518: it is only
  aligned to the 14-pixel patch grid, and the depth map is resized back to the
  original image size.
- **Three backends, one pipeline.** The same model runs on PyTorch
  (reference), ONNX Runtime (CUDA or CPU), or TensorRT. Conversion happens
  inside the graph — `Load → Export to ONNX → (Fuse) → (Build TensorRT) → Estimate` —
  with progress printed to the console. No command line, no refresh.
- **DA2 and DAD in one node set.** Distill-Any-Depth uses the same architecture
  as Depth Anything V2, so both share the same nodes. The encoder
  (small/base/large/giant) and the family (DA2 vs DAD) are detected
  automatically.
- **Auto-download.** Pick a supported checkpoint in the Load node and it is
  downloaded into `ComfyUI/models/depthanything/` if missing.

Measured on an RTX 5070 Ti (FP16, 784×1176):

| Encoder | PyTorch | ONNX Runtime (fused) | TensorRT |
| --- | ---: | ---: | ---: |
| Small (`vits`) | 29.1 ms | 20.5 ms | 7.5 ms |
| Base (`vitb`) | 51.0 ms | 37.4 ms | 16.5 ms |
| Large (`vitl`) | 127.3 ms | 93.5 ms | 47.3 ms |

TensorRT is about 2–2.7× faster than fused ONNX Runtime, which is about 1.4×
faster than PyTorch. All backends closely match the FP32 reference output
(correlation ≈ 1.0, normalized error ≈ 1e-3).

## Installation

### Option 1: ComfyUI Manager (recommended)

Search for `DepthAnythingAccel` in ComfyUI Manager and install it.

### Option 2: manual

```powershell
git clone https://github.com/PineCookie/ComfyUI-DepthAnythingAccel "$env:ComfyUI\custom_nodes\ComfyUI-DepthAnythingAccel"
& "$env:ComfyUI\.venv\Scripts\python.exe" -m pip install -r "$env:ComfyUI\custom_nodes\ComfyUI-DepthAnythingAccel\requirements.txt"
```

### Optional backends

The PyTorch nodes run on anything ComfyUI runs on and need nothing extra.
ONNX Runtime and TensorRT are optional; install what fits your machine into
ComfyUI's Python:

| Machine | ONNX Runtime | TensorRT |
| --- | --- | --- |
| NVIDIA GPU | `onnxruntime-gpu` | `tensorrt-cu12` or `tensorrt-cu13`, matching `torch.version.cuda` |
| Mac, AMD, Intel, CPU only | `onnxruntime` (runs on CPU) | not supported |

- Install only one of `onnxruntime` / `onnxruntime-gpu`; they overwrite each
  other. If another node pack installed `onnxruntime`, uninstall it before
  installing `onnxruntime-gpu`.
- `Export to ONNX` and `Fuse ONNX Attention` need `onnx` (`onnxslim` is
  optional). Export is verified on CUDA and CPU.
- FP16/BF16 precision in the PyTorch loader only takes effect on CUDA/ROCm;
  elsewhere it runs in FP32.

## Models

Weights are not bundled. Pick a checkpoint in the Load node's dropdown to
download it automatically, or place files in `ComfyUI/models/depthanything/`
yourself (subfolders work).

| Model | Source | License |
| --- | --- | --- |
| DA2 Small | `Kijai/DepthAnythingV2-safetensors` (default) or `depth-anything/Depth-Anything-V2-Small` (`.pth`) | Apache-2.0 |
| DA2 Base / Large | `Kijai/DepthAnythingV2-safetensors` (default) or `depth-anything/Depth-Anything-V2-Base` / `-Large` (`.pth`) | CC-BY-NC-4.0, **non-commercial** |
| Distill-Any-Depth Small / Base / Large | `xingyang1/Distill-Any-Depth` | Apache-2.0 (model card) |

> ⚠️ Kijai's conversions are labeled CC-BY-4.0, but they are converted from the
> official weights, so treat Base / Large as non-commercial either way. ONNX
> models and engines you export carry the same license as their checkpoint.
> See [THIRD_PARTY.md](THIRD_PARTY.md).

DA2 Giant has not been released publicly; a `vitg` checkpoint you supply
yourself is detected and loaded.

## Quick start

```text
[Load Image] -> [Estimate Depth (PyTorch)] -> [Save Image]
                         ^
        [Load Depth Model (PyTorch)]
```

Nodes are in the `DepthAnythingAccel` category. Pick a model file and
(optionally) a precision; the encoder and DA2/DAD family are detected from the
checkpoint.

## Faster: ONNX Runtime / TensorRT

Conversion nodes are **chainable**: each takes the previous model and returns a
ready-to-use model handle, so nothing needs a ComfyUI refresh. Generated files
are written to `ComfyUI/models/depthanything/`, so the `Load ...` nodes can
reuse them later.

**ONNX Runtime (~1.4× faster)**

```text
[Load Depth Model (PyTorch)] -> [Export to ONNX] -> [Fuse ONNX Attention] -> [Estimate Depth (ONNX)]
```

`Fuse ONNX Attention` rewrites the exported graph to use ONNX Runtime's fused
attention, which is where the speedup comes from. It is optional; the unfused
model also runs, just slower.

**TensorRT (~2–2.7× faster again, NVIDIA only)**

```text
[Load Depth Model (PyTorch)] -> [Export to ONNX] -> [Build TensorRT Engine] -> [Estimate Depth (TensorRT)]
```

> Build engines from the unfused ONNX model. An engine accepts any input up to
> its `Max H` / `Max W` and is tied to the TensorRT version and GPU that built
> it — rebuild after upgrading TensorRT or changing GPUs (engines built by
> these nodes report a version mismatch on load).

### Input size limits

To avoid running out of VRAM, the PyTorch and ONNX nodes refuse inputs above
about 2.6 MP in FP32 and 4.7 MP in FP16/BF16; resize larger images first.
TensorRT is limited by the engine's `Max H` / `Max W` instead. A batch of
images is processed in one call on ONNX Runtime and one image at a time on
PyTorch and TensorRT.

## Nodes

| Node | Does |
| --- | --- |
| `Load Depth Model (PyTorch)` / `Estimate Depth (PyTorch)` | Reference backend |
| `Load Depth Model (ONNX)` / `Estimate Depth (ONNX)` | ONNX Runtime backend |
| `Load Depth Model (TensorRT)` / `Estimate Depth (TensorRT)` | TensorRT backend |
| `Export to ONNX` | PyTorch model → ONNX model |
| `Fuse ONNX Attention` | ONNX model → fused ONNX model |
| `Build TensorRT Engine` | Unfused ONNX model → TensorRT engine |

## License

Source code is licensed under the Apache License 2.0; see [LICENSE](LICENSE).
Model weights are licensed separately by their owners; see
[THIRD_PARTY.md](THIRD_PARTY.md).

## For developers

CLI tools, tests, and development notes are in [DEVELOPMENT.md](DEVELOPMENT.md).
