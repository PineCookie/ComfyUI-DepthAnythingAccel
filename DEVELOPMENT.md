# Development notes

CLI tools and test commands for developing DepthAnythingAccel. End-user documentation
is in [README.md](README.md).

`tools/` holds the runtime conversion modules that the in-ComfyUI nodes import;
`devtools/` holds the development and benchmark utilities below.

## Test without starting ComfyUI

Define your local ComfyUI root once, then run the tests from this directory:

```powershell
$comfyuiRoot = "C:\path\to\ComfyUI"
$python = Join-Path $comfyuiRoot ".venv\Scripts\python.exe"
& $python -m unittest discover -s tests -v
```

Run the local reference benchmark against an existing input image:

```powershell
& $python devtools\benchmark_reference.py `
  --image (Join-Path $comfyuiRoot "input\your-image.png") `
  --model-path "depth_anything_v2_vitl_fp32.safetensors" `
  --encoder vitl `
  --precision fp32
```

The benchmark reports native inference dimensions, latency, output shape, and
peak CUDA memory. It does not require a running ComfyUI server.

## Dynamic ONNX development

Export the current DA2 reference at a non-square, patch-aligned tracing size:

```powershell
& $python tools\export_onnx.py `
  --output artifacts\da2_vitl_fp32_dynamic.onnx `
  --model-path "depth_anything_v2_vitl_fp32.safetensors" `
  --encoder vitl `
  --height 560 `
  --width 784
```

Add `--precision fp16` to export an FP16 graph. FP32 remains the conservative
default; FP16 is substantially faster on supported NVIDIA GPUs and showed low
normalized error on the tested anime images. The ONNX loader derives buffer
dtypes from the graph and rejects inconsistent precision metadata.

Replace the explicit attention subgraphs with ONNX Runtime's fused attention
operator. This is the performance-critical step for the current `vitl` graph:

```powershell
& $python tools\fuse_onnx_attention.py `
  --input artifacts\da2_vitl_fp32_dynamic.onnx `
  --output artifacts\da2_vitl_fp32_dynamic_fused.onnx `
  --encoder vitl `
  --operator attention
```

Benchmark the fused graph independently:

```powershell
& $python devtools\benchmark_onnx.py `
  --model artifacts\da2_vitl_fp32_dynamic_fused.onnx `
  --image (Join-Path $comfyuiRoot "input\your-image.png") `
  --warmup 1 `
  --runs 3 `
  --json
```

Compare the ONNX output with the PyTorch reference at the same resolution:

```powershell
& $python devtools\compare_reference_onnx.py `
  --model artifacts\da2_vitl_fp32_dynamic_fused.onnx `
  --image (Join-Path $comfyuiRoot "input\your-image.png") `
  --height 784 `
  --width 1176 `
  --device cuda `
  --json
```

The explicit graph is the canonical portable artifact. The attention-fused
graph uses ONNX Runtime's `com.microsoft` operator domain and is therefore an
ORT-specific deployment artifact, not a TensorRT interchange model. Fusion
validates the selected encoder, block count, QKV shapes, bias wiring, and graph
consumer boundaries before rewriting any nodes.

## TensorRT CLI tools

Engines are built from the explicit (non-fused) ONNX graph:

```powershell
& $python tools\export_tensorrt.py `
  --input artifacts\da2_vitl_fp16_dynamic.onnx `
  --output artifacts\da2_vitl_fp16_dynamic.trt `
  --min-hw 14 14 --opt-hw 1022 1022 --max-hw 2016 2016
```

Engine precision follows the ONNX graph dtype (build from an FP16 graph for an
FP16 engine). Benchmark and validate the engine:

```powershell
& $python devtools\benchmark_tensorrt.py --engine artifacts\da2_vitl_fp16_dynamic.trt --image <image>

& $python devtools\compare_reference_tensorrt.py --engine artifacts\da2_vitl_fp16_dynamic.trt `
  --reference-model depth_anything_v2_vitl_fp32.safetensors --encoder vitl --precision fp16 `
  --image <image> --height 784 --width 1176
```

## Other CLI tools

All development utilities live under `devtools/`:

- `devtools/convert_checkpoint.py` - convert `.pth` to `.safetensors` (fp32/fp16/bf16).
- `devtools/compare_checkpoint_formats.py` - prove a `.pth`/`.safetensors`
  conversion is lossless (key, shape, dtype, and value comparison).
- `devtools/validate_model_coverage.py` - validate real DA2/DAD checkpoints
  against the shared runtime.
- `devtools/compare_reference_precisions.py` - compare FP32/BF16/FP16 autocast
  against the FP32 reference.
- `devtools/compare_onnx_precisions.py`, `devtools/validate_p0.py` - additional
  diagnostics.

## Reports

Per-phase validation reports (parity, provenance, precision, TensorRT) are
kept locally under the git-ignored `artifacts/dev_notes/` folder and are not
committed to Git.
