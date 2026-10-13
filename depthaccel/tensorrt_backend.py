"""TensorRT backend for dynamic-shape engines built from the explicit graph."""

from dataclasses import dataclass, field
from pathlib import Path

import torch

from .paths import resolve_model_file
from .preprocessing import PATCH_SIZE, prepare_image, restore_depth


@dataclass
class TensorrtDepthModel:
    engine: object
    input_name: str
    output_name: str
    model_path: Path
    device: torch.device
    precision: str
    torch_dtype: torch.dtype
    min_hw: tuple[int, int]
    opt_hw: tuple[int, int]
    max_hw: tuple[int, int]
    stream: torch.cuda.Stream
    _context: object = field(default=None, repr=False)


def _engine_version_path(engine_path: Path) -> Path:
    """Sidecar file recording the TensorRT version that built an engine."""
    return Path(str(engine_path) + ".version")


def check_engine_version(engine_path: Path, current_version: str) -> None:
    """Raise if an engine was built with a different TensorRT version.

    Serialized engines are tied to the TensorRT version that built them; a
    silent mismatch would otherwise surface as confusing deserialize failures.
    Engines without a sidecar (built by other tools) are skipped.
    """
    sidecar = _engine_version_path(engine_path)
    if not sidecar.is_file():
        return
    built = sidecar.read_text(encoding="utf-8").strip()
    if built and built != current_version:
        raise RuntimeError(
            f"TensorRT engine {engine_path.name} was built with TensorRT "
            f"{built}, but the installed runtime is {current_version}. Engines "
            "are version-specific; rebuild it with tools/export_tensorrt.py or "
            "the Build TensorRT Engine node."
        )


def load_tensorrt_model(
    model_path: str | Path,
    device: str = "cuda",
    comfyui_root: Path | None = None,
) -> TensorrtDepthModel:
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            "TensorRT is not installed. Install tensorrt to load TensorRT engines."
        ) from error
    from .trt_logger import get_trt_logger

    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("TensorRT engines require a CUDA device.")

    path = resolve_model_file(
        model_path,
        description="TensorRT engine",
        comfyui_root=comfyui_root,
        search_plugin_artifacts=True,
    )
    check_engine_version(path, getattr(trt, "__version__", "unknown"))
    runtime = trt.Runtime(get_trt_logger())
    with path.open("rb") as source:
        engine = runtime.deserialize_cuda_engine(source.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {path}")
    names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    inputs = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
    outputs = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            "Expected one TensorRT input and one output, "
            f"got {len(inputs)} inputs and {len(outputs)} outputs."
        )

    input_name, output_name = inputs[0], outputs[0]
    dtype = engine.get_tensor_dtype(input_name)
    if dtype == trt.DataType.HALF:
        precision, torch_dtype = "fp16", torch.float16
    elif dtype == trt.DataType.FLOAT:
        precision, torch_dtype = "fp32", torch.float32
    else:
        raise ValueError(f"Unsupported TensorRT input dtype: {dtype}")

    min_shape, opt_shape, max_shape = engine.get_tensor_profile_shape(input_name, 0)
    stream = torch.cuda.Stream(device=target)

    return TensorrtDepthModel(
        engine=engine,
        input_name=input_name,
        output_name=output_name,
        model_path=path,
        device=target,
        precision=precision,
        torch_dtype=torch_dtype,
        min_hw=(min_shape[2], min_shape[3]),
        opt_hw=(opt_shape[2], opt_shape[3]),
        max_hw=(max_shape[2], max_shape[3]),
        stream=stream,
    )


def _bind(
    model: TensorrtDepthModel,
    prepared: torch.Tensor,
) -> tuple[object, torch.Tensor]:
    """Bind the input and output buffers for the prepared tensor's shape.

    A single execution context is created lazily and reused across shapes:
    TensorRT requires ``set_input_shape`` before every enqueue and it cheaply
    accepts repeated calls, so recreating a context per shape change (e.g. a
    video stream of varying sizes) is unnecessary.
    """
    batch, channels, height, width = prepared.shape
    shape = (batch, channels, height, width)
    if model._context is None:
        model._context = model.engine.create_execution_context()
    if not model._context.set_input_shape(model.input_name, shape):
        raise RuntimeError(f"TensorRT rejected input shape {shape}.")

    context = model._context
    depth = torch.empty(
        (batch, height, width),
        device=model.device,
        dtype=model.torch_dtype,
    )
    context.set_tensor_address(model.input_name, prepared.data_ptr())
    context.set_tensor_address(model.output_name, depth.data_ptr())
    return context, depth


def infer_tensorrt_raw(
    model: TensorrtDepthModel,
    prepared: torch.Tensor,
) -> torch.Tensor:
    """Run TensorRT on a prepared BCHW tensor without output normalization."""
    prepared = prepared.contiguous().to(device=model.device, dtype=model.torch_dtype)
    batch, channels, height, width = prepared.shape
    if batch != 1:
        raise ValueError(
            "The TensorRT engine profile has batch 1; run each sample individually."
        )
    if channels != 3:
        raise ValueError(f"Expected 3 channels, got {channels}.")
    if height < model.min_hw[0] or width < model.min_hw[1] or height > model.max_hw[0] or width > model.max_hw[1]:
        # The profile bounds each side independently, so a valid total pixel
        # count can still exceed a per-side limit; name the bounds to rebuild.
        required_h = -(-height // PATCH_SIZE) * PATCH_SIZE
        required_w = -(-width // PATCH_SIZE) * PATCH_SIZE
        raise ValueError(
            f"TensorRT input {height}x{width} is outside the engine profile "
            f"[{model.min_hw[0]},{model.min_hw[1]}]..[{model.max_hw[0]},{model.max_hw[1]}]. "
            f"The profile limits each side independently, not the total pixel "
            f"count, so rebuild the engine with Max H / Max W of at least "
            f"{required_h}x{required_w} (multiples of {PATCH_SIZE})."
        )

    context, depth = _bind(model, prepared)
    # TensorRT runs on its own stream. Order the producer (the PyTorch current
    # stream that filled ``prepared``) before the engine stream, and the engine
    # stream before the current stream that consumes ``depth``; otherwise TRT
    # can read a half-written input buffer or the caller can read a not-yet-run
    # output in a streamed ComfyUI pipeline.
    current_stream = torch.cuda.current_stream(model.device)
    model.stream.wait_stream(current_stream)
    if not context.execute_async_v3(model.stream.cuda_stream):
        raise RuntimeError("TensorRT inference failed.")
    current_stream.wait_stream(model.stream)
    return depth


def estimate_depth_tensorrt(model: TensorrtDepthModel, image: torch.Tensor) -> torch.Tensor:
    """Return a normalized ComfyUI IMAGE tensor at the original size."""
    prepared, geometry = prepare_image(image)
    outputs = [infer_tensorrt_raw(model, sample.unsqueeze(0)) for sample in prepared]
    depth = torch.cat(outputs, dim=0)
    return restore_depth(depth, geometry).cpu()
