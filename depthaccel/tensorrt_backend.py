"""TensorRT backend for dynamic-shape engines built from the explicit graph."""

from dataclasses import dataclass, field
from pathlib import Path

import torch

from .paths import resolve_model_file
from .preprocessing import PATCH_SIZE, prepare_image, restore_depth, validate_image_tensor


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
    use_cuda_graph: bool = True
    _context: object = field(default=None, repr=False)
    _last_shape: tuple[int, int, int, int] | None = field(default=None, repr=False)
    _graph_shape: tuple[int, int, int, int] | None = field(default=None, repr=False)
    _graph: torch.cuda.CUDAGraph | None = field(default=None, repr=False)
    _graph_input: torch.Tensor | None = field(default=None, repr=False)
    _graph_output: torch.Tensor | None = field(default=None, repr=False)


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


def _resolve_cuda_device(device: str | torch.device) -> torch.device:
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("TensorRT engines require a CUDA device.")
    if target.index is None:
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
        return torch.device("cuda", device_id)
    return target


def _validate_engine_contract(
    engine,
    trt,
) -> tuple[str, str, str, torch.dtype, tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Validate that a TensorRT engine matches DepthAccel's I/O contract."""
    names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    inputs = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
    outputs = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            "Expected one TensorRT input and one output, "
            f"got {len(inputs)} inputs and {len(outputs)} outputs."
        )

    input_name, output_name = inputs[0], outputs[0]
    input_dtype = engine.get_tensor_dtype(input_name)
    output_dtype = engine.get_tensor_dtype(output_name)
    if input_dtype == trt.DataType.HALF:
        precision, torch_dtype = "fp16", torch.float16
    elif input_dtype == trt.DataType.FLOAT:
        precision, torch_dtype = "fp32", torch.float32
    else:
        raise ValueError(f"Unsupported TensorRT input dtype: {input_dtype}")
    if output_dtype != input_dtype:
        raise ValueError(
            "DepthAccel TensorRT engines must use matching input/output dtypes, "
            f"got input={input_dtype}, output={output_dtype}."
        )

    input_shape = tuple(engine.get_tensor_shape(input_name))
    output_shape = tuple(engine.get_tensor_shape(output_name))
    if len(input_shape) != 4 or (input_shape[1] not in (-1, 3)):
        raise ValueError(f"Expected TensorRT input [B,3,H,W], got {input_shape}.")
    if len(output_shape) != 3:
        raise ValueError(f"Expected TensorRT output [B,H,W], got {output_shape}.")

    if getattr(engine, "num_optimization_profiles", 1) < 1:
        raise ValueError("TensorRT engine must define at least one optimization profile.")
    min_shape, opt_shape, max_shape = engine.get_tensor_profile_shape(input_name, 0)
    if len(min_shape) != 4 or min_shape[1] != 3 or max_shape[1] != 3:
        raise ValueError(
            f"Expected TensorRT input profile [B,3,H,W], got min={tuple(min_shape)}, max={tuple(max_shape)}."
        )
    if min_shape[0] > 1 or max_shape[0] < 1:
        raise ValueError(
            f"TensorRT engine profile must support batch 1, got batch range {min_shape[0]}..{max_shape[0]}."
        )

    return (
        input_name,
        output_name,
        precision,
        torch_dtype,
        (int(min_shape[2]), int(min_shape[3])),
        (int(opt_shape[2]), int(opt_shape[3])),
        (int(max_shape[2]), int(max_shape[3])),
    )


def load_tensorrt_model(
    model_path: str | Path,
    device: str | torch.device = "cuda",
    comfyui_root: Path | None = None,
) -> TensorrtDepthModel:
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            "TensorRT is not installed. Install tensorrt-cu12 or tensorrt-cu13 "
            "to match torch.version.cuda."
        ) from error
    from .trt_logger import get_trt_logger

    target = _resolve_cuda_device(device)

    path = resolve_model_file(
        model_path,
        description="TensorRT engine",
        comfyui_root=comfyui_root,
        search_plugin_artifacts=True,
    )
    check_engine_version(path, getattr(trt, "__version__", "unknown"))
    with torch.cuda.device(target):
        runtime = trt.Runtime(get_trt_logger())
        with path.open("rb") as source:
            engine = runtime.deserialize_cuda_engine(source.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {path}")

    (
        input_name,
        output_name,
        precision,
        torch_dtype,
        min_hw,
        opt_hw,
        max_hw,
    ) = _validate_engine_contract(engine, trt)
    stream = torch.cuda.Stream(device=target)

    return TensorrtDepthModel(
        engine=engine,
        input_name=input_name,
        output_name=output_name,
        model_path=path,
        device=target,
        precision=precision,
        torch_dtype=torch_dtype,
        min_hw=min_hw,
        opt_hw=opt_hw,
        max_hw=max_hw,
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


def _clear_cuda_graph(model: TensorrtDepthModel) -> None:
    model._graph = None
    model._graph_shape = None
    model._graph_input = None
    model._graph_output = None


def _capture_cuda_graph(
    model: TensorrtDepthModel,
    shape: tuple[int, int, int, int],
) -> None:
    """Capture a TensorRT CUDA Graph for ``shape`` on ``model.stream``."""
    batch, channels, height, width = shape
    _clear_cuda_graph(model)
    if model._context is None:
        model._context = model.engine.create_execution_context()
    context = model._context
    if not context.set_input_shape(model.input_name, shape):
        raise RuntimeError(f"TensorRT rejected input shape {shape}.")

    static_in = torch.empty(shape, device=model.device, dtype=model.torch_dtype)
    static_out = torch.empty(
        (batch, height, width), device=model.device, dtype=model.torch_dtype
    )
    context.set_tensor_address(model.input_name, static_in.data_ptr())
    context.set_tensor_address(model.output_name, static_out.data_ptr())

    # Synchronize the stream before graph capture to ensure prior eager runs
    # finished cleanly on this execution context.
    model.stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=model.stream):
            if not context.execute_async_v3(model.stream.cuda_stream):
                raise RuntimeError("TensorRT CUDA Graph capture failed.")
    except Exception:
        _clear_cuda_graph(model)
        raise

    model._graph = graph
    model._graph_shape = shape
    model._graph_input = static_in
    model._graph_output = static_out


def infer_tensorrt_raw(
    model: TensorrtDepthModel,
    prepared: torch.Tensor,
    *,
    _allow_output_alias: bool = False,
) -> torch.Tensor:
    """Run TensorRT on a prepared BCHW tensor without output normalization."""
    if prepared.ndim != 4:
        raise ValueError(f"Expected prepared tensor [B,3,H,W], got {tuple(prepared.shape)}.")
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
        raise ValueError(
            f"TensorRT input {height}x{width} is outside the engine profile "
            f"[{model.min_hw[0]},{model.min_hw[1]}]..[{model.max_hw[0]},{model.max_hw[1]}]. "
            f"The profile limits each side independently, not the total pixel "
            f"count, so rebuild the engine with Max H / Max W of at least "
            f"{height}x{width} (multiples of {PATCH_SIZE})."
        )

    shape = (batch, channels, height, width)
    prepared = prepared.contiguous().to(device=model.device, dtype=model.torch_dtype)
    current_stream = torch.cuda.current_stream(model.device)

    # Lazily capture a CUDA Graph once a shape repeats (e.g. video frames or
    # repeated same-resolution runs), avoiding capture overhead on one-off sizes.
    if model.use_cuda_graph and model._graph_shape != shape:
        if model._graph_shape is not None:
            _clear_cuda_graph(model)
        elif model._last_shape == shape:
            _capture_cuda_graph(model, shape)

    if model.use_cuda_graph and model._graph is not None and model._graph_shape == shape:
        model.stream.wait_stream(current_stream)
        prepared.record_stream(model.stream)
        with torch.cuda.stream(model.stream):
            model._graph_input.copy_(prepared)
            model._graph.replay()
            depth = model._graph_output if _allow_output_alias else model._graph_output.clone()
        depth.record_stream(model.stream)
        current_stream.wait_stream(model.stream)
        return depth

    context, depth = _bind(model, prepared)
    # TensorRT runs on its own stream. Order the producer (the PyTorch current
    # stream that filled ``prepared``) before the engine stream, and the engine
    # stream before the current stream that consumes ``depth``; otherwise TRT
    # can read a half-written input buffer or the caller can read a not-yet-run
    # output in a streamed ComfyUI pipeline.
    model.stream.wait_stream(current_stream)
    prepared.record_stream(model.stream)
    depth.record_stream(model.stream)
    if not context.execute_async_v3(model.stream.cuda_stream):
        raise RuntimeError("TensorRT inference failed.")
    current_stream.wait_stream(model.stream)
    model._last_shape = shape
    return depth


def estimate_depth_tensorrt(model: TensorrtDepthModel, image: torch.Tensor) -> torch.Tensor:
    """Return a normalized ComfyUI IMAGE tensor at the original size."""
    batch, _, _ = validate_image_tensor(image)
    outputs = []
    with torch.inference_mode():
        for index in range(batch):
            prepared, geometry = prepare_image(
                image[index : index + 1],
                device=model.device,
                dtype=model.torch_dtype,
            )
            depth = infer_tensorrt_raw(model, prepared, _allow_output_alias=True)
            outputs.append(restore_depth(depth, geometry).cpu())
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
