"""ONNX Runtime backend for dynamic-shape DA2/DAD depth models.

On CUDA the session runs on the PyTorch current stream and input/output
buffers are bound by raw pointer (no CPU staging). Keep the ``torch`` import
above ONNX Runtime: torch preloads its bundled CUDA/cuDNN libraries first,
which avoids library-version conflicts when ORT initializes on Windows.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from .paths import resolve_model_file
from .preprocessing import prepare_image, restore_depth


DA2_METADATA_CONFIGS = {
    "vits": {"hidden_size": "384", "num_heads": "6"},
    "vitb": {"hidden_size": "768", "num_heads": "12"},
    "vitl": {"hidden_size": "1024", "num_heads": "16"},
    "vitg": {"hidden_size": "1536", "num_heads": "24"},
}


@dataclass
class OnnxDepthModel:
    session: object
    input_name: str
    output_name: str
    model_path: Path
    providers: tuple[str, ...]
    device: torch.device
    metadata: dict[str, str]
    compute_stream: int | None
    precision: str
    torch_dtype: torch.dtype


def _prepare_binding_tensor(
    prepared: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a contiguous tensor suitable for raw-pointer binding."""
    return prepared.contiguous().to(
        device=device,
        dtype=dtype,
        non_blocking=True,
    )


def _validate_model_contract(input_info, output_info, metadata: dict[str, str]) -> None:
    supported_types = {"tensor(float)": "fp32", "tensor(float16)": "fp16"}
    if input_info.type not in supported_types or output_info.type != input_info.type:
        raise ValueError(
            "DepthAccel ONNX models must use matching float32 or float16 input/output, got "
            f"input={input_info.type}, output={output_info.type}."
        )
    if len(input_info.shape) != 4 or input_info.shape[1] != 3:
        raise ValueError(f"Expected ONNX input [B,3,H,W], got {input_info.shape}.")
    if len(output_info.shape) != 3:
        raise ValueError(f"Expected ONNX output [B,H,W], got {output_info.shape}.")
    if not all(isinstance(input_info.shape[index], str) for index in (0, 2, 3)):
        raise ValueError(f"Expected dynamic ONNX batch/H/W axes, got {input_info.shape}.")
    if not all(isinstance(output_info.shape[index], str) for index in (0, 1, 2)):
        raise ValueError(f"Expected dynamic ONNX output axes, got {output_info.shape}.")
    architecture = metadata.get("depthaccel.architecture")
    if architecture is not None and architecture not in {
        "depth-anything-v2",
        "distill-any-depth",
    }:
        raise ValueError(f"Unsupported DepthAccel architecture metadata: {architecture!r}.")
    patch_size = metadata.get("depthaccel.patch_size")
    if patch_size is not None and patch_size != "14":
        raise ValueError(f"Unsupported DepthAccel patch size metadata: {patch_size!r}.")
    precision = metadata.get("depthaccel.precision")
    actual_precision = supported_types[input_info.type]
    if precision is not None and precision != actual_precision:
        raise ValueError(
            f"ONNX precision metadata is {precision!r}, but graph tensors are "
            f"{actual_precision!r}."
        )
    encoder = metadata.get("depthaccel.encoder")
    if encoder is not None:
        expected = DA2_METADATA_CONFIGS.get(encoder)
        if expected is None:
            raise ValueError(f"Unsupported DepthAccel encoder metadata: {encoder!r}.")
        for field, expected_value in expected.items():
            actual = metadata.get(f"depthaccel.{field}")
            if actual is not None and actual != expected_value:
                raise ValueError(
                    f"ONNX metadata mismatch for {encoder}: {field}={actual!r}, "
                    f"expected {expected_value!r}."
                )


def load_onnx_model(
    model_path: str | Path,
    device: str = "cuda",
    comfyui_root: Path | None = None,
    *,
    enable_profiling: bool = False,
    profile_prefix: str | None = None,
    use_user_compute_stream: bool = True,
) -> OnnxDepthModel:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "ONNX Runtime is not installed. Install the optional runtime "
            "dependency with: uv pip install onnxruntime-gpu"
        ) from error

    target = torch.device(device)
    path = resolve_model_file(
        model_path,
        description="ONNX depth model",
        comfyui_root=comfyui_root,
        search_plugin_artifacts=True,
    )
    available = ort.get_available_providers()
    compute_stream = None
    if target.type == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. "
                f"Available providers: {available}"
            )
        device_id = target.index or 0
        provider_options = {
            "device_id": device_id,
            "do_copy_in_default_stream": True,
        }
        if use_user_compute_stream:
            compute_stream = int(torch.cuda.current_stream(target).cuda_stream)
            provider_options["user_compute_stream"] = str(compute_stream)
        providers = [
            ("CUDAExecutionProvider", provider_options),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    options.enable_profiling = enable_profiling
    if profile_prefix is not None:
        options.profile_file_prefix = profile_prefix
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    actual_providers = tuple(session.get_providers())
    if target.type == "cuda" and actual_providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"ONNX Runtime did not select CUDA: {actual_providers}")

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            "Expected one ONNX input and one output, "
            f"got {len(inputs)} inputs and {len(outputs)} outputs."
        )
    metadata = dict(session.get_modelmeta().custom_metadata_map)
    _validate_model_contract(inputs[0], outputs[0], metadata)
    precision = "fp16" if inputs[0].type == "tensor(float16)" else "fp32"
    torch_dtype = torch.float16 if precision == "fp16" else torch.float32

    return OnnxDepthModel(
        session=session,
        input_name=inputs[0].name,
        output_name=outputs[0].name,
        model_path=path,
        providers=actual_providers,
        device=target,
        metadata=metadata,
        compute_stream=compute_stream,
        precision=precision,
        torch_dtype=torch_dtype,
    )


def end_onnx_profiling(reference: OnnxDepthModel) -> Path:
    """Finish ORT profiling and return the generated JSON path."""
    return Path(reference.session.end_profiling()).resolve()


def estimate_depth_onnx(reference: OnnxDepthModel, image: torch.Tensor) -> torch.Tensor:
    prepared, geometry = prepare_image(image)
    depth = infer_onnx_raw(reference, prepared)
    return restore_depth(depth, geometry).cpu()


def infer_onnx_raw(reference: OnnxDepthModel, prepared: torch.Tensor) -> torch.Tensor:
    """Run ONNX on a prepared BCHW tensor without output normalization."""
    import numpy as np

    if reference.device.type == "cuda":
        # Bind the PyTorch CUDA buffers directly to ORT, removing the external
        # CPU NumPy staging from the hot path; output restoration stays on CUDA
        # until the final ComfyUI tensor copy.
        device_id = reference.device.index or 0
        # I/O binding only receives a raw pointer and shape. prepare_image
        # returns a permuted BCHW view whose strides ORT cannot be told about,
        # so make the source buffer contiguous before the (CPU-origin) transfer
        # instead of forcing a second copy on CUDA.
        prepared = _prepare_binding_tensor(prepared, reference.device, reference.torch_dtype)
        current_stream = torch.cuda.current_stream(reference.device)
        if reference.compute_stream is not None and int(current_stream.cuda_stream) != reference.compute_stream:
            # The ORT session is tied to its creation stream; finish the work
            # that produced this external buffer before handing ORT its pointer.
            current_stream.synchronize()
        depth = torch.empty(
            (prepared.shape[0], prepared.shape[2], prepared.shape[3]),
            device=reference.device,
            dtype=reference.torch_dtype,
        )
        io_binding = reference.session.io_binding()
        io_binding.bind_input(
            name=reference.input_name,
            device_type="cuda",
            device_id=device_id,
            element_type=np.float16 if reference.precision == "fp16" else np.float32,
            shape=tuple(prepared.shape),
            buffer_ptr=prepared.data_ptr(),
        )
        io_binding.bind_output(
            name=reference.output_name,
            device_type="cuda",
            device_id=device_id,
            element_type=np.float16 if reference.precision == "fp16" else np.float32,
            shape=tuple(depth.shape),
            buffer_ptr=depth.data_ptr(),
        )
        io_binding.synchronize_inputs()
        reference.session.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()
        return depth

    input_array = prepared.to(dtype=reference.torch_dtype).detach().cpu().numpy()
    output = reference.session.run(
        [reference.output_name],
        {reference.input_name: input_array},
    )[0]
    if not isinstance(output, np.ndarray):
        raise TypeError(f"ONNX output must be a NumPy array, got {type(output).__name__}.")

    depth = torch.from_numpy(output)
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)
    if depth.ndim != 3:
        raise ValueError(f"Expected ONNX depth output [B,H,W], got {tuple(depth.shape)}.")
    return depth
