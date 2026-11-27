"""Build a TensorRT engine from the canonical explicit ONNX graph.

The attention-fused ONNX artifacts use the `com.microsoft` operator domain and
are ORT-specific, so TensorRT engines are always built from the explicit
dynamic graphs emitted by `tools/export_onnx.py`.

Engine precision follows the ONNX graph dtype: build from an FP16 explicit
graph for an FP16 engine, or from the FP32 graph for an FP32 engine.
"""

import argparse
import os
from pathlib import Path
import sys
import tempfile
import time

import tensorrt as trt


REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    # Imported as part of the DepthAccel package (ComfyUI node runtime).
    from ..depthaccel.trt_logger import get_trt_logger
except ImportError:
    # Run directly as a CLI script: bootstrap the plugin folder onto sys.path.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from depthaccel.trt_logger import get_trt_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-hw", type=int, nargs=2, default=(14, 14), metavar=("H", "W"))
    parser.add_argument("--opt-hw", type=int, nargs=2, default=(1022, 1022), metavar=("H", "W"))
    parser.add_argument("--max-hw", type=int, nargs=2, default=(2688, 2688), metavar=("H", "W"))
    parser.add_argument("--workspace-gb", type=int, default=4)
    parser.add_argument("--timing-cache", type=Path, help="Persistent timing cache file to speed up repeated builds.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, staging_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".part",
    )
    os.close(fd)
    staging = Path(staging_name)
    try:
        staging.write_bytes(data)
        os.replace(staging, target)
    finally:
        if staging.exists():
            staging.unlink(missing_ok=True)


def validate_profile_bounds(
    min_hw: tuple[int, int],
    opt_hw: tuple[int, int],
    max_hw: tuple[int, int],
    workspace_gb: int = 4,
) -> None:
    """Validate TensorRT optimization profile bounds and workspace size."""
    for label, value in (("min", min_hw), ("opt", opt_hw), ("max", max_hw)):
        if len(value) != 2:
            raise ValueError(f"{label} shape must be a (height, width) pair, got {value}.")
        if value[0] < 14 or value[1] < 14:
            raise ValueError(f"{label} height/width must be at least 14, got {value}.")
        if value[0] % 14 or value[1] % 14:
            raise ValueError(f"{label} height/width must be divisible by 14, got {value}.")
    if not (min_hw[0] <= opt_hw[0] <= max_hw[0] and min_hw[1] <= opt_hw[1] <= max_hw[1]):
        raise ValueError(
            "TensorRT profile needs min <= opt <= max for each side, got "
            f"min={min_hw[0]}x{min_hw[1]}, opt={opt_hw[0]}x{opt_hw[1]}, max={max_hw[0]}x{max_hw[1]}."
        )
    if workspace_gb < 1:
        raise ValueError(f"workspace_gb must be at least 1, got {workspace_gb}.")


def _network_creation_flags(trt_module) -> tuple[int, bool]:
    """Return ``(flags, strongly_typed)`` compatible with TensorRT 10.x and 11.x."""
    if hasattr(trt_module.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        return 1 << int(trt_module.NetworkDefinitionCreationFlag.STRONGLY_TYPED), True
    return 0, False


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    min_hw: tuple[int, int],
    opt_hw: tuple[int, int],
    max_hw: tuple[int, int],
    workspace_gb: int = 4,
    timing_cache: Path | None = None,
    verbose: bool = False,
) -> None:
    validate_profile_bounds(min_hw, opt_hw, max_hw, workspace_gb=workspace_gb)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    started = time.perf_counter()
    print(f"[DepthAnythingAccel] TensorRT: parsing ONNX graph {onnx_path.name}...", flush=True)
    logger = get_trt_logger(verbose)
    builder = trt.Builder(logger)
    network_flags, strongly_typed = _network_creation_flags(trt)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(
            str(parser.get_error(i)) for i in range(parser.num_errors)
        )
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errors}")
    if network.num_inputs != 1:
        raise ValueError(
            f"Expected one ONNX input for TensorRT build, got {network.num_inputs}."
        )
    print(
        f"[DepthAnythingAccel] TensorRT: graph parsed in {time.perf_counter() - started:.1f}s",
        flush=True,
    )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if timing_cache is not None:
        cache_data = timing_cache.read_bytes() if timing_cache.is_file() else b""
        config.set_timing_cache(config.create_timing_cache(cache_data), True)

    input_tensor = network.get_input(0)
    if (
        not strongly_typed
        and input_tensor.dtype == trt.DataType.HALF
        and hasattr(trt.BuilderFlag, "FP16")
    ):
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_tensor.name,
        (1, 3, min_hw[0], min_hw[1]),
        (1, 3, opt_hw[0], opt_hw[1]),
        (1, 3, max_hw[0], max_hw[1]),
    )
    config.add_optimization_profile(profile)

    print(
        f"[DepthAnythingAccel] TensorRT: building engine (profile min={min_hw} "
        f"opt={opt_hw} max={max_hw}); first builds can take minutes...",
        flush=True,
    )
    build_started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}.")
    print(
        f"[DepthAnythingAccel] TensorRT: engine built in {time.perf_counter() - build_started:.1f}s",
        flush=True,
    )

    _atomic_write_bytes(engine_path, bytes(serialized))
    # Record the building TensorRT version so the loader can refuse a silent
    # engine/runtime version mismatch (see depthaccel/tensorrt_backend.py).
    _atomic_write_bytes(
        Path(str(engine_path) + ".version"),
        trt.__version__.encode("utf-8"),
    )

    if timing_cache is not None:
        _atomic_write_bytes(timing_cache, bytes(config.get_timing_cache().serialize()))

    print(f"Input:  {onnx_path.resolve()}")
    print(f"Output: {engine_path.resolve()}")
    print(f"Profile min: {min_hw}, opt: {opt_hw}, max: {max_hw}")
    print(f"Input dtype: {input_tensor.dtype}")
    print(f"Input name: {input_tensor.name}")
    print(
        f"[DepthAnythingAccel] TensorRT: done in {time.perf_counter() - started:.1f}s total",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    validate_profile_bounds(
        tuple(args.min_hw),
        tuple(args.opt_hw),
        tuple(args.max_hw),
        workspace_gb=args.workspace_gb,
    )
    build_engine(
        args.input,
        args.output,
        min_hw=tuple(args.min_hw),
        opt_hw=tuple(args.opt_hw),
        max_hw=tuple(args.max_hw),
        workspace_gb=args.workspace_gb,
        timing_cache=args.timing_cache,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
