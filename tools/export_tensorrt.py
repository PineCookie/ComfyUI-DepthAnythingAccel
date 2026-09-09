"""Build a TensorRT engine from the canonical explicit ONNX graph.

The attention-fused ONNX artifacts use the `com.microsoft` operator domain and
are ORT-specific, so TensorRT engines are always built from the explicit
dynamic graphs emitted by `tools/export_onnx.py`.

Engine precision follows the ONNX graph dtype: build from an FP16 explicit
graph for an FP16 engine, or from the FP32 graph for an FP32 engine.
"""

import argparse
from pathlib import Path
import sys
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
    parser.add_argument("--max-hw", type=int, nargs=2, default=(2016, 2016), metavar=("H", "W"))
    parser.add_argument("--workspace-gb", type=int, default=4)
    parser.add_argument("--timing-cache", type=Path, help="Persistent timing cache file to speed up repeated builds.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


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
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    started = time.perf_counter()
    print(f"[DepthAnythingAccel] TensorRT: parsing ONNX graph {onnx_path.name}...", flush=True)
    logger = get_trt_logger(verbose)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with onnx_path.open("rb") as source:
        if not parser.parse(source.read()):
            errors = "\n".join(
                parser.get_error(i) for i in range(parser.num_errors)
            )
            raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errors}")
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

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    # Record the building TensorRT version so the loader can refuse a silent
    # engine/runtime version mismatch (see depthaccel/tensorrt_backend.py).
    Path(str(engine_path) + ".version").write_text(trt.__version__, encoding="utf-8")

    if timing_cache is not None:
        timing_cache.parent.mkdir(parents=True, exist_ok=True)
        timing_cache.write_bytes(config.get_timing_cache().serialize())

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
    for label, value in (("min", args.min_hw), ("opt", args.opt_hw), ("max", args.max_hw)):
        if value[0] % 14 or value[1] % 14:
            raise ValueError(f"--{label}-hw must be divisible by 14, got {value}.")
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
