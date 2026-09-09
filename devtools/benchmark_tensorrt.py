"""Benchmark a dynamic-shape TensorRT engine without starting ComfyUI."""

import argparse
import json
from pathlib import Path
import platform
import sys

import tensorrt as trt
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depthaccel.preprocessing import prepare_image
from depthaccel.trt_logger import get_trt_logger
from devtools.benchmark_reference import load_image, resize_input


SHAPES = {
    "portrait": (784, 1176),
    "landscape": (1176, 784),
    "square": (1022, 1022),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--shapes", choices=sorted(SHAPES), nargs="+", default=["portrait", "landscape", "square"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def load_engine(path: Path) -> trt.ICudaEngine:
    runtime = trt.Runtime(get_trt_logger())
    with path.open("rb") as source:
        engine = runtime.deserialize_cuda_engine(source.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {path}")
    return engine


def make_executor(
    engine: trt.ICudaEngine,
    prepared: torch.Tensor,
    fp16: bool,
) -> tuple[trt.IExecutionContext, torch.Tensor]:
    """Bind a dynamic-shape context to the prepared input and output buffers."""
    context = engine.create_execution_context()
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    batch, channels, height, width = prepared.shape
    context.set_input_shape(input_name, (batch, channels, height, width))

    input_dtype = torch.float16 if fp16 else torch.float32
    depth = torch.empty(
        (batch, height, width),
        device=prepared.device,
        dtype=input_dtype,
    )
    context.set_tensor_address(input_name, prepared.data_ptr())
    context.set_tensor_address(output_name, depth.data_ptr())
    return context, depth


def timed_runs(
    context: trt.IExecutionContext,
    stream: torch.cuda.Stream,
    device: torch.device,
    runs: int,
) -> list[float]:
    """Run the bound context on a non-default stream, timing with CUDA events."""
    timings = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT inference failed.")
        end.record(stream)
        torch.cuda.synchronize(device)
        timings.append(start.elapsed_time(end))
    return timings


def main() -> None:
    args = parse_args()
    if not args.engine.is_file():
        raise FileNotFoundError(args.engine)
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("--warmup must be non-negative and --runs must be positive")

    device = torch.device("cuda")
    engine = load_engine(args.engine)
    input_tensor = engine.get_tensor_name(0)
    fp16 = engine.get_tensor_dtype(input_tensor) == trt.DataType.HALF

    image = resize_input(load_image(args.image), *SHAPES["portrait"])
    stream = torch.cuda.Stream(device=device)
    rows = []
    for shape_name in args.shapes:
        height, width = SHAPES[shape_name]
        resized = resize_input(image, height, width)
        prepared, _ = prepare_image(resized)
        prepared = prepared.contiguous().to(device=device, dtype=torch.float16 if fp16 else torch.float32)

        context, depth = make_executor(engine, prepared, fp16)
        for _ in range(args.warmup):
            context.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize(device)

        timings = timed_runs(context, stream, device, args.runs)

        rows.append(
            {
                "shape": shape_name,
                "height": height,
                "width": width,
                "fp16": fp16,
                "runs_ms": [round(value, 3) for value in timings],
                "mean_ms": round(sum(timings) / len(timings), 3),
                "output_shape": list(depth.shape),
                "output_finite": bool(torch.isfinite(depth).all()),
            }
        )

    result = {
        "engine": str(args.engine.resolve()),
        "engine_bytes": args.engine.stat().st_size,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "tensorrt": trt.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "rows": rows,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.as_json:
        print(payload)
    else:
        for row in rows:
            print(
                f"{row['shape']:<10} {row['height']}x{row['width']} "
                f"mean={row['mean_ms']}ms runs={row['runs_ms']}"
            )


if __name__ == "__main__":
    main()
