"""Benchmark a dynamic DA2 ONNX model without starting ComfyUI."""

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depthaccel.onnx_backend import estimate_depth_onnx, load_onnx_model
from devtools.benchmark_reference import load_image, resize_input, synchronize
from depthaccel.preprocessing import prepare_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("--warmup must be non-negative and --runs must be positive")

    device = torch.device(args.device)
    image = resize_input(load_image(args.image), args.height, args.width)
    _, geometry = prepare_image(image)
    load_started = time.perf_counter()
    model = load_onnx_model(args.model, device=device.type)
    load_seconds = time.perf_counter() - load_started

    with torch.inference_mode():
        for _ in range(args.warmup):
            estimate_depth_onnx(model, image)
        synchronize(device)

        timings = []
        output = None
        for _ in range(args.runs):
            synchronize(device)
            started = time.perf_counter()
            output = estimate_depth_onnx(model, image)
            synchronize(device)
            timings.append((time.perf_counter() - started) * 1000)

    result = {
        "image": str(args.image),
        "model": str(args.model.resolve()),
        "model_bytes": args.model.stat().st_size,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "onnxruntime_providers": list(model.providers),
        "precision": model.precision,
        "original_shape": [image.shape[1], image.shape[2]],
        "inference_shape": [geometry.inference_height, geometry.inference_width],
        "output_shape": list(output.shape),
        "session_load_seconds": round(load_seconds, 3),
        "runs_ms": [round(value, 3) for value in timings],
        "mean_ms": round(sum(timings) / len(timings), 3),
        "peak_vram_mib": None,
        "output_min": round(float(output.min()), 6),
        "output_max": round(float(output.max()), 6),
    }
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
