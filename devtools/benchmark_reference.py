"""Benchmark the local Kijai DA2 PyTorch implementation without ComfyUI."""

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPO_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(COMFYUI_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFYUI_ROOT))

from depthaccel.preprocessing import prepare_image
from depthaccel.reference import estimate_depth, load_reference_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        default="depth_anything_v2_vitl_fp32.safetensors",
    )
    parser.add_argument("--encoder", choices=["vits", "vitb", "vitl", "vitg"], default="vitl")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def load_image(path: Path) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    with Image.open(path) as source:
        rgb = source.convert("RGB")
        array = np.asarray(rgb)
    return torch.from_numpy(array.copy()).float().div_(255).unsqueeze(0)


def resize_input(image: torch.Tensor, height: int | None, width: int | None) -> torch.Tensor:
    if (height is None) != (width is None):
        raise ValueError("--height and --width must be provided together")
    if height is None:
        return image
    if height < 1 or width < 1:
        raise ValueError("Input dimensions must be positive")
    return F.interpolate(
        image.permute(0, 3, 1, 2),
        size=(height, width),
        mode="bilinear",
    ).permute(0, 2, 3, 1).contiguous()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("--warmup must be non-negative and --runs must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    device = torch.device(args.device)
    image = resize_input(load_image(args.image), args.height, args.width)
    _, geometry = prepare_image(image)

    load_started = time.perf_counter()
    reference = load_reference_model(
        model_path=args.model_path,
        encoder=args.encoder,
        precision=args.precision,
        device=device,
        comfyui_root=COMFYUI_ROOT,
    )
    load_seconds = time.perf_counter() - load_started
    from comfyui_version import __version__ as comfyui_version

    with torch.inference_mode():
        for _ in range(args.warmup):
            estimate_depth(reference, image)
        synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        timings = []
        output = None
        for _ in range(args.runs):
            synchronize(device)
            started = time.perf_counter()
            output = estimate_depth(reference, image)
            synchronize(device)
            timings.append((time.perf_counter() - started) * 1000)

    result = {
        "image": str(args.image),
        "model": str(reference.model_path),
        "model_bytes": reference.model_path.stat().st_size,
        "comfyui": comfyui_version,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "encoder": args.encoder,
        "precision": args.precision,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "original_shape": [image.shape[1], image.shape[2]],
        "inference_shape": [geometry.inference_height, geometry.inference_width],
        "output_shape": list(output.shape),
        "model_load_seconds": round(load_seconds, 3),
        "runs_ms": [round(value, 3) for value in timings],
        "mean_ms": round(sum(timings) / len(timings), 3),
        "peak_vram_mib": (
            round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
            if device.type == "cuda"
            else None
        ),
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
