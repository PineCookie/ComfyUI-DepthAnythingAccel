"""Compare normalized depth output from the PyTorch and ONNX backends."""

import argparse
import gc
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPO_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depthaccel.onnx_backend import infer_onnx_raw, load_onnx_model
from depthaccel.preprocessing import prepare_image, restore_depth
from depthaccel.reference import infer_reference_raw, load_reference_model
from devtools.benchmark_reference import load_image, resize_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--reference-model", default="depth_anything_v2_vitl_fp32.safetensors")
    parser.add_argument("--encoder", choices=["vits", "vitb", "vitl", "vitg"], default="vitl")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Output shapes differ: reference={tuple(reference.shape)}, candidate={tuple(candidate.shape)}")

    difference = (reference.float() - candidate.float()).abs()
    reference_flat = reference.float().flatten()
    candidate_flat = candidate.float().flatten()
    reference_centered = reference_flat - reference_flat.mean()
    candidate_centered = candidate_flat - candidate_flat.mean()
    correlation = torch.dot(reference_centered, candidate_centered) / (
        reference_centered.norm() * candidate_centered.norm()
    ).clamp_min(torch.finfo(torch.float32).eps)

    return {
        "shape": list(reference.shape),
        "mean_absolute_error": float(difference.mean()),
        "max_absolute_error": float(difference.max()),
        "rmse": float(torch.sqrt((difference.square()).mean())),
        "correlation": float(correlation.clamp(-1, 1)),
        "reference_min": float(reference.min()),
        "reference_max": float(reference.max()),
        "candidate_min": float(candidate.min()),
        "candidate_max": float(candidate.max()),
    }


def main() -> None:
    args = parse_args()
    if args.height < 1 or args.width < 1:
        raise ValueError("Input dimensions must be positive")

    device = torch.device(args.device)
    image = resize_input(load_image(args.image), args.height, args.width)
    prepared, geometry = prepare_image(image)

    with torch.inference_mode():
        reference_model = load_reference_model(
            model_path=args.reference_model,
            encoder=args.encoder,
            precision=args.precision,
            device=device,
            comfyui_root=COMFYUI_ROOT,
        )
        reference_raw = infer_reference_raw(reference_model, prepared).cpu()

    # The two models are evaluated sequentially so the comparison does not
    # require enough VRAM to hold both inference allocators at once.
    del reference_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    onnx_model = load_onnx_model(args.model, device=device.type)
    with torch.inference_mode():
        candidate_raw = infer_onnx_raw(onnx_model, prepared).cpu()

    reference_output = restore_depth(reference_raw, geometry)
    candidate_output = restore_depth(candidate_raw, geometry)

    result = {
        "image": str(args.image),
        "onnx_model": str(args.model.resolve()),
        "reference_model": args.reference_model,
        "device": str(device),
        "onnxruntime_providers": list(onnx_model.providers),
        "raw_comparison": compare(reference_raw, candidate_raw),
        "normalized_comparison": compare(reference_output, candidate_output),
    }
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
