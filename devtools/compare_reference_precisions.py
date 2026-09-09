"""Compare DA2/DAD CUDA autocast precisions against an FP32 reference."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPO_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depthaccel.preprocessing import prepare_image, restore_depth
from depthaccel.reference import infer_reference_raw, load_reference_model
from devtools.benchmark_reference import load_image, resize_input, synchronize
from devtools.compare_reference_onnx import compare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--model-path", default="depth_anything_v2_vitl_fp32.safetensors")
    parser.add_argument("--encoder", choices=["vits", "vitb", "vitl", "vitg"], default="vitl")
    parser.add_argument("--model-family", choices=["auto", "da2", "dad"], default="auto")
    parser.add_argument("--height", type=int, default=784)
    parser.add_argument("--width", type=int, default=1176)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def infer(model, prepared: torch.Tensor, geometry) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    synchronize(model.device)
    torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    raw = infer_reference_raw(model, prepared).cpu()
    synchronize(model.device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    peak_mib = torch.cuda.max_memory_allocated(model.device) / 1024**2
    return raw, restore_depth(raw, geometry), elapsed_ms, peak_mib


def main() -> None:
    args = parse_args()
    if args.height < 1 or args.width < 1:
        raise ValueError("Input dimensions must be positive")
    device = torch.device("cuda")
    model = load_reference_model(
        args.model_path,
        encoder=args.encoder,
        precision="fp32",
        device=device,
        comfyui_root=COMFYUI_ROOT,
        model_family=args.model_family,
    )
    rows = []
    with torch.inference_mode():
        for image_path in args.images:
            image = resize_input(load_image(image_path), args.height, args.width)
            prepared, geometry = prepare_image(image)
            model.precision = "fp32"
            reference_raw, reference_output, fp32_ms, fp32_peak = infer(
                model, prepared, geometry
            )
            for precision in ("bf16", "fp16"):
                model.precision = precision
                candidate_raw, candidate_output, elapsed_ms, peak_mib = infer(
                    model, prepared, geometry
                )
                rows.append(
                    {
                        "image": image_path.name,
                        "precision": precision,
                        "fp32_ms": round(fp32_ms, 3),
                        "candidate_ms": round(elapsed_ms, 3),
                        "fp32_peak_mib": round(fp32_peak, 1),
                        "candidate_peak_mib": round(peak_mib, 1),
                        "raw": compare(reference_raw, candidate_raw),
                        "normalized": compare(reference_output, candidate_output),
                    }
                )
    result = {
        "model": str(model.model_path),
        "model_family": model.model_family,
        "encoder": model.encoder,
        "size": [args.height, args.width],
        "rows": rows,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
