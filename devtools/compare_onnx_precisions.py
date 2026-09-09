"""Compare FP32 and FP16 ONNX models with the PyTorch FP32 reference."""

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPO_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depthaccel.onnx_backend import infer_onnx_raw, load_onnx_model
from depthaccel.preprocessing import prepare_image, restore_depth
from depthaccel.reference import infer_reference_raw, load_reference_model
from devtools.benchmark_reference import load_image, resize_input, synchronize
from devtools.compare_reference_onnx import compare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--fp32-model", type=Path, required=True)
    parser.add_argument("--fp16-model", type=Path, required=True)
    parser.add_argument("--reference-model", default="depth_anything_v2_vitl_fp32.safetensors")
    parser.add_argument("--encoder", choices=["vits", "vitb", "vitl", "vitg"], default="vitl")
    parser.add_argument("--height", type=int, default=784)
    parser.add_argument("--width", type=int, default=1176)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    samples = []
    for path in args.images:
        image = resize_input(load_image(path), args.height, args.width)
        prepared, geometry = prepare_image(image)
        samples.append((path, prepared, geometry))

    reference = load_reference_model(
        args.reference_model,
        encoder=args.encoder,
        precision="fp32",
        device=device,
        comfyui_root=COMFYUI_ROOT,
    )
    references = []
    with torch.inference_mode():
        for _, prepared, geometry in samples:
            raw = infer_reference_raw(reference, prepared).cpu()
            references.append((raw, restore_depth(raw, geometry)))
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    rows = []
    for model_path in (args.fp32_model, args.fp16_model):
        model = load_onnx_model(model_path, device="cuda")
        with torch.inference_mode():
            for (image_path, prepared, geometry), (reference_raw, reference_output) in zip(
                samples, references
            ):
                synchronize(device)
                started = time.perf_counter()
                candidate_raw = infer_onnx_raw(model, prepared).cpu()
                synchronize(device)
                elapsed_ms = (time.perf_counter() - started) * 1000
                candidate_output = restore_depth(candidate_raw, geometry)
                rows.append(
                    {
                        "image": image_path.name,
                        "precision": model.precision,
                        "latency_ms": round(elapsed_ms, 3),
                        "raw": compare(reference_raw, candidate_raw),
                        "normalized": compare(reference_output, candidate_output),
                    }
                )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    result = {
        "reference": args.reference_model,
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
