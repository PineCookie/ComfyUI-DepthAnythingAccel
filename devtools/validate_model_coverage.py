"""Validate real DA2/DAD checkpoints against the shared native runtime."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPO_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depthaccel.preprocessing import prepare_image
from depthaccel.reference import (
    MODEL_CONFIGS,
    ReferenceDepthModel,
    detect_checkpoint_encoder,
    infer_reference_raw,
    load_checkpoint_state_dict,
)
from depthaccel.models.da2 import DepthAnythingV2
from devtools.benchmark_reference import load_image, resize_input, synchronize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        nargs=4,
        action="append",
        metavar=("LABEL", "FAMILY", "ENCODER", "PATH"),
        required=True,
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--height", type=int, default=560)
    parser.add_argument("--width", type=int, default=784)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(
    label: str,
    family: str,
    encoder: str,
    path: Path,
    prepared: torch.Tensor,
) -> dict[str, object]:
    state_dict = load_checkpoint_state_dict(path)
    detected = detect_checkpoint_encoder(state_dict)
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    mismatched = sorted(
        key
        for key in set(expected) & set(state_dict)
        if expected[key].shape != state_dict[key].shape
    )
    model.load_state_dict(state_dict, strict=True)
    del state_dict, expected
    model.eval().cuda()
    reference = ReferenceDepthModel(
        model=model,
        device=torch.device("cuda"),
        precision="fp32",
        model_path=path,
        encoder=encoder,
        model_family=family,
    )
    with torch.inference_mode():
        infer_reference_raw(reference, prepared)
        synchronize(reference.device)
        torch.cuda.reset_peak_memory_stats(reference.device)
        started = time.perf_counter()
        depth = infer_reference_raw(reference, prepared)
        synchronize(reference.device)
        elapsed_ms = (time.perf_counter() - started) * 1000
    result = {
        "label": label,
        "family": family,
        "encoder": encoder,
        "detected_encoder": detected,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "checkpoint_keys": len(model.state_dict()),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": mismatched,
        "output_shape": list(depth.shape),
        "output_dtype": str(depth.dtype),
        "output_finite": bool(torch.isfinite(depth).all()),
        "output_min": float(depth.min()),
        "output_max": float(depth.max()),
        "latency_ms": round(elapsed_ms, 3),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }
    del reference, model, depth
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if args.height < 14 or args.width < 14:
        raise ValueError("Validation dimensions must be at least one patch")
    image = resize_input(load_image(args.image), args.height, args.width)
    prepared, geometry = prepare_image(image)
    rows = [
        validate_checkpoint(label, family, encoder, Path(path), prepared)
        for label, family, encoder, path in args.checkpoint
    ]
    result = {
        "image": args.image.name,
        "requested_size": [args.height, args.width],
        "inference_size": [geometry.inference_height, geometry.inference_width],
        "checkpoints": rows,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
