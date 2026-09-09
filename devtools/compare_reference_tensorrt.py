"""Compare normalized depth output from the PyTorch and TensorRT backends."""

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPO_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depthaccel.preprocessing import prepare_image, restore_depth
from depthaccel.reference import infer_reference_raw, load_reference_model
from devtools.benchmark_reference import load_image, resize_input
from devtools.benchmark_tensorrt import load_engine, make_executor
from devtools.compare_reference_onnx import compare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--reference-model", default="depth_anything_v2_vitl_fp32.safetensors")
    parser.add_argument("--encoder", choices=["vits", "vitb", "vitl", "vitg"], default="vitl")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.height % 14 or args.width % 14:
        raise ValueError("--height and --width must be divisible by 14")

    device = torch.device("cuda")
    fp16 = args.precision == "fp16"
    image = resize_input(load_image(args.image), args.height, args.width)
    prepared, geometry = prepare_image(image)
    prepared_gpu = prepared.contiguous().to(
        device=device,
        dtype=torch.float16 if fp16 else torch.float32,
    )

    reference = load_reference_model(
        args.reference_model,
        encoder=args.encoder,
        precision=args.precision,
        device=device,
        comfyui_root=COMFYUI_ROOT,
    )
    reference_raw = infer_reference_raw(reference, prepared).cpu()
    reference_output = restore_depth(reference_raw, geometry)

    engine = load_engine(args.engine)
    context, depth = make_executor(engine, prepared_gpu, fp16)
    stream = torch.cuda.Stream(device=device)
    context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize(device)
    candidate_raw = depth.detach().cpu().float()
    candidate_output = restore_depth(candidate_raw, geometry)

    result = {
        "engine": str(args.engine.resolve()),
        "encoder": args.encoder,
        "precision": args.precision,
        "raw": compare(reference_raw.float(), candidate_raw),
        "normalized": compare(reference_output, candidate_output),
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)


if __name__ == "__main__":
    main()
