"""Convert a DA2/DAD checkpoint to safetensors, optionally lowering precision."""

import argparse
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depthaccel.reference import detect_checkpoint_encoder, load_checkpoint_state_dict


_TARGET_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def convert_checkpoint(
    state_dict: dict[str, torch.Tensor],
    dtype: str,
) -> dict[str, torch.Tensor]:
    """Cast floating-point tensors to the target dtype, leaving int/bool intact."""
    if dtype not in _TARGET_DTYPES:
        raise ValueError(f"Unsupported conversion dtype: {dtype}")
    target = _TARGET_DTYPES[dtype]
    converted = {}
    for key, value in state_dict.items():
        if value.is_floating_point():
            value = value.to(target)
        converted[key] = value.contiguous()
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    from safetensors.torch import save_file

    state_dict = load_checkpoint_state_dict(args.input)
    encoder = detect_checkpoint_encoder(state_dict)
    converted = convert_checkpoint(state_dict, args.dtype)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        converted,
        str(args.output),
        metadata={
            "depthaccel.encoder": encoder,
            "depthaccel.precision": args.dtype,
        },
    )

    print(f"Input:  {args.input.resolve()}")
    print(f"Output: {args.output.resolve()}")
    print(f"Encoder: {encoder}")
    print(f"Precision: {args.dtype}")
    print(f"Tensors: {len(converted)}")


if __name__ == "__main__":
    main()
