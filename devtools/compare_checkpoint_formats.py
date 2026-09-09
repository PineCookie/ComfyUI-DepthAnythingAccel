"""Compare an official DA2/DAD checkpoint with a converted safetensors file.

Useful to prove a format conversion is lossless before trusting the converted
files in a benchmark or parity matrix. Both files are loaded through the same
normalizing loader used for inference, so key-prefix differences (for example
the DAD Large `backbone.` prefix) are resolved before comparison.
"""

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depthaccel.reference import (
    detect_checkpoint_encoder,
    load_checkpoint_state_dict,
)


def compare_state_dicts(
    official: dict[str, torch.Tensor],
    converted: dict[str, torch.Tensor],
    atol: float = 1e-5,
    rtol: float = 1e-5,
    max_diffs: int = 20,
) -> dict[str, object]:
    """Compare two state dicts by key, shape, dtype, and value."""
    official_keys = set(official)
    converted_keys = set(converted)
    missing = sorted(official_keys - converted_keys)
    unexpected = sorted(converted_keys - official_keys)
    common = sorted(official_keys & converted_keys)

    shape_mismatches: list[tuple[str, list[int], list[int]]] = []
    dtype_mismatches: list[tuple[str, str, str]] = []
    within = 0
    out_of_tolerance = 0
    diffs: list[dict[str, object]] = []
    max_abs = 0.0
    max_rel = 0.0
    worst_key: str | None = None

    for key in common:
        a = official[key]
        b = converted[key]
        if tuple(a.shape) != tuple(b.shape):
            shape_mismatches.append((key, list(a.shape), list(b.shape)))
            continue
        if a.dtype != b.dtype:
            dtype_mismatches.append((key, str(a.dtype), str(b.dtype)))
        a_float = a.to(torch.float32)
        b_float = b.to(torch.float32)
        abs_diff = (a_float - b_float).abs()
        rel_diff = abs_diff / b_float.abs().clamp_min(1e-6)
        key_max_abs = float(abs_diff.max())
        key_max_rel = float(rel_diff.max())
        if key_max_abs > max_abs:
            max_abs = key_max_abs
        if key_max_rel > max_rel:
            max_rel = key_max_rel
            worst_key = key
        if torch.allclose(a_float, b_float, atol=atol, rtol=rtol):
            within += 1
        else:
            out_of_tolerance += 1
            if len(diffs) < max_diffs:
                diffs.append(
                    {
                        "key": key,
                        "dtype": str(a.dtype),
                        "converted_dtype": str(b.dtype),
                        "max_abs_diff": key_max_abs,
                        "max_rel_diff": key_max_rel,
                    }
                )

    return {
        "official_keys": len(official_keys),
        "converted_keys": len(converted_keys),
        "common_keys": len(common),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": [
            {
                "key": key,
                "official_shape": official_shape,
                "converted_shape": converted_shape,
            }
            for key, official_shape, converted_shape in shape_mismatches
        ],
        "dtype_mismatches": [
            {
                "key": key,
                "official_dtype": official_dtype,
                "converted_dtype": converted_dtype,
            }
            for key, official_dtype, converted_dtype in dtype_mismatches[:max_diffs]
        ],
        "within_tolerance": within,
        "out_of_tolerance": out_of_tolerance,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "worst_key": worst_key,
        "sample_diffs": diffs,
        "keys_match": not missing and not unexpected,
        "shapes_match": not shape_mismatches,
        "values_match": out_of_tolerance == 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument(
        "--encoder",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Optional expected encoder shared by both checkpoints.",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--max-diffs", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _detect_encoder(path: Path, state_dict: dict[str, torch.Tensor]) -> str:
    try:
        return detect_checkpoint_encoder(state_dict)
    except ValueError:
        return "unknown"


def main() -> None:
    args = parse_args()
    for path in (args.official, args.converted):
        if not path.is_file():
            raise FileNotFoundError(path)

    official = load_checkpoint_state_dict(args.official)
    converted = load_checkpoint_state_dict(args.converted)
    official_encoder = _detect_encoder(args.official, official)
    converted_encoder = _detect_encoder(args.converted, converted)
    if args.encoder and (
        official_encoder != args.encoder or converted_encoder != args.encoder
    ):
        raise ValueError(
            f"Expected encoder {args.encoder!r}, detected official={official_encoder!r}, "
            f"converted={converted_encoder!r}."
        )
    if (
        official_encoder != "unknown"
        and converted_encoder != "unknown"
        and official_encoder != converted_encoder
    ):
        raise ValueError(
            f"Checkpoints use different encoders: official={official_encoder!r}, "
            f"converted={converted_encoder!r}."
        )

    report = compare_state_dicts(
        official,
        converted,
        atol=args.atol,
        rtol=args.rtol,
        max_diffs=args.max_diffs,
    )
    report["official_path"] = str(args.official.resolve())
    report["converted_path"] = str(args.converted.resolve())
    report["official_encoder"] = official_encoder
    report["converted_encoder"] = converted_encoder

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")

    ok = bool(
        report["keys_match"] and report["shapes_match"] and report["values_match"]
    )
    print(
        f"\nVerdict: {'MATCH' if ok else 'MISMATCH'} "
        f"(keys={report['keys_match']}, shapes={report['shapes_match']}, "
        f"values={report['values_match']})",
        file=sys.stderr,
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
