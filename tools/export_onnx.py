"""Export the local DA2 PyTorch reference to a dynamic-shape ONNX model."""

import argparse
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPO_ROOT.parents[1]

try:
    # Imported as part of the DepthAccel package (ComfyUI node runtime): use
    # package-relative imports so the plugin does not depend on its folder
    # being on sys.path.
    from ..depthaccel.reference import load_reference_model
except ImportError:
    # Run directly as a CLI script (python tools/export_onnx.py), where there
    # is no enclosing package: bootstrap the plugin folder onto sys.path and
    # import depthaccel as a top-level package.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from depthaccel.reference import load_reference_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        default="depth_anything_v2_vitl_fp32.safetensors",
    )
    parser.add_argument("--encoder", choices=["vits", "vitb", "vitl", "vitg"], default="vitl")
    parser.add_argument("--height", type=int, default=560)
    parser.add_argument("--width", type=int, default=784)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--model-family", choices=["auto", "da2", "dad"], default="auto")
    parser.add_argument("--skip-simplify", action="store_true")
    return parser.parse_args()


def export_dynamic_onnx(
    model_path: str,
    output_path: str | Path,
    *,
    encoder: str = "vitl",
    precision: str = "fp32",
    height: int = 560,
    width: int = 784,
    device: str = "cuda",
    model_family: str = "auto",
    opset: int = 17,
    skip_simplify: bool = False,
    reference=None,
) -> Path:
    """Export a DA2 reference to a dynamic-shape ONNX model.

    ``reference`` optionally reuses an already-loaded model instead of reading
    the checkpoint again; it must be a ``depthaccel.reference.ReferenceDepthModel``
    for the same checkpoint and encoder.
    """
    if height < 14 or width < 14:
        raise ValueError("Export dimensions must be at least one 14-pixel patch.")
    if height % 14 or width % 14:
        raise ValueError("Export dimensions must be divisible by 14.")

    output = Path(output_path)
    started = time.perf_counter()
    if reference is None:
        print(
            f"[DepthAnythingAccel] ONNX: loading {model_path} checkpoint "
            f"({encoder}, {precision})...",
            flush=True,
        )
        torch_device = torch.device(device)
        reference = load_reference_model(
            model_path=model_path,
            encoder=encoder,
            precision=precision,
            device=torch_device,
            comfyui_root=COMFYUI_ROOT,
            model_family=model_family,
        )
    else:
        print(
            f"[DepthAnythingAccel] ONNX: using loaded model "
            f"({reference.encoder}, {precision})...",
            flush=True,
        )
    export_dtype = torch.float16 if precision == "fp16" else torch.float32
    reference.model.to(dtype=export_dtype)
    dummy = torch.randn(
        1, 3, height, width, device=reference.device, dtype=export_dtype
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[DepthAnythingAccel] ONNX: tracing graph at {height}x{width} "
        "(tracing + constant folding can take a while)...",
        flush=True,
    )
    torch.onnx.export(
        reference.model,
        (dummy,),
        str(output),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["depth"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "depth": {0: "batch", 1: "height", 2: "width"},
        },
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
    )
    print(
        f"[DepthAnythingAccel] ONNX: tracing finished in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )

    import onnx

    model = onnx.load(str(output), load_external_data=True)
    if not skip_simplify:
        try:
            import onnxslim

            print("[DepthAnythingAccel] ONNX: simplifying graph with onnxslim...", flush=True)
            slimmed = onnxslim.slim(model)
            if isinstance(slimmed, onnx.ModelProto):
                model = slimmed
        except ImportError:
            print("onnxslim is not installed; skipping graph simplification.")
    metadata = {
        "depthaccel.architecture": (
            "distill-any-depth" if reference.model_family == "dad" else "depth-anything-v2"
        ),
        "depthaccel.encoder": encoder,
        "depthaccel.patch_size": "14",
        "depthaccel.precision": precision,
        "depthaccel.source_commit": "Depth-Anything-V2@a561b849ebae10a6f5ef49e26c83cbbcd36c71bf",
    }
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    print("[DepthAnythingAccel] ONNX: validating graph and saving...", flush=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(output))
    print(
        f"Exported: {output.resolve()} "
        f"(total {time.perf_counter() - started:.1f}s)",
        flush=True,
    )
    return output.resolve()


def main() -> None:
    args = parse_args()
    export_dynamic_onnx(
        args.model_path,
        args.output,
        encoder=args.encoder,
        precision=args.precision,
        height=args.height,
        width=args.width,
        device=args.device,
        model_family=args.model_family,
        opset=args.opset,
        skip_simplify=args.skip_simplify,
    )


if __name__ == "__main__":
    main()
