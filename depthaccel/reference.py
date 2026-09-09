"""PyTorch reference backend for DA2/DAD depth estimation.

Loads a DA2 or architecture-compatible DAD checkpoint into the repository-owned
``models.da2`` architecture and runs it with per-sample native-resolution
inference. FP32 is the conservative default; FP16/BF16 use CUDA autocast.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch

from .paths import default_comfyui_root, resolve_model_file
from .preprocessing import prepare_image, restore_depth


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
EMBED_DIM_TO_ENCODER = {384: "vits", 768: "vitb", 1024: "vitl", 1536: "vitg"}


@dataclass
class ReferenceDepthModel:
    model: torch.nn.Module
    device: torch.device
    precision: str
    model_path: Path
    encoder: str
    model_family: str


def _resolve_device() -> torch.device:
    import comfy.model_management as model_management

    return model_management.get_torch_device()


def load_checkpoint_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        loaded = load_file(str(path), device="cpu")
    else:
        loaded = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(loaded, dict) and "state_dict" in loaded:
            loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise TypeError(f"Checkpoint must contain a state dict, got {type(loaded).__name__}.")
    return normalize_checkpoint_state_dict(loaded)


def normalize_checkpoint_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize the alternate DAD Large backbone prefix without relaxing strict load."""
    has_pretrained = any(key.startswith("pretrained.") for key in state_dict)
    has_backbone = any(key.startswith("backbone.") for key in state_dict)
    if not has_backbone:
        return state_dict
    if has_pretrained:
        raise ValueError("Checkpoint mixes 'backbone.' and 'pretrained.' key prefixes.")

    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("backbone.blocks.0."):
            target = "pretrained.blocks." + key[len("backbone.blocks.0.") :]
        elif key.startswith("backbone."):
            target = "pretrained." + key[len("backbone.") :]
        else:
            target = key
        if target in normalized:
            raise ValueError(f"Checkpoint key normalization collision: {target!r}.")
        normalized[target] = value
    return normalized


def detect_checkpoint_encoder(state_dict: dict[str, torch.Tensor]) -> str:
    key = "pretrained.patch_embed.proj.weight"
    weight = state_dict.get(key)
    if weight is None or not hasattr(weight, "shape") or len(weight.shape) != 4:
        raise ValueError(f"Cannot identify DA2 encoder: missing or invalid {key!r}.")
    encoder = EMBED_DIM_TO_ENCODER.get(int(weight.shape[0]))
    if encoder is None:
        raise ValueError(
            f"Cannot identify DA2 encoder from embedding dimension {int(weight.shape[0])}."
        )
    return encoder


def resolve_model_family(model_path: Path, model_family: str) -> str:
    if model_family not in {"auto", "da2", "dad"}:
        raise ValueError(f"Unsupported model family: {model_family}")
    if model_family != "auto":
        return model_family
    name = str(model_path).lower()
    return "dad" if "distill" in name or "dad" in name else "da2"


def load_reference_model(
    model_path: str,
    encoder: str | None = "auto",
    precision: str = "fp32",
    device: torch.device | None = None,
    comfyui_root: Path | None = None,
    model_family: str = "auto",
) -> ReferenceDepthModel:
    """Load a checkpoint, detecting the encoder when ``encoder`` is auto/None."""
    if encoder not in (None, "auto") and encoder not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported DA2 encoder: {encoder}")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported precision: {precision}")

    root = comfyui_root or default_comfyui_root()
    path = resolve_model_file(
        model_path,
        description="Depth model",
        comfyui_root=root,
    )
    resolved_family = resolve_model_family(path, model_family)
    from .models.da2 import DepthAnythingV2

    state_dict = load_checkpoint_state_dict(path)
    detected_encoder = detect_checkpoint_encoder(state_dict)
    if encoder in (None, "auto"):
        encoder = detected_encoder
    elif detected_encoder != encoder:
        raise ValueError(
            f"Checkpoint is DA2 {detected_encoder}, but loader encoder is {encoder}. "
            f"Select encoder={detected_encoder!r}."
        )

    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "DA2 checkpoint does not match the reference architecture: "
            f"missing={missing_keys[:5]}, unexpected={unexpected_keys[:5]}"
        )

    target_device = device or _resolve_device()
    model.eval().to(target_device)
    return ReferenceDepthModel(
        model=model,
        device=target_device,
        precision=precision,
        model_path=path,
        encoder=encoder,
        model_family=resolved_family,
    )


def estimate_depth(reference: ReferenceDepthModel, image: torch.Tensor) -> torch.Tensor:
    prepared, geometry = prepare_image(image)
    depth = infer_reference_raw(reference, prepared)
    return restore_depth(depth, geometry)


def infer_reference_raw(
    reference: ReferenceDepthModel,
    prepared: torch.Tensor,
) -> torch.Tensor:
    """Run DA2 on a prepared BCHW tensor without output normalization."""
    outputs = []
    autocast = nullcontext()
    if reference.precision != "fp32" and reference.device.type == "cuda":
        dtype = torch.float16 if reference.precision == "fp16" else torch.bfloat16
        autocast = torch.autocast("cuda", dtype=dtype)

    with autocast:
        for sample in prepared:
            depth = reference.model(sample.unsqueeze(0).to(reference.device))
            outputs.append(depth.detach().cpu())

    return torch.cat(outputs, dim=0)
