"""Image preprocessing shared by every DepthAccel backend.

ComfyUI IMAGE tensors ([B,H,W,C], 0..1) are normalized and resized to a
14-pixel patch-aligned grid for the model, then the raw depth output is
per-image min-max normalized and restored to the original resolution.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


PATCH_SIZE = 14
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

# Conservative native-resolution budgets in aligned pixels, derived from the
# P0 native-2K OOM finding on a 16 GB GPU: FP32 vitl OOMs near 3.5 MP while the
# FP16 paths comfortably cover the dynamic profile up to 2016x2016. The FP32
# budget keeps headroom below that boundary while still allowing typical
# 1536x1536 (~2.36 MP) canvases. Backends that exceed their budget fail fast
# with an actionable error instead of exhausting VRAM
# (see guard_inference_resolution).
MAX_RESOLUTION_PIXELS = {
    "fp32": 2_621_440,  # ~2.6 MP, covers 1536x1536 canvases
    "fp16": 4_718_592,  # above the 2016x2016 (4.06 MP) dynamic profile
    "bf16": 4_718_592,
}


@dataclass(frozen=True)
class ImageGeometry:
    original_height: int
    original_width: int
    inference_height: int
    inference_width: int


def align_to_patch(size: int, patch_size: int = PATCH_SIZE) -> int:
    if size < 1:
        raise ValueError(f"Image dimensions must be positive, got {size}.")
    return max(patch_size, (size // patch_size) * patch_size)


def prepare_image(
    image: torch.Tensor,
    patch_size: int = PATCH_SIZE,
) -> tuple[torch.Tensor, ImageGeometry]:
    """Prepare a ComfyUI IMAGE tensor for the DA2 model."""
    if image.ndim != 4:
        raise ValueError(f"Expected IMAGE with shape [B,H,W,C], got {tuple(image.shape)}.")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3 image channels, got {image.shape[-1]}.")

    batch, original_height, original_width, _ = image.shape
    if batch < 1:
        raise ValueError("IMAGE batch must not be empty.")

    inference_height = align_to_patch(original_height, patch_size)
    inference_width = align_to_patch(original_width, patch_size)

    image = image.permute(0, 3, 1, 2)
    if (inference_height, inference_width) != (original_height, original_width):
        image = F.interpolate(
            image,
            size=(inference_height, inference_width),
            mode="bilinear",
        )

    mean = image.new_tensor(IMAGE_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(IMAGE_STD).view(1, 3, 1, 1)
    image = (image - mean) / std

    geometry = ImageGeometry(
        original_height=original_height,
        original_width=original_width,
        inference_height=inference_height,
        inference_width=inference_width,
    )
    return image, geometry


def normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim != 3:
        raise ValueError(f"Expected depth with shape [B,H,W], got {tuple(depth.shape)}.")

    # Run the statistics and the division in FP32 even when a backend returned
    # FP16 depth: FP16 epsilon (~9.8e-4) is far too coarse for clamp_min, which
    # would distort near-flat regions, and the min/max themselves only carry
    # FP16 precision.
    depth = depth.float()
    depth_min = depth.amin(dim=(-2, -1), keepdim=True)
    depth_max = depth.amax(dim=(-2, -1), keepdim=True)
    denominator = (depth_max - depth_min).clamp_min(torch.finfo(depth.dtype).eps)
    return ((depth - depth_min) / denominator).clamp(0, 1)


def guard_inference_resolution(
    height: int,
    width: int,
    precision: str,
    *,
    patch_size: int = PATCH_SIZE,
) -> tuple[int, int]:
    """Fail fast when a native resolution would exceed the precision budget.

    Returns the patch-aligned ``(height, width)`` that the backend will infer
    on. Raises ``ValueError`` when the aligned pixel count exceeds
    :data:`MAX_RESOLUTION_PIXELS` for the precision, so the node reports an
    actionable error instead of exhausting VRAM. Precisions without a budget
    (or an explicit smaller budget) are not limited here.
    """
    aligned_height = align_to_patch(height, patch_size)
    aligned_width = align_to_patch(width, patch_size)
    limit = MAX_RESOLUTION_PIXELS.get(precision)
    pixels = aligned_height * aligned_width
    if limit is not None and pixels > limit:
        raise ValueError(
            f"Input {height}x{width} aligns to {aligned_height}x{aligned_width} "
            f"({pixels:,} px), which exceeds the {precision.upper()} budget of "
            f"{limit:,} px. Resize the image, or use an FP16 ONNX/TensorRT "
            "backend for higher native resolutions."
        )
    return aligned_height, aligned_width


def restore_depth(depth: torch.Tensor, geometry: ImageGeometry) -> torch.Tensor:
    """Return a normalized ComfyUI IMAGE tensor at the original size."""
    depth = normalize_depth(depth)
    if depth.shape[-2:] != (geometry.original_height, geometry.original_width):
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=(geometry.original_height, geometry.original_width),
            mode="bilinear",
        ).squeeze(1)

    return depth.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous().float()
