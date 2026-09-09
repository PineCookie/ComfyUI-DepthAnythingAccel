"""Model download helpers for DepthAccel checkpoints.

Known checkpoints (MODEL_SOURCES) are stored in ComfyUI's
``models/depthanything`` folder under their canonical key name — the same name
that appears in the Load node dropdown — so a downloaded file is always
addressable by that single name. A model that is already present locally is
used as-is without importing huggingface_hub; downloads run through the
huggingface_hub cache and are then copied into the model folder, keeping the
folder free of huggingface cache files.

License notes mirror the conservative classification in THIRD_PARTY.md.
"""

import os
from pathlib import Path
import shutil

from .paths import default_model_dir


MODEL_SOURCES: dict[str, dict[str, str]] = {
    # Kijai safetensors — the default download family.
    "depth_anything_v2_vits_fp32.safetensors": {
        "repo_id": "Kijai/DepthAnythingV2-safetensors",
        "filename": "depth_anything_v2_vits_fp32.safetensors",
        "encoder": "vits",
        "license": "Apache-2.0 (upstream Small)",
    },
    "depth_anything_v2_vitb_fp32.safetensors": {
        "repo_id": "Kijai/DepthAnythingV2-safetensors",
        "filename": "depth_anything_v2_vitb_fp32.safetensors",
        "encoder": "vitb",
        "license": "CC-BY-NC-4.0 (upstream Base)",
    },
    "depth_anything_v2_vitl_fp32.safetensors": {
        "repo_id": "Kijai/DepthAnythingV2-safetensors",
        "filename": "depth_anything_v2_vitl_fp32.safetensors",
        "encoder": "vitl",
        "license": "CC-BY-NC-4.0 (upstream Large)",
    },
    # Official DA2 .pth releases.
    "depth_anything_v2_vits.pth": {
        "repo_id": "depth-anything/Depth-Anything-V2-Small",
        "filename": "depth_anything_v2_vits.pth",
        "encoder": "vits",
        "license": "Apache-2.0",
    },
    "depth_anything_v2_vitb.pth": {
        "repo_id": "depth-anything/Depth-Anything-V2-Base",
        "filename": "depth_anything_v2_vitb.pth",
        "encoder": "vitb",
        "license": "CC-BY-NC-4.0",
    },
    "depth_anything_v2_vitl.pth": {
        "repo_id": "depth-anything/Depth-Anything-V2-Large",
        "filename": "depth_anything_v2_vitl.pth",
        "encoder": "vitl",
        "license": "CC-BY-NC-4.0",
    },
    "depth_anything_v2_vitg.pth": {
        "repo_id": "depth-anything/Depth-Anything-V2-Giant",
        "filename": "depth_anything_v2_vitg.pth",
        "encoder": "vitg",
        "license": "CC-BY-NC-4.0",
    },
    # Distill-Any-Depth (state-dict compatible with the shared DA2 runtime).
    # The upstream repository stores these under subfolders named after their
    # size; each is downloaded under the canonical flat key name below.
    "distill-any-depth-small.safetensors": {
        "repo_id": "xingyang1/Distill-Any-Depth",
        "filename": "small/model.safetensors",
        "encoder": "vits",
        "license": "Apache-2.0 (HF model card); see THIRD_PARTY.md",
    },
    "distill-any-depth-base.safetensors": {
        "repo_id": "xingyang1/Distill-Any-Depth",
        "filename": "base/model.safetensors",
        "encoder": "vitb",
        "license": "Apache-2.0 (HF model card); see THIRD_PARTY.md",
    },
    "distill-any-depth-large.safetensors": {
        "repo_id": "xingyang1/Distill-Any-Depth",
        "filename": "large/model.safetensors",
        "encoder": "vitl",
        "license": "Apache-2.0 (HF model card); see THIRD_PARTY.md",
    },
}


def known_model_names() -> list[str]:
    return list(MODEL_SOURCES)


def download_model(
    model_ref: str,
    dest_dir: Path | None = None,
    comfyui_root: Path | None = None,
) -> Path:
    """Return the canonical path of a known checkpoint, downloading if needed.

    ``model_ref`` must be a key from :data:`MODEL_SOURCES`. When the local file
    is already present it is returned without touching huggingface_hub; only an
    actual download imports the hub and copies the cached file into the model
    folder under the canonical key name.
    """
    source = MODEL_SOURCES.get(model_ref)
    if source is None:
        raise ValueError(
            f"Unknown model {model_ref!r}. Use one of the known names: "
            + ", ".join(known_model_names())
        )

    dest = dest_dir or default_model_dir(comfyui_root)
    target = dest / model_ref
    if target.is_file():
        return target.resolve()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "Auto-download needs the huggingface_hub package. Install it with "
            f"'pip install huggingface-hub', or place the file manually at: "
            f"{target}"
        ) from error

    dest.mkdir(parents=True, exist_ok=True)
    # huggingface_hub downloads into its own cache directory; copy the result
    # into the canonical file name so the model folder stays clean and every
    # load path agrees on one file name.
    cached = Path(
        hf_hub_download(repo_id=source["repo_id"], filename=source["filename"])
    )
    if not target.is_file():
        staging = target.with_name(target.name + ".part")
        try:
            shutil.copyfile(cached, staging)
            os.replace(staging, target)
        finally:
            if staging.exists():
                staging.unlink(missing_ok=True)
    return target.resolve()
