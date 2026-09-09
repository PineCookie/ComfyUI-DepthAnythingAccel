"""Shared ComfyUI/plugin path helpers.

Every DepthAccel backend resolves model files the same way: an existing
absolute path wins, then the search falls back through the ComfyUI
``models/depthanything`` folder, the ComfyUI root, the plugin's own
``artifacts`` folder, and finally the current working directory. These helpers
are the permissive resolution used by CLI tools and runtime loaders; the node
layer additionally restricts user-supplied dropdown values to the registered
``depthanything`` folder (see ``nodes.py``).
"""

from pathlib import Path


def _plugin_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_comfyui_root() -> Path:
    """Return the ComfyUI root this plugin is installed under.

    The plugin lives at ``<ComfyUI>/custom_nodes/ComfyUI-DepthAccel``, so the
    depthaccel package sits two levels above the custom_nodes directory.
    """
    return _plugin_root().parents[1]


def default_model_dir(comfyui_root: Path | None = None) -> Path:
    """Return the directory ComfyUI registers for ``depthanything`` models.

    Prefers an already-registered ``folder_paths`` entry; otherwise falls back
    to ``<ComfyUI>/models/depthanything``.
    """
    try:
        import folder_paths

        paths = folder_paths.get_folder_paths("depthanything")
        if paths:
            return Path(paths[0])
    except Exception:
        pass
    return (comfyui_root or default_comfyui_root()) / "models" / "depthanything"


def resolve_model_file(
    model_ref: str | Path,
    *,
    description: str,
    comfyui_root: Path | None = None,
    search_plugin_artifacts: bool = False,
) -> Path:
    """Resolve ``model_ref`` to an existing file, or raise ``FileNotFoundError``.

    Search order: ``<ComfyUI>/models/depthanything/<ref>``, ``<ComfyUI>/<ref>``,
    optionally ``<plugin>/artifacts/<ref>``, then ``cwd/<ref>``. An existing
    absolute path is returned directly. ``description`` names the kind of file
    in the error message (e.g. ``"TensorRT engine"``).
    """
    path = Path(model_ref).expanduser()
    if path.is_absolute() and path.is_file():
        return path.resolve()

    root = comfyui_root or default_comfyui_root()
    candidates = [
        root / "models" / "depthanything" / path,
        root / path,
    ]
    if search_plugin_artifacts:
        candidates.append(_plugin_root() / "artifacts" / path)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{description} was not found. Searched: {searched}")
