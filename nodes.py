import os
from pathlib import Path as _Path

from comfy_api.latest import io

from .depthaccel.onnx_backend import estimate_depth_onnx, load_onnx_model
from .depthaccel.preprocessing import guard_inference_resolution
from .depthaccel.reference import estimate_depth, load_reference_model
from .depthaccel.tensorrt_backend import estimate_depth_tensorrt, load_tensorrt_model


DepthAccelModelIO = io.Custom("DEPTHACCEL_MODEL")
DepthAccelOnnxModelIO = io.Custom("DEPTHACCEL_ONNX_MODEL")
DepthAccelTensorrtModelIO = io.Custom("DEPTHACCEL_TENSORRT_MODEL")

_MODEL_FOLDER = "depthanything"


def _model_folder_dirs() -> list[_Path]:
    """Registered depthanything folders, falling back to the ComfyUI default."""
    try:
        import folder_paths

        paths = folder_paths.get_folder_paths(_MODEL_FOLDER)
        if paths:
            return [_Path(p) for p in paths]
    except Exception:
        pass
    return [_Path(__file__).resolve().parents[2] / "models" / _MODEL_FOLDER]


def _model_folder_dir() -> str:
    """Return the first registered depthanything model folder."""
    return str(_model_folder_dirs()[0])


def _model_folder_files(*suffixes: str) -> list[str]:
    """List model filenames currently present in the depthanything folders."""
    try:
        import folder_paths

        files = folder_paths.get_filename_list(_MODEL_FOLDER)
    except Exception:
        files = []
        for folder in _model_folder_dirs():
            if folder.is_dir():
                files += [p.name for p in folder.iterdir() if p.is_file()]
    if not suffixes:
        return sorted(set(files))
    lowered = tuple(s.lower() for s in suffixes)
    return sorted(f for f in files if f.lower().endswith(lowered))


def _default_option(options: list[str], preferred: str) -> str:
    if not options:
        return ""
    return preferred if preferred in options else options[0]


def _preferred_option(options: list[str], *preferred: str) -> str:
    """Return the first preferred value present in ``options``, else options[0].

    Used for dropdown defaults so the selected file is a real one in a sensible
    order instead of whatever sorts first — the ORT-fused graph
    (``..._fused.onnx``) is preferred over the explicit ``..._dynamic.onnx``.
    """
    for name in preferred:
        if name in options:
            return name
    return options[0] if options else ""


# Encoder tier tokens in ascending model size, for ordering the dropdown.
_MODEL_TIER_TOKENS = (
    ("vits", 0), ("small", 0),
    ("vitb", 1), ("base", 1),
    ("vitl", 2), ("large", 2),
    ("vitg", 3), ("giant", 3),
)


def _model_sort_key(name: str) -> tuple[int, int, str]:
    """Sort key: model family (DA2 before DAD), then size tier, then name.

    Unrecognized names sort after every known tier, so manual checkpoints stay
    grouped by size rather than sprinkled alphabetically across the list.
    """
    lowered = name.lower()
    family = 1 if ("distill" in lowered or lowered.startswith("dad")) else 0
    tier = next((rank for token, rank in _MODEL_TIER_TOKENS if token in lowered), 4)
    return (family, tier, lowered)


def _resolve_precision(precision: str) -> str:
    """Resolve the Load node's ``auto`` precision to an actual value.

    FP16 autocast is the accelerated reference mode on CUDA; FP32 is the
    conservative default everywhere else.
    """
    if precision != "auto":
        return precision
    try:
        import torch

        if torch.cuda.is_available():
            return "fp16"
    except Exception:
        pass
    return "fp32"


def _checkpoint_options() -> list[str]:
    """Existing checkpoints plus known auto-downloadable models.

    Grouped by family then size tier (small/base/large/giant), so the DA2 and
    DAD entries of the same size sit next to each other instead of being
    scattered alphabetically.
    """
    from .depthaccel.download import known_model_names

    return sorted(
        set(_model_folder_files(".pth", ".safetensors")) | set(known_model_names()),
        key=_model_sort_key,
    )


def _onnx_options() -> list[str]:
    return _model_folder_files(".onnx")


def _engine_options() -> list[str]:
    return _model_folder_files(".trt", ".engine", ".plan")


def _is_within_folder(path: _Path) -> bool:
    """True if ``path`` (resolving symlinks) stays inside a registered folder.

    Mirrors ComfyUI's own ``folder_paths.is_within_directory`` containment
    check so a crafted ``..``/absolute path can never read outside the model
    folder even if ``get_full_path_or_raise`` were given a hostile name.
    """
    path = path.resolve()
    for folder in _model_folder_dirs():
        try:
            folder_real = folder.resolve()
            if os.path.commonpath([str(folder_real), str(path)]) == str(folder_real):
                return True
        except ValueError:
            continue
    return False


def _require_in_folder(path: _Path, suffixes: tuple[str, ...]) -> _Path:
    """Return ``path`` resolved, after asserting it is a real file, inside a
    registered depthanything folder, with an allowed extension."""
    resolved = _Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Model file not found: {resolved} (expected inside "
            f"{_model_folder_dir()})."
        )
    if suffixes and resolved.suffix.lower() not in suffixes:
        raise ValueError(
            f"Expected a {'/'.join(suffixes)} file, got {resolved.name!r}."
        )
    if not _is_within_folder(resolved):
        raise FileNotFoundError(
            f"Refusing to load {str(path)!r}: it resolves outside the "
            f"ComfyUI/models/depthanything model folder."
        )
    return resolved


def _resolve_existing(name: str, suffixes: tuple[str, ...]) -> _Path:
    """Resolve a dropdown value to an existing file inside the model folder.

    Uses the official ``folder_paths.get_full_path_or_raise`` resolution (which
    only looks in the registered folders) plus a realpath containment guard, so
    a value that is not already a file in ``models/depthanything`` is rejected
    instead of being searched across cwd/artifacts/arbitrary absolute paths.
    """
    import folder_paths

    full = folder_paths.get_full_path_or_raise(_MODEL_FOLDER, name)
    return _require_in_folder(_Path(full), suffixes)


def _resolve_checkpoint(model: str) -> _Path:
    """Resolve a checkpoint dropdown value inside the model folder.

    Known-but-missing models are auto-downloaded into ``models/depthanything``
    first; anything else must already be a file inside that folder.
    """
    from .depthaccel.download import MODEL_SOURCES, download_model

    if model in MODEL_SOURCES:
        path = download_model(model)
        return _require_in_folder(_Path(path), (".pth", ".safetensors"))
    return _resolve_existing(model, (".pth", ".safetensors"))


class DepthAccelLoadReferenceModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        options = _checkpoint_options()
        return io.Schema(
            node_id="DepthAccelLoadReferenceModel",
            display_name="Load Depth Model (PyTorch)",
            category="DepthAnythingAccel/Reference",
            description=(
                "Loads a DA2 or architecture-compatible DAD checkpoint from "
                "ComfyUI/models/depthanything into DepthAccel's runtime. The "
                "encoder is detected from the checkpoint."
            ),
            inputs=[
                io.Combo.Input(
                    "model",
                    options=options,
                    default=_preferred_option(
                        options,
                        "depth_anything_v2_vitl_fp32.safetensors",
                        "depth_anything_v2_vitl.pth",
                    ),
                    tooltip=(
                        "Checkpoint in ComfyUI/models/depthanything. A known "
                        "model that is not present is downloaded automatically."
                    ),
                ),
                io.Combo.Input(
                    "precision",
                    options=["auto", "fp32", "fp16", "bf16"],
                    default="auto",
                    tooltip=(
                        "auto uses FP16 autocast on CUDA and FP32 elsewhere; "
                        "the loaded checkpoint is always kept in its native "
                        "dtype."
                    ),
                ),
            ],
            outputs=[DepthAccelModelIO.Output("model")],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model: str,
        precision: str,
    ) -> io.NodeOutput:
        path = _resolve_checkpoint(model)
        resolved_precision = _resolve_precision(precision)
        reference = load_reference_model(
            model_path=str(path),
            precision=resolved_precision,
        )
        return io.NodeOutput(reference)


class DepthAccelEstimateDepth(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DepthAccelEstimateDepth",
            display_name="Estimate Depth (PyTorch)",
            category="DepthAnythingAccel/Reference",
            description=(
                "Runs the PyTorch DA2 reference at native aspect ratio, "
                "aligning inference height and width to patch size 14."
            ),
            inputs=[
                DepthAccelModelIO.Input("model"),
                io.Image.Input("image"),
            ],
            outputs=[io.Image.Output("depth")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, image: "io.Image.Type") -> io.NodeOutput:
        if getattr(image, "ndim", None) == 4:
            guard_inference_resolution(
                int(image.shape[1]), int(image.shape[2]), model.precision
            )
        return io.NodeOutput(estimate_depth(model, image))


class DepthAccelLoadOnnxModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        options = _onnx_options()
        return io.Schema(
            node_id="DepthAccelLoadOnnxModel",
            display_name="Load Depth Model (ONNX)",
            category="DepthAnythingAccel/ONNX",
            description=(
                "Loads a dynamic-shape DA2 ONNX model from "
                "ComfyUI/models/depthanything with ONNX Runtime CUDA."
            ),
            inputs=[
                io.Combo.Input(
                    "model",
                    options=options,
                    default=_preferred_option(
                        options,
                        "da2_vitl_fp16_dynamic_fused.onnx",
                        "da2_vitl_fp16_dynamic.onnx",
                        "da2_vitb_fp16_dynamic_fused.onnx",
                        "da2_vitb_fp16_dynamic.onnx",
                        "da2_vits_fp16_dynamic_fused.onnx",
                        "da2_vits_fp16_dynamic.onnx",
                        "dad_vitl_fp16_dynamic_fused.onnx",
                        "dad_vitl_fp16_dynamic.onnx",
                        "dad_vitb_fp16_dynamic_fused.onnx",
                        "dad_vitb_fp16_dynamic.onnx",
                        "dad_vits_fp16_dynamic_fused.onnx",
                        "dad_vits_fp16_dynamic.onnx",
                    ),
                    tooltip="ONNX model in ComfyUI/models/depthanything.",
                ),
                io.Combo.Input(
                    "device",
                    options=["cuda", "cpu"],
                    default="cuda",
                ),
            ],
            outputs=[DepthAccelOnnxModelIO.Output("onnx_model")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model: str, device: str) -> io.NodeOutput:
        path = _resolve_existing(model, (".onnx",))
        return io.NodeOutput(
            load_onnx_model(
                model_path=str(path),
                device=device,
            )
        )


class DepthAccelEstimateDepthOnnx(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DepthAccelEstimateDepthOnnx",
            display_name="Estimate Depth (ONNX)",
            category="DepthAnythingAccel/ONNX",
            description=(
                "Runs a dynamic-shape DA2 ONNX model at native aspect ratio, "
                "aligning inference height and width to patch size 14."
            ),
            inputs=[
                DepthAccelOnnxModelIO.Input("onnx_model"),
                io.Image.Input("image"),
            ],
            outputs=[io.Image.Output("depth")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, onnx_model, image: "io.Image.Type") -> io.NodeOutput:
        if getattr(image, "ndim", None) == 4:
            guard_inference_resolution(
                int(image.shape[1]), int(image.shape[2]), onnx_model.precision
            )
        return io.NodeOutput(estimate_depth_onnx(onnx_model, image))


class DepthAccelLoadTensorrtModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        options = _engine_options()
        return io.Schema(
            node_id="DepthAccelLoadTensorrtModel",
            display_name="Load Depth Model (TensorRT)",
            category="DepthAnythingAccel/TensorRT",
            description=(
                "Loads a dynamic-shape TensorRT engine from "
                "ComfyUI/models/depthanything."
            ),
            inputs=[
                io.Combo.Input(
                    "model",
                    options=options,
                    default=_preferred_option(
                        options,
                        "da2_vitl_fp16_dynamic.trt",
                        "da2_vitb_fp16_dynamic.trt",
                        "da2_vits_fp16_dynamic.trt",
                    ),
                    tooltip="TensorRT engine in ComfyUI/models/depthanything.",
                ),
            ],
            outputs=[DepthAccelTensorrtModelIO.Output("trt_model")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model: str) -> io.NodeOutput:
        path = _resolve_existing(model, (".trt", ".engine", ".plan"))
        return io.NodeOutput(
            load_tensorrt_model(
                model_path=str(path),
                device="cuda",
            )
        )


class DepthAccelEstimateDepthTensorrt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DepthAccelEstimateDepthTensorrt",
            display_name="Estimate Depth (TensorRT)",
            category="DepthAnythingAccel/TensorRT",
            description=(
                "Runs a dynamic-shape TensorRT engine at native aspect ratio, "
                "aligning inference height and width to patch size 14."
            ),
            inputs=[
                DepthAccelTensorrtModelIO.Input("trt_model"),
                io.Image.Input("image"),
            ],
            outputs=[io.Image.Output("depth")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, trt_model, image: "io.Image.Type") -> io.NodeOutput:
        return io.NodeOutput(estimate_depth_tensorrt(trt_model, image))


class DepthAccelExportOnnx(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DepthAccelExportOnnx",
            display_name="Export to ONNX",
            category="DepthAnythingAccel/Convert",
            description=(
                "Exports the loaded PyTorch model to a dynamic-shape ONNX graph "
                "in ComfyUI/models/depthanything and returns it as an ONNX "
                "model handle. The encoder and checkpoint come from the input "
                "model. Requires the optional onnxruntime dependency to return "
                "the handle."
            ),
            inputs=[
                DepthAccelModelIO.Input("model"),
                io.Combo.Input(
                    "precision",
                    options=["fp32", "fp16"],
                    default="fp16",
                ),
                io.Combo.Input(
                    "device",
                    options=["cuda", "cpu"],
                    default="cuda",
                ),
                io.Int.Input(
                    "height",
                    display_name="Trace height",
                    default=560,
                    min=14,
                    step=14,
                    tooltip=(
                        "Trace size only — inference stays dynamic for any "
                        "resolution. Smaller values export faster."
                    ),
                ),
                io.Int.Input(
                    "width",
                    display_name="Trace width",
                    default=784,
                    min=14,
                    step=14,
                    tooltip=(
                        "Trace size only — inference stays dynamic for any "
                        "resolution. Smaller values export faster."
                    ),
                ),
            ],
            outputs=[DepthAccelOnnxModelIO.Output("onnx_model")],
            is_experimental=True,
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        precision: str,
        device: str,
        height: int,
        width: int,
    ) -> io.NodeOutput:
        from .tools.export_onnx import export_dynamic_onnx

        # Name by family + encoder so a DAD and a DA2 checkpoint with the same
        # architecture (both vitl for large) never collide or masquerade as
        # each other: da2_vitl_... vs dad_vitl_...
        output = (
            _Path(_model_folder_dir())
            / f"{model.model_family}_{model.encoder}_{precision}_dynamic.onnx"
        )
        export_dynamic_onnx(
            str(model.model_path),
            output,
            encoder=model.encoder,
            precision=precision,
            height=height,
            width=width,
            reference=model,
        )
        try:
            onnx_model = load_onnx_model(str(output), device=device)
        except Exception as error:
            raise RuntimeError(
                f"Exported {output}, but it could not be loaded as an ONNX "
                f"model handle ({error}). Install onnxruntime-gpu to chain it "
                "directly."
            ) from error
        return io.NodeOutput(onnx_model)


class DepthAccelFuseOnnxAttention(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DepthAccelFuseOnnxAttention",
            display_name="Fuse ONNX Attention",
            category="DepthAnythingAccel/Convert",
            description=(
                "Rewrites an explicit-graph ONNX model into ONNX Runtime's "
                "fused-attention graph (writes ``<name>_fused.onnx`` next to "
                "the source) and returns the fused model handle. The encoder "
                "is read from the graph metadata."
            ),
            inputs=[DepthAccelOnnxModelIO.Input("onnx_model")],
            outputs=[DepthAccelOnnxModelIO.Output("onnx_model")],
            is_experimental=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, onnx_model) -> io.NodeOutput:
        from .tools.fuse_onnx_attention import fuse_attention_model

        source = _Path(onnx_model.model_path)
        output = _Path(_model_folder_dir()) / f"{source.stem}_fused.onnx"
        fuse_attention_model(source, output)
        fused_model = load_onnx_model(str(output), device=onnx_model.device.type)
        return io.NodeOutput(fused_model)


class DepthAccelBuildTensorrtEngine(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DepthAccelBuildTensorrtEngine",
            display_name="Build TensorRT Engine",
            category="DepthAnythingAccel/Convert",
            description=(
                "Builds a dynamic-shape TensorRT engine from an explicit-graph "
                "ONNX model (writes a ``.trt`` engine next to the source) and "
                "returns it as a TensorRT model handle. Requires the optional "
                "tensorrt dependency."
            ),
            inputs=[
                DepthAccelOnnxModelIO.Input("onnx_model"),
                io.Int.Input("min_h", display_name="Min H", default=14, min=14, step=14),
                io.Int.Input("min_w", display_name="Min W", default=14, min=14, step=14),
                io.Int.Input("opt_h", display_name="Opt H", default=1022, min=14, step=14),
                io.Int.Input("opt_w", display_name="Opt W", default=1022, min=14, step=14),
                io.Int.Input("max_h", display_name="Max H", default=2016, min=14, step=14),
                io.Int.Input("max_w", display_name="Max W", default=2016, min=14, step=14),
                io.Int.Input(
                    "workspace_gb",
                    display_name="Workspace GB",
                    default=4,
                    min=1,
                ),
            ],
            outputs=[DepthAccelTensorrtModelIO.Output("trt_model")],
            is_experimental=True,
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        onnx_model,
        min_h: int,
        min_w: int,
        opt_h: int,
        opt_w: int,
        max_h: int,
        max_w: int,
        workspace_gb: int,
    ) -> io.NodeOutput:
        from .tools.export_tensorrt import build_engine

        for label, value in (
            ("min", (min_h, min_w)),
            ("opt", (opt_h, opt_w)),
            ("max", (max_h, max_w)),
        ):
            if value[0] % 14 or value[1] % 14:
                raise ValueError(
                    f"{label} height/width must be divisible by 14, got {value}."
                )

        source = _Path(onnx_model.model_path)
        if source.stem.endswith("_fused"):
            raise ValueError(
                "Build TensorRT Engine needs the explicit (non-fused) ONNX "
                "graph; feed it the unfused model."
            )
        output = _Path(_model_folder_dir()) / f"{source.stem}.trt"
        build_engine(
            source,
            output,
            min_hw=(min_h, min_w),
            opt_hw=(opt_h, opt_w),
            max_hw=(max_h, max_w),
            workspace_gb=workspace_gb,
        )
        engine_model = load_tensorrt_model(str(output), device="cuda")
        return io.NodeOutput(engine_model)
