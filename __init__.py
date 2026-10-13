from pathlib import Path
from typing_extensions import override

import folder_paths
from comfy_api.latest import ComfyExtension, io

from .nodes import (
    DepthAccelBuildTensorrtEngine,
    DepthAccelEstimateDepth,
    DepthAccelEstimateDepthOnnx,
    DepthAccelEstimateDepthTensorrt,
    DepthAccelExportOnnx,
    DepthAccelFuseOnnxAttention,
    DepthAccelLoadOnnxModel,
    DepthAccelLoadReferenceModel,
    DepthAccelLoadTensorrtModel,
)


class DepthAccelExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        folder_paths.add_model_folder_path(
            "depthanything",
            str(Path(folder_paths.models_dir) / "depthanything"),
        )

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DepthAccelLoadReferenceModel,
            DepthAccelEstimateDepth,
            DepthAccelLoadOnnxModel,
            DepthAccelEstimateDepthOnnx,
            DepthAccelLoadTensorrtModel,
            DepthAccelEstimateDepthTensorrt,
            DepthAccelExportOnnx,
            DepthAccelFuseOnnxAttention,
            DepthAccelBuildTensorrtEngine,
        ]


async def comfy_entrypoint() -> DepthAccelExtension:
    return DepthAccelExtension()
