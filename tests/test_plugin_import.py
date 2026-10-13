import asyncio
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPO_ROOT.parents[1]
PACKAGE_NAME = "comfyui_depthaccel_test_package"


def _load_extension(package_name: str = PACKAGE_NAME):
    if str(COMFYUI_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFYUI_ROOT))

    spec = importlib.util.spec_from_file_location(
        package_name,
        REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return asyncio.run(module.comfy_entrypoint())


class PluginImportTests(unittest.TestCase):
    def test_new_api_entrypoint_exposes_reference_and_onnx_nodes(self):
        extension = _load_extension()
        nodes = asyncio.run(extension.get_node_list())
        node_ids = [node.define_schema().node_id for node in nodes]

        self.assertEqual(
            node_ids,
            [
                "DepthAccelLoadReferenceModel",
                "DepthAccelEstimateDepth",
                "DepthAccelLoadOnnxModel",
                "DepthAccelEstimateDepthOnnx",
                "DepthAccelLoadTensorrtModel",
                "DepthAccelEstimateDepthTensorrt",
                "DepthAccelExportOnnx",
                "DepthAccelFuseOnnxAttention",
                "DepthAccelBuildTensorrtEngine",
            ],
        )

    def test_plugin_loads_without_optional_tensorrt(self):
        # tensorrt is optional; a None entry makes any import of it fail.
        with mock.patch.dict(sys.modules, {"tensorrt": None}):
            extension = _load_extension("comfyui_depthaccel_no_trt_package")
            nodes = asyncio.run(extension.get_node_list())
        self.assertEqual(len(nodes), 9)

    def test_conversion_nodes_have_outputs_so_they_execute(self):
        # ComfyUI only runs nodes that contribute to the graph's outputs; a
        # conversion node without outputs would never execute.
        extension = _load_extension()
        nodes = asyncio.run(extension.get_node_list())
        schemas = {node.define_schema().node_id: node.define_schema() for node in nodes}
        for node_id in (
            "DepthAccelExportOnnx",
            "DepthAccelFuseOnnxAttention",
            "DepthAccelBuildTensorrtEngine",
        ):
            schema = schemas[node_id]
            self.assertTrue(schema.outputs, f"{node_id} must have outputs")
            # Conversion nodes are terminal roots: ComfyUI executes them (and
            # their ancestors) even when their handle is not connected onward.
            self.assertTrue(
                schema.is_output_node, f"{node_id} must be an output node"
            )


if __name__ == "__main__":
    unittest.main()
