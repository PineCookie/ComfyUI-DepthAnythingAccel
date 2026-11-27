"""Tests for node-side dropdown helper behavior."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPO_ROOT.parents[1]
PACKAGE_NAME = "comfyui_depthaccel_node_helpers_test_package"


def _load_nodes():
    if str(COMFYUI_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFYUI_ROOT))

    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module.nodes


class NodeHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes = _load_nodes()

    def test_preferred_option_prefers_fused_over_explicit(self):
        # Lexicographic order would put the vitb explicit graph first; the
        # preference chain must still select the vitl fused graph.
        options = [
            "da2_vitb_fp16_dynamic.onnx",
            "da2_vitl_fp16_dynamic_fused.onnx",
            "da2_vitl_fp16_dynamic.onnx",
        ]
        chosen = self.nodes._preferred_option(
            options,
            "da2_vitl_fp16_dynamic_fused.onnx",
            "da2_vitl_fp16_dynamic.onnx",
        )
        self.assertEqual(chosen, "da2_vitl_fp16_dynamic_fused.onnx")

    def test_preferred_option_prefers_high_encoder_fused(self):
        options = [
            "da2_vitb_fp16_dynamic_fused.onnx",
            "da2_vitb_fp16_dynamic.onnx",
            "da2_vits_fp16_dynamic_fused.onnx",
        ]
        chosen = self.nodes._preferred_option(
            options,
            "da2_vitl_fp16_dynamic_fused.onnx",
            "da2_vitb_fp16_dynamic_fused.onnx",
            "da2_vitb_fp16_dynamic.onnx",
            "da2_vits_fp16_dynamic_fused.onnx",
        )
        self.assertEqual(chosen, "da2_vitb_fp16_dynamic_fused.onnx")

    def test_preferred_option_falls_back_to_first_present(self):
        options = ["da2_vitb_fp16_dynamic.onnx", "da2_vitl_fp16_dynamic.onnx"]
        chosen = self.nodes._preferred_option(
            options,
            "da2_vitl_fp16_dynamic_fused.onnx",
            "da2_vitb_fp16_dynamic_fused.onnx",
        )
        self.assertEqual(chosen, "da2_vitb_fp16_dynamic.onnx")

    def test_preferred_option_handles_empty_options(self):
        self.assertEqual(self.nodes._preferred_option([], "anything.onnx"), "")

    def test_model_sort_groups_by_family_then_size(self):
        names = [
            "distill-any-depth-large.safetensors",
            "depth_anything_v2_vits.pth",
            "depth_anything_v2_vitl_fp32.safetensors",
            "distill-any-depth-small.safetensors",
            "depth_anything_v2_vitb.pth",
            "unknown_custom.safetensors",
        ]
        ordered = sorted(names, key=self.nodes._model_sort_key)
        self.assertEqual(
            ordered,
            [
                "depth_anything_v2_vits.pth",
                "depth_anything_v2_vitb.pth",
                "depth_anything_v2_vitl_fp32.safetensors",
                "unknown_custom.safetensors",
                "distill-any-depth-small.safetensors",
                "distill-any-depth-large.safetensors",
            ],
        )

    def test_resolve_precision_auto_picks_cuda_fp16(self):
        with mock.patch("comfy.model_management.get_torch_device", return_value=torch.device("cuda")):
            self.assertEqual(self.nodes._resolve_precision("auto"), "fp16")

    def test_resolve_precision_auto_picks_cpu_fp32(self):
        with mock.patch("comfy.model_management.get_torch_device", return_value=torch.device("cpu")):
            self.assertEqual(self.nodes._resolve_precision("auto"), "fp32")

    def test_resolve_precision_passes_explicit_values_through(self):
        self.assertEqual(self.nodes._resolve_precision("bf16"), "bf16")

    def test_cuda_device_arg_preserves_active_gpu_index(self):
        with mock.patch("comfy.model_management.get_torch_device", return_value=torch.device("cuda", 1)):
            self.assertEqual(self.nodes._cuda_device_arg(), "cuda:1")
        with mock.patch("comfy.model_management.get_torch_device", return_value=torch.device("cpu")):
            self.assertEqual(self.nodes._cuda_device_arg(), "cuda")

    def test_free_cuda_vram_invokes_comfy_memory_management_on_cuda_only(self):
        cuda_dev = torch.device("cuda", 1)
        with (
            mock.patch("comfy.model_management.get_torch_device", return_value=cuda_dev),
            mock.patch("comfy.model_management.free_memory") as free_mock,
            mock.patch("comfy.model_management.soft_empty_cache") as cache_mock,
        ):
            self.nodes._free_cuda_vram(1024)
            free_mock.assert_called_once_with(1024, cuda_dev)
            cache_mock.assert_called_once_with()

        with (
            mock.patch("comfy.model_management.get_torch_device", return_value=torch.device("cpu")),
            mock.patch("comfy.model_management.free_memory") as free_mock,
            mock.patch("comfy.model_management.soft_empty_cache") as cache_mock,
        ):
            self.nodes._free_cuda_vram(1024)
            free_mock.assert_not_called()
            cache_mock.assert_not_called()

    def test_relative_model_subdir_preserves_nested_folder(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            nested = root / "dad" / "model.onnx"
            nested.parent.mkdir(parents=True)
            nested.write_bytes(b"onnx")
            with mock.patch("folder_paths.get_folder_paths", return_value=[str(root)]):
                self.assertEqual(self.nodes._relative_model_subdir(nested), Path("dad"))
                self.assertEqual(self.nodes._relative_model_subdir(root / "flat.onnx"), Path("."))

    def test_build_tensorrt_node_passes_persistent_timing_cache(self):
        import importlib
        import tempfile
        from types import SimpleNamespace

        try:
            export_trt = importlib.import_module(PACKAGE_NAME + ".tools.export_tensorrt")
        except ImportError:
            self.skipTest("tensorrt is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            onnx_file = root / "da2_vits_fp16_dynamic.onnx"
            onnx_file.write_bytes(b"onnx")
            fake_handle = SimpleNamespace(model_path=onnx_file, metadata={})
            with (
                mock.patch("folder_paths.get_folder_paths", return_value=[str(root)]),
                mock.patch.object(self.nodes, "_free_cuda_vram"),
                mock.patch.object(export_trt, "build_engine") as build_mock,
                mock.patch.object(self.nodes, "load_tensorrt_model", return_value="trt_handle"),
            ):
                result = self.nodes.DepthAccelBuildTensorrtEngine.execute(
                    fake_handle, 14, 14, 560, 784, 2016, 2016, 4
                )

            self.assertEqual(result.args, ("trt_handle",))
            build_mock.assert_called_once()
            _, kwargs = build_mock.call_args
            self.assertEqual(kwargs["timing_cache"], root / ".trt_timing.cache")


if __name__ == "__main__":
    unittest.main()
