"""Tests for node-side dropdown helper behavior."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


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
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(self.nodes._resolve_precision("auto"), "fp16")

    def test_resolve_precision_auto_picks_cpu_fp32(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(self.nodes._resolve_precision("auto"), "fp32")

    def test_resolve_precision_passes_explicit_values_through(self):
        self.assertEqual(self.nodes._resolve_precision("bf16"), "bf16")


if __name__ == "__main__":
    unittest.main()
