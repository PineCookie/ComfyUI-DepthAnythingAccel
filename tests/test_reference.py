import unittest

import torch

from pathlib import Path

from depthaccel.reference import (
    MODEL_CONFIGS,
    ReferenceDepthModel,
    detect_checkpoint_encoder,
    estimate_depth,
    infer_reference_raw,
    load_checkpoint_state_dict,
    load_reference_model,
    normalize_checkpoint_state_dict,
    resolve_model_family,
)


class ReferenceCheckpointTests(unittest.TestCase):
    def test_detects_all_supported_encoder_dimensions(self):
        for dimension, encoder in ((384, "vits"), (768, "vitb"), (1024, "vitl"), (1536, "vitg")):
            state_dict = {
                "pretrained.patch_embed.proj.weight": torch.empty(dimension, 3, 14, 14)
            }
            self.assertEqual(detect_checkpoint_encoder(state_dict), encoder)

    def test_rejects_missing_architecture_key(self):
        with self.assertRaisesRegex(ValueError, "missing or invalid"):
            detect_checkpoint_encoder({})

    def test_model_family_can_be_explicit_or_inferred_from_name(self):
        self.assertEqual(resolve_model_family(Path("model.safetensors"), "dad"), "dad")
        self.assertEqual(resolve_model_family(Path("distill_any_depth_vits.safetensors"), "auto"), "dad")
        self.assertEqual(resolve_model_family(Path("depth_anything_v2_vits.pth"), "auto"), "da2")
        # Parent folders must not affect detection.
        self.assertEqual(resolve_model_family(Path("C:/Users/Soledad/distilled/depth_anything_v2_vits.pth"), "auto"), "da2")

    def test_normalizes_dad_large_backbone_prefix(self):
        tensor = torch.empty(384, 3, 14, 14)
        normalized = normalize_checkpoint_state_dict(
            {
                "backbone.patch_embed.proj.weight": tensor,
                "backbone.blocks.0.11.norm1.weight": torch.empty(384),
                "depth_head.projects.0.weight": torch.empty(48, 384, 1, 1),
            }
        )
        self.assertIs(normalized["pretrained.patch_embed.proj.weight"], tensor)
        self.assertIn("pretrained.blocks.11.norm1.weight", normalized)
        self.assertIn("depth_head.projects.0.weight", normalized)

    def test_rejects_mixed_checkpoint_prefixes(self):
        with self.assertRaisesRegex(ValueError, "mixes"):
            normalize_checkpoint_state_dict(
                {"backbone.cls_token": torch.empty(1), "pretrained.cls_token": torch.empty(1)}
            )

    def test_load_checkpoint_state_dict_normalizes_safetensors_prefix(self):
        from tempfile import TemporaryDirectory

        from safetensors.torch import save_file

        state = {
            "backbone.patch_embed.proj.weight": torch.zeros(384, 3, 14, 14),
            "depth_head.projects.0.weight": torch.zeros(48, 384, 1, 1),
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            save_file(state, str(path))
            loaded = load_checkpoint_state_dict(path)

        self.assertIn("pretrained.patch_embed.proj.weight", loaded)
        self.assertIn("depth_head.projects.0.weight", loaded)
        self.assertNotIn("backbone.patch_embed.proj.weight", loaded)

    def test_infer_reference_raw_disables_autograd(self):
        observed_grad_enabled = []

        class RecordingModel(torch.nn.Module):
            def forward(self, x):
                observed_grad_enabled.append(torch.is_grad_enabled())
                return x[:, 0]

        reference = ReferenceDepthModel(
            model=RecordingModel(),
            device=torch.device("cpu"),
            precision="fp32",
            model_path=Path("model.safetensors"),
            encoder="vits",
            model_family="da2",
        )
        prepared = torch.zeros(2, 3, 14, 14, requires_grad=True)
        out = infer_reference_raw(reference, prepared)

        self.assertEqual(observed_grad_enabled, [False, False])
        self.assertFalse(out.requires_grad)
        self.assertEqual(tuple(out.shape), (2, 14, 14))

    def test_load_reference_model_materializes_fp16_checkpoint_from_meta_to_fp32(self):
        from tempfile import TemporaryDirectory

        from safetensors.torch import save_file

        from depthaccel.models.da2 import DepthAnythingV2

        source = DepthAnythingV2(**MODEL_CONFIGS["vits"]).eval()
        fp16_state = {
            key: (value.half() if value.is_floating_point() else value)
            for key, value in source.state_dict().items()
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth_anything_v2_vits_fp16.safetensors"
            save_file(fp16_state, str(path))
            loaded = load_reference_model(
                str(path),
                encoder="auto",
                precision="fp32",
                device=torch.device("cpu"),
            )

        self.assertEqual(loaded.encoder, "vits")
        self.assertTrue(
            all(
                param.device.type == "cpu" and param.dtype == torch.float32
                for param in loaded.model.parameters()
            )
        )
        out = estimate_depth(loaded, torch.zeros(1, 14, 14, 3))
        self.assertEqual(tuple(out.shape), (1, 14, 14, 3))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_load_reference_model_uses_native_fp16_weights_and_channels_last_on_cuda(self):
        from tempfile import TemporaryDirectory

        from safetensors.torch import save_file

        from depthaccel.models.da2 import DepthAnythingV2

        source = DepthAnythingV2(**MODEL_CONFIGS["vits"]).eval()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth_anything_v2_vits_fp32.safetensors"
            save_file(source.state_dict(), str(path))
            loaded = load_reference_model(
                str(path),
                encoder="auto",
                precision="fp16",
                device=torch.device("cuda", 0),
            )

        self.assertEqual(loaded.torch_dtype, torch.float16)
        self.assertEqual(next(loaded.model.parameters()).dtype, torch.float16)
        self.assertTrue(
            loaded.model.depth_head.projects[0].weight.is_contiguous(
                memory_format=torch.channels_last
            )
        )
        out = estimate_depth(loaded, torch.rand(1, 14, 14, 3))
        self.assertEqual(tuple(out.shape), (1, 14, 14, 3))
        self.assertEqual(out.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
