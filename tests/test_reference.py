import unittest

import torch

from pathlib import Path

from depthaccel.reference import (
    detect_checkpoint_encoder,
    load_checkpoint_state_dict,
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


if __name__ == "__main__":
    unittest.main()
