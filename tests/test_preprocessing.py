import unittest

import torch

from depthaccel.preprocessing import (
    align_to_patch,
    guard_inference_resolution,
    normalize_depth,
    prepare_image,
    restore_depth,
)


class PreprocessingTests(unittest.TestCase):
    def test_align_to_patch_rounds_down_and_keeps_one_patch(self):
        self.assertEqual(align_to_patch(1024), 1022)
        self.assertEqual(align_to_patch(1), 14)

    def test_prepare_image_uses_native_patch_aligned_shape(self):
        image = torch.zeros(2, 1024, 1536, 3)
        prepared, geometry = prepare_image(image)

        self.assertEqual(tuple(prepared.shape), (2, 3, 1022, 1526))
        self.assertEqual((geometry.original_height, geometry.original_width), (1024, 1536))
        self.assertEqual((geometry.inference_height, geometry.inference_width), (1022, 1526))

    def test_prepare_image_keeps_already_aligned_shape(self):
        image = torch.zeros(1, 784, 1176, 3)
        prepared, geometry = prepare_image(image)

        self.assertEqual(tuple(prepared.shape), (1, 3, 784, 1176))
        self.assertEqual(geometry.original_height, geometry.inference_height)
        self.assertEqual(geometry.original_width, geometry.inference_width)

    def test_small_image_is_not_reduced_to_zero(self):
        image = torch.zeros(1, 5, 9, 3)
        prepared, geometry = prepare_image(image)

        self.assertEqual(tuple(prepared.shape), (1, 3, 14, 14))
        self.assertEqual((geometry.inference_height, geometry.inference_width), (14, 14))

    def test_restore_depth_returns_original_comfy_image_shape(self):
        image = torch.zeros(1, 15, 17, 3)
        _, geometry = prepare_image(image)
        depth = torch.arange(14 * 14, dtype=torch.float32).reshape(1, 14, 14)

        restored = restore_depth(depth, geometry)

        self.assertEqual(tuple(restored.shape), (1, 15, 17, 3))
        self.assertTrue(torch.all(restored >= 0))
        self.assertTrue(torch.all(restored <= 1))

    def test_normalize_depth_promotes_fp16_to_fp32(self):
        # FP16 epsilon is ~9.8e-4; normalizing in the source dtype would
        # clamp_min with that value and distort a low-range map. The result
        # must be FP32 and stay in [0, 1].
        depth = torch.zeros(1, 8, 8, dtype=torch.float16) + 100.0
        depth[..., :1] = 100.125  # small representable range (2 fp16 ulps at 100)

        normalized = normalize_depth(depth)

        self.assertEqual(normalized.dtype, torch.float32)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(torch.all(normalized >= 0))
        self.assertTrue(torch.all(normalized <= 1))

    def test_guard_inference_resolution_allows_within_budget(self):
        self.assertEqual(guard_inference_resolution(1024, 1536, "fp32"), (1022, 1526))
        self.assertEqual(guard_inference_resolution(2016, 2016, "fp16"), (2016, 2016))

    def test_guard_inference_resolution_rejects_over_budget(self):
        with self.assertRaisesRegex(ValueError, "FP32 budget"):
            guard_inference_resolution(1428, 2464, "fp32")
        # The same native size is fine for the FP16 budget.
        self.assertEqual(guard_inference_resolution(1428, 2464, "fp16"), (1428, 2464))

    def test_guard_inference_resolution_unknown_precision_is_unlimited(self):
        self.assertEqual(guard_inference_resolution(4096, 4096, "custom"), (4088, 4088))


if __name__ == "__main__":
    unittest.main()
