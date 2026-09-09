import unittest

import torch

from depthaccel.onnx_backend import _prepare_binding_tensor, _validate_model_contract
from depthaccel.preprocessing import prepare_image


class OnnxBackendTests(unittest.TestCase):
    def test_iobinding_tensor_is_contiguous(self):
        image = torch.zeros(1, 15, 17, 3)
        prepared, _ = prepare_image(image)

        self.assertFalse(prepared.is_contiguous())
        binding_tensor = _prepare_binding_tensor(prepared, torch.device("cpu"))

        self.assertTrue(binding_tensor.is_contiguous())
        self.assertEqual(binding_tensor.dtype, torch.float32)
        self.assertTrue(torch.equal(binding_tensor, prepared))

    def test_model_contract_accepts_dynamic_fp32_da2(self):
        class Info:
            def __init__(self, value_type, shape):
                self.type = value_type
                self.shape = shape

        _validate_model_contract(
            Info("tensor(float)", ["batch", 3, "height", "width"]),
            Info("tensor(float)", ["batch", "height", "width"]),
            {"depthaccel.architecture": "depth-anything-v2", "depthaccel.patch_size": "14"},
        )

    def test_model_contract_accepts_dynamic_fp16_da2(self):
        class Info:
            def __init__(self, value_type, shape):
                self.type = value_type
                self.shape = shape

        _validate_model_contract(
            Info("tensor(float16)", ["batch", 3, "height", "width"]),
            Info("tensor(float16)", ["batch", "height", "width"]),
            {"depthaccel.precision": "fp16"},
        )

    def test_model_contract_accepts_dad_architecture_metadata(self):
        class Info:
            def __init__(self, value_type, shape):
                self.type = value_type
                self.shape = shape

        _validate_model_contract(
            Info("tensor(float)", ["batch", 3, "height", "width"]),
            Info("tensor(float)", ["batch", "height", "width"]),
            {"depthaccel.architecture": "distill-any-depth"},
        )

    def test_model_contract_rejects_precision_metadata_mismatch(self):
        class Info:
            def __init__(self, value_type, shape):
                self.type = value_type
                self.shape = shape

        with self.assertRaisesRegex(ValueError, "precision metadata"):
            _validate_model_contract(
                Info("tensor(float16)", ["batch", 3, "height", "width"]),
                Info("tensor(float16)", ["batch", "height", "width"]),
                {"depthaccel.precision": "fp32"},
            )

    def test_model_contract_rejects_static_spatial_axes(self):
        class Info:
            def __init__(self, value_type, shape):
                self.type = value_type
                self.shape = shape

        with self.assertRaisesRegex(ValueError, "dynamic ONNX batch/H/W"):
            _validate_model_contract(
                Info("tensor(float)", [1, 3, 518, 518]),
                Info("tensor(float)", [1, 518, 518]),
                {},
            )

    def test_model_contract_rejects_encoder_metadata_mismatch(self):
        class Info:
            def __init__(self, value_type, shape):
                self.type = value_type
                self.shape = shape

        with self.assertRaisesRegex(ValueError, "metadata mismatch for vitl"):
            _validate_model_contract(
                Info("tensor(float)", ["batch", 3, "height", "width"]),
                Info("tensor(float)", ["batch", "height", "width"]),
                {
                    "depthaccel.encoder": "vitl",
                    "depthaccel.hidden_size": "768",
                    "depthaccel.num_heads": "16",
                },
            )


if __name__ == "__main__":
    unittest.main()
