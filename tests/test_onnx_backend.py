import sys
import unittest
from unittest import mock

import torch

from depthaccel.onnx_backend import (
    _prepare_binding_tensor,
    _resolve_target_device,
    _validate_model_contract,
    create_deferred_onnx_model,
    estimate_depth_onnx,
    infer_onnx_raw,
)
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

    def test_deferred_onnx_handle_does_not_require_onnxruntime(self):
        with mock.patch.dict(sys.modules, {"onnxruntime": None}):
            handle = create_deferred_onnx_model(
                "model.onnx",
                device="cpu",
                precision="fp16",
                metadata={"depthaccel.encoder": "vits"},
            )
        self.assertIsNone(handle.session)
        self.assertEqual(handle.precision, "fp16")
        self.assertEqual(handle.torch_dtype, torch.float16)
        self.assertEqual(handle.metadata["depthaccel.encoder"], "vits")

    def test_resolve_target_device_uses_active_cuda_index(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.current_device", return_value=1),
        ):
            self.assertEqual(_resolve_target_device("cuda"), torch.device("cuda", 1))
            self.assertEqual(_resolve_target_device("cuda:0"), torch.device("cuda", 0))
            self.assertEqual(_resolve_target_device("cpu"), torch.device("cpu"))

    def test_estimate_depth_onnx_processes_batch_one_frame_at_a_time(self):
        handle = create_deferred_onnx_model("model.onnx", device="cpu", precision="fp32")
        seen_batches = []

        def fake_infer(ref, prepared):
            seen_batches.append(prepared.shape[0])
            return torch.arange(
                prepared.shape[2] * prepared.shape[3], dtype=torch.float32
            ).reshape(1, prepared.shape[2], prepared.shape[3])

        image = torch.zeros(3, 15, 17, 3)
        with mock.patch("depthaccel.onnx_backend.infer_onnx_raw", side_effect=fake_infer):
            out = estimate_depth_onnx(handle, image)

        self.assertEqual(seen_batches, [1, 1, 1])
        self.assertEqual(tuple(out.shape), (3, 15, 17, 3))
        self.assertEqual(out.device.type, "cpu")

    def test_deferred_handle_loads_and_caches_session_once(self):
        import numpy as np

        class FakeSession:
            def run(self, outputs, feeds):
                arr = next(iter(feeds.values()))
                return [np.zeros((arr.shape[0], arr.shape[2], arr.shape[3]), dtype=np.float32)]

        handle = create_deferred_onnx_model("model.onnx", device="cpu", precision="fp32")
        loaded = create_deferred_onnx_model("model.onnx", device="cpu", precision="fp32")
        loaded.session = FakeSession()
        loaded.input_name = "image"
        loaded.output_name = "depth"
        loaded.providers = ("CPUExecutionProvider",)

        prepared = torch.zeros(1, 3, 14, 14)
        with mock.patch("depthaccel.onnx_backend.load_onnx_model", return_value=loaded) as load_mock:
            out1 = infer_onnx_raw(handle, prepared)
            out2 = infer_onnx_raw(handle, prepared)

        load_mock.assert_called_once()
        self.assertIs(handle.session, loaded.session)
        self.assertEqual(tuple(out1.shape), (1, 14, 14))
        self.assertEqual(tuple(out2.shape), (1, 14, 14))


if __name__ == "__main__":
    unittest.main()
