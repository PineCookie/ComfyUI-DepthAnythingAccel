"""Tests for TensorRT engine contract validation, device selection, and batching."""

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from depthaccel.tensorrt_backend import (
    TensorrtDepthModel,
    _resolve_cuda_device,
    _validate_engine_contract,
    estimate_depth_tensorrt,
    infer_tensorrt_raw,
)


def _fake_trt():
    return SimpleNamespace(
        TensorIOMode=SimpleNamespace(INPUT="INPUT", OUTPUT="OUTPUT"),
        DataType=SimpleNamespace(HALF="HALF", FLOAT="FLOAT", INT32="INT32"),
    )


class _FakeEngine:
    def __init__(
        self,
        *,
        input_dtype="HALF",
        output_dtype="HALF",
        input_shape=(-1, 3, -1, -1),
        output_shape=(-1, -1, -1),
        profile=((1, 3, 14, 14), (1, 3, 560, 784), (1, 3, 2016, 2016)),
        num_profiles=1,
    ):
        self._names = ["image", "depth"]
        self._modes = {"image": "INPUT", "depth": "OUTPUT"}
        self._dtypes = {"image": input_dtype, "depth": output_dtype}
        self._shapes = {"image": input_shape, "depth": output_shape}
        self._profile = profile
        self.num_io_tensors = len(self._names)
        self.num_optimization_profiles = num_profiles

    def get_tensor_name(self, index):
        return self._names[index]

    def get_tensor_mode(self, name):
        return self._modes[name]

    def get_tensor_dtype(self, name):
        return self._dtypes[name]

    def get_tensor_shape(self, name):
        return self._shapes[name]

    def get_tensor_profile_shape(self, name, profile_index):
        return self._profile


class TensorrtBackendTests(unittest.TestCase):
    def test_validate_engine_contract_accepts_matching_fp16(self):
        trt = _fake_trt()
        result = _validate_engine_contract(_FakeEngine(), trt)
        self.assertEqual(
            result,
            ("image", "depth", "fp16", torch.float16, (14, 14), (560, 784), (2016, 2016)),
        )

    def test_validate_engine_contract_rejects_mismatched_io_dtypes(self):
        trt = _fake_trt()
        engine = _FakeEngine(input_dtype="HALF", output_dtype="FLOAT")
        with self.assertRaisesRegex(ValueError, "matching input/output dtypes"):
            _validate_engine_contract(engine, trt)

    def test_validate_engine_contract_rejects_bad_output_rank(self):
        trt = _fake_trt()
        engine = _FakeEngine(output_shape=(-1, 1, -1, -1))
        with self.assertRaisesRegex(ValueError, r"output \[B,H,W\]"):
            _validate_engine_contract(engine, trt)

    def test_validate_engine_contract_rejects_missing_profile(self):
        trt = _fake_trt()
        engine = _FakeEngine(num_profiles=0)
        with self.assertRaisesRegex(ValueError, "optimization profile"):
            _validate_engine_contract(engine, trt)

    def test_resolve_cuda_device_uses_active_gpu_index(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.current_device", return_value=1),
        ):
            self.assertEqual(_resolve_cuda_device("cuda"), torch.device("cuda", 1))
            self.assertEqual(_resolve_cuda_device("cuda:0"), torch.device("cuda", 0))
        with self.assertRaisesRegex(ValueError, "CUDA"):
            _resolve_cuda_device("cpu")

    def test_estimate_depth_tensorrt_restores_each_frame_to_cpu(self):
        seen_batches = []

        def fake_infer(model, prepared, **_kwargs):
            seen_batches.append(prepared.shape[0])
            return torch.arange(
                prepared.shape[2] * prepared.shape[3], dtype=torch.float32
            ).reshape(1, prepared.shape[2], prepared.shape[3])

        image = torch.zeros(3, 15, 17, 3)
        fake_model = SimpleNamespace(device=torch.device("cpu"), torch_dtype=torch.float32)
        with mock.patch("depthaccel.tensorrt_backend.infer_tensorrt_raw", side_effect=fake_infer):
            out = estimate_depth_tensorrt(fake_model, image)

        self.assertEqual(seen_batches, [1, 1, 1])
        self.assertEqual(tuple(out.shape), (3, 15, 17, 3))
        self.assertEqual(out.device.type, "cpu")

    def test_infer_tensorrt_raw_validates_bounds_before_gpu_transfer(self):
        from pathlib import Path

        model = TensorrtDepthModel(
            engine=None,
            input_name="image",
            output_name="depth",
            model_path=Path("fake.trt"),
            device=torch.device("cuda", 0),
            precision="fp16",
            torch_dtype=torch.float16,
            min_hw=(14, 14),
            opt_hw=(560, 784),
            max_hw=(1022, 1022),
            stream=None,
        )
        oversized = torch.zeros(1, 3, 2016, 2016)
        with (
            mock.patch.object(torch.Tensor, "to", side_effect=AssertionError("should not copy to GPU")),
            self.assertRaisesRegex(ValueError, "outside the engine profile"),
        ):
            infer_tensorrt_raw(model, oversized)

    def test_export_tensorrt_profile_bounds_and_strongly_typed_flags(self):
        try:
            from tools.export_tensorrt import _network_creation_flags, validate_profile_bounds
        except ImportError:
            self.skipTest("tensorrt is not installed")

        validate_profile_bounds((14, 14), (560, 784), (2016, 2016), workspace_gb=4)

        with self.assertRaisesRegex(ValueError, "min <= opt <= max"):
            validate_profile_bounds((560, 14), (280, 784), (2016, 2016))
        with self.assertRaisesRegex(ValueError, "at least 14"):
            validate_profile_bounds((0, 14), (560, 784), (2016, 2016))
        with self.assertRaisesRegex(ValueError, "divisible by 14"):
            validate_profile_bounds((14, 14), (561, 784), (2016, 2016))
        with self.assertRaisesRegex(ValueError, "workspace_gb"):
            validate_profile_bounds((14, 14), (560, 784), (2016, 2016), workspace_gb=0)

        fake_trt_modern = SimpleNamespace(
            NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=0)
        )
        self.assertEqual(_network_creation_flags(fake_trt_modern), (1, True))

        fake_trt_legacy = SimpleNamespace(NetworkDefinitionCreationFlag=SimpleNamespace())
        self.assertEqual(_network_creation_flags(fake_trt_legacy), (0, False))


if __name__ == "__main__":
    unittest.main()
