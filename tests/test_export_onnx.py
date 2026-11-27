from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depthaccel.models.da2 import DepthAnythingV2
from depthaccel.reference import MODEL_CONFIGS
from tools.export_onnx import _save_and_check_onnx, export_dynamic_onnx


class ExportOnnxTests(unittest.TestCase):
    def test_export_leaves_supplied_model_dtype_and_weights_untouched(self):
        import onnx

        model = DepthAnythingV2(**MODEL_CONFIGS["vits"]).eval()
        weight = model.pretrained.patch_embed.proj.weight.detach().clone()
        reference = SimpleNamespace(
            model=model,
            # Deliberately differ from model's actual CPU parameter device to
            # verify tracing follows the module's parameters, not reference.device.
            device=torch.device("cuda", 0),
            encoder="vits",
            model_family="da2",
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "tiny.onnx"
            # Omit encoder= (defaults to "vitl") to verify metadata records
            # reference.encoder instead of the default parameter.
            export_dynamic_onnx(
                "unused",
                output,
                precision="fp16",
                height=14,
                width=14,
                skip_simplify=True,
                reference=reference,
            )
            exported = onnx.load(str(output))
            metadata = {entry.key: entry.value for entry in exported.metadata_props}

        self.assertEqual(metadata["depthaccel.encoder"], "vits")
        self.assertEqual(metadata["depthaccel.precision"], "fp16")
        self.assertEqual(model.pretrained.patch_embed.proj.weight.dtype, torch.float32)
        self.assertTrue(torch.equal(model.pretrained.patch_embed.proj.weight, weight))

    def test_save_and_check_spills_external_data_above_2gib_threshold(self):
        from unittest import mock

        fake_onnx = SimpleNamespace(
            save_model=mock.Mock(),
            save=mock.Mock(),
            checker=SimpleNamespace(check_model=mock.Mock()),
        )
        huge_model = SimpleNamespace(ByteSize=lambda: 2 << 30)
        output = Path("vitg_fp32.onnx")

        _save_and_check_onnx(fake_onnx, huge_model, output)

        fake_onnx.save_model.assert_called_once_with(
            huge_model,
            str(output),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="vitg_fp32.onnx.data",
        )
        fake_onnx.checker.check_model.assert_called_once_with(str(output))
        fake_onnx.save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
