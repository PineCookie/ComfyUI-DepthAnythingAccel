from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.export_onnx import export_dynamic_onnx


class _TinyDepth(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x).squeeze(1)


class ExportOnnxTests(unittest.TestCase):
    def test_export_leaves_supplied_model_dtype_and_weights_untouched(self):
        model = _TinyDepth().eval()
        weight = model.conv.weight.detach().clone()
        reference = SimpleNamespace(
            model=model,
            device=torch.device("cpu"),
            encoder="vits",
            model_family="da2",
        )
        with tempfile.TemporaryDirectory() as tmp:
            export_dynamic_onnx(
                "unused",
                Path(tmp) / "tiny.onnx",
                encoder="vits",
                precision="fp16",
                height=14,
                width=14,
                skip_simplify=True,
                reference=reference,
            )

        self.assertEqual(model.conv.weight.dtype, torch.float32)
        self.assertTrue(torch.equal(model.conv.weight, weight))


if __name__ == "__main__":
    unittest.main()
