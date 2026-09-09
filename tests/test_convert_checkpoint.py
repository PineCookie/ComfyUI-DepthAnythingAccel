import unittest

import torch

from devtools.convert_checkpoint import convert_checkpoint


class ConvertCheckpointTests(unittest.TestCase):
    def test_fp16_casts_floats_and_keeps_integers(self):
        state = {
            "weight": torch.ones(2, 3, dtype=torch.float32),
            "bias": torch.zeros(4, dtype=torch.float32),
            "num_batches_tracked": torch.tensor(7, dtype=torch.int64),
        }

        converted = convert_checkpoint(state, "fp16")

        self.assertEqual(converted["weight"].dtype, torch.float16)
        self.assertEqual(converted["bias"].dtype, torch.float16)
        self.assertEqual(converted["num_batches_tracked"].dtype, torch.int64)
        self.assertEqual(converted["num_batches_tracked"].item(), 7)

    def test_fp32_keeps_floating_dtype(self):
        converted = convert_checkpoint({"weight": torch.ones(1)}, "fp32")
        self.assertEqual(converted["weight"].dtype, torch.float32)

    def test_rejects_unknown_dtype(self):
        with self.assertRaises(ValueError):
            convert_checkpoint({"weight": torch.ones(1)}, "int8")


if __name__ == "__main__":
    unittest.main()
