import unittest

import torch

from devtools.compare_checkpoint_formats import compare_state_dicts


class CompareCheckpointFormatsTests(unittest.TestCase):
    def test_identical_state_dicts_match(self):
        state = {"a": torch.ones(2, 3), "b": torch.zeros(1)}
        result = compare_state_dicts(state, dict(state))

        self.assertTrue(result["keys_match"])
        self.assertTrue(result["shapes_match"])
        self.assertTrue(result["values_match"])
        self.assertEqual(result["out_of_tolerance"], 0)

    def test_missing_and_unexpected_keys_are_reported(self):
        result = compare_state_dicts(
            {"a": torch.zeros(1), "b": torch.zeros(1)},
            {"a": torch.zeros(1), "c": torch.zeros(1)},
        )

        self.assertEqual(result["missing_keys"], ["b"])
        self.assertEqual(result["unexpected_keys"], ["c"])
        self.assertFalse(result["keys_match"])

    def test_shape_mismatch_is_reported(self):
        result = compare_state_dicts(
            {"a": torch.zeros(2, 3)},
            {"a": torch.zeros(3, 2)},
        )

        self.assertFalse(result["shapes_match"])
        self.assertEqual(
            result["shape_mismatches"][0]["official_shape"], [2, 3]
        )
        self.assertEqual(
            result["shape_mismatches"][0]["converted_shape"], [3, 2]
        )

    def test_dtype_mismatch_is_reported_but_equal_values_still_match(self):
        result = compare_state_dicts(
            {"a": torch.ones(1, dtype=torch.float32)},
            {"a": torch.ones(1, dtype=torch.float16)},
        )

        self.assertEqual(
            result["dtype_mismatches"][0]["official_dtype"], "torch.float32"
        )
        self.assertEqual(
            result["dtype_mismatches"][0]["converted_dtype"], "torch.float16"
        )
        self.assertTrue(result["values_match"])

    def test_value_difference_is_reported(self):
        result = compare_state_dicts(
            {"a": torch.zeros(1)},
            {"a": torch.ones(1)},
            atol=1e-6,
            rtol=1e-6,
        )

        self.assertFalse(result["values_match"])
        self.assertAlmostEqual(result["max_abs_diff"], 1.0)
        self.assertEqual(result["worst_key"], "a")


if __name__ == "__main__":
    unittest.main()
