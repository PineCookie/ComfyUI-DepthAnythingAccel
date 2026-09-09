import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper

from tools.fuse_onnx_attention import validate_attention_graph


def make_attention_graph(hidden_size: int = 4) -> onnx.ModelProto:
    prefix = "/blocks.0/attn/"
    nodes = [
        helper.make_node("MatMul", ["input", "qkv_weight"], ["qkv_mm"], name=prefix + "qkv/MatMul"),
        helper.make_node("Add", ["qkv_bias", "qkv_mm"], ["qkv"], name=prefix + "qkv/Add"),
        helper.make_node("Identity", ["qkv"], ["attention"], name=prefix + "Reshape_1"),
        helper.make_node("MatMul", ["attention", "proj_weight"], ["projected"], name=prefix + "proj/MatMul"),
        helper.make_node("Add", ["proj_bias", "projected"], ["output"], name=prefix + "proj/Add"),
    ]
    initializers = [
        numpy_helper.from_array(np.zeros((hidden_size, hidden_size * 3), np.float32), "qkv_weight"),
        numpy_helper.from_array(np.zeros((hidden_size * 3,), np.float32), "qkv_bias"),
        numpy_helper.from_array(np.zeros((hidden_size, hidden_size), np.float32), "proj_weight"),
        numpy_helper.from_array(np.zeros((hidden_size,), np.float32), "proj_bias"),
    ]
    graph = helper.make_graph(
        nodes,
        "test",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, ["batch", "sequence", hidden_size])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)],
        initializer=initializers,
    )
    return helper.make_model(graph)


class AttentionGraphValidationTests(unittest.TestCase):
    def test_expected_graph_is_accepted(self):
        validate_attention_graph(make_attention_graph(), hidden_size=4, num_heads=2, num_blocks=1)

    def test_wrong_hidden_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "QKV weight shape"):
            validate_attention_graph(make_attention_graph(), hidden_size=8, num_heads=2, num_blocks=1)

    def test_unexpected_external_consumer_is_rejected(self):
        model = make_attention_graph()
        model.graph.node.append(
            helper.make_node("Identity", ["qkv"], ["leak"], name="/outside/consumer")
        )
        with self.assertRaisesRegex(ValueError, "external consumers"):
            validate_attention_graph(model, hidden_size=4, num_heads=2, num_blocks=1)


if __name__ == "__main__":
    unittest.main()
