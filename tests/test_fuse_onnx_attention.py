import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper

from tools.fuse_onnx_attention import fuse_attention, validate_attention_graph


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

    def test_fuse_attention_mha_preserves_qkv_add_producer(self):
        model = make_attention_graph(hidden_size=4)
        fused_blocks = fuse_attention(
            model, num_heads=2, hidden_size=4, operator="mha", num_blocks=1
        )
        self.assertEqual(fused_blocks, 1)

        nodes_by_name = {node.name: node for node in model.graph.node}
        self.assertIn("/blocks.0/attn/qkv/MatMul", nodes_by_name)
        self.assertIn("/blocks.0/attn/qkv/Add", nodes_by_name)
        self.assertIn("/blocks.0/attn/FusedAttention/Split", nodes_by_name)
        self.assertIn("/blocks.0/attn/FusedAttention", nodes_by_name)

        qkv_add = nodes_by_name["/blocks.0/attn/qkv/Add"]
        split = nodes_by_name["/blocks.0/attn/FusedAttention/Split"]
        mha = nodes_by_name["/blocks.0/attn/FusedAttention"]
        proj = nodes_by_name["/blocks.0/attn/proj/MatMul"]
        self.assertEqual(split.input[0], qkv_add.output[0])
        self.assertEqual(list(mha.input[:3]), list(split.output))
        self.assertEqual(proj.input[0], mha.output[0])

    def test_fuse_attention_fused_qkv_rewires_projection(self):
        model = make_attention_graph(hidden_size=4)
        fused_blocks = fuse_attention(
            model, num_heads=2, hidden_size=4, operator="attention", num_blocks=1
        )
        self.assertEqual(fused_blocks, 1)

        nodes_by_name = {node.name: node for node in model.graph.node}
        self.assertNotIn("/blocks.0/attn/qkv/MatMul", nodes_by_name)
        self.assertNotIn("/blocks.0/attn/qkv/Add", nodes_by_name)
        attn = nodes_by_name["/blocks.0/attn/FusedAttention"]
        proj = nodes_by_name["/blocks.0/attn/proj/MatMul"]
        self.assertEqual(list(attn.input), ["input", "qkv_weight", "qkv_bias"])
        self.assertEqual(proj.input[0], attn.output[0])


if __name__ == "__main__":
    unittest.main()
