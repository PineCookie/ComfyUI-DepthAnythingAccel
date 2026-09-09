"""Replace DA2's explicit self-attention subgraphs with ORT fused attention."""

import argparse
from collections import Counter
import re
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper


BLOCK_ATTN = re.compile(r"^/blocks\.(\d+)/attn/")

DA2_ENCODERS = {
    "vits": {"hidden_size": 384, "num_heads": 6, "num_blocks": 12},
    "vitb": {"hidden_size": 768, "num_heads": 12, "num_blocks": 12},
    "vitl": {"hidden_size": 1024, "num_heads": 16, "num_blocks": 24},
    "vitg": {"hidden_size": 1536, "num_heads": 24, "num_blocks": 40},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--encoder",
        choices=sorted(DA2_ENCODERS),
        default=None,
        help="DA2 encoder; defaults to the graph's depthaccel.encoder metadata.",
    )
    parser.add_argument("--num-heads", type=int, help="Override used only after validation.")
    parser.add_argument("--hidden-size", type=int, help="Override used only after validation.")
    parser.add_argument(
        "--operator",
        choices=["mha", "attention"],
        default="attention",
        help="Fused operator to emit; attention also fuses the QKV projection.",
    )
    return parser.parse_args()


def operator_counts(model: onnx.ModelProto) -> Counter[str]:
    return Counter(node.op_type for node in model.graph.node)


def _unique_node(nodes_by_name: dict[str, list], name: str):
    matches = nodes_by_name.get(name, [])
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one node named {name!r}, found {len(matches)}")
    return matches[0]


def validate_attention_graph(
    model: onnx.ModelProto,
    *,
    hidden_size: int,
    num_heads: int,
    num_blocks: int,
) -> None:
    """Fail closed unless the graph is the exact DA2 attention layout we replace."""
    if hidden_size % num_heads:
        raise ValueError("hidden size must be divisible by the number of heads")

    nodes_by_name: dict[str, list] = {}
    for node in model.graph.node:
        nodes_by_name.setdefault(node.name, []).append(node)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}

    observed_blocks = {
        int(match.group(1))
        for node in model.graph.node
        if (match := BLOCK_ATTN.match(node.name))
    }
    expected_blocks = set(range(num_blocks))
    if observed_blocks != expected_blocks:
        raise ValueError(
            "DA2 attention blocks do not match the selected encoder: "
            f"expected={sorted(expected_blocks)}, observed={sorted(observed_blocks)}"
        )

    consumers: dict[str, list] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    for block in range(num_blocks):
        prefix = f"/blocks.{block}/attn/"
        qkv_matmul = _unique_node(nodes_by_name, prefix + "qkv/MatMul")
        qkv_add = _unique_node(nodes_by_name, prefix + "qkv/Add")
        proj_matmul = _unique_node(nodes_by_name, prefix + "proj/MatMul")
        proj_add = _unique_node(nodes_by_name, prefix + "proj/Add")

        if len(qkv_matmul.input) != 2 or len(qkv_matmul.output) != 1:
            raise ValueError(f"Unexpected QKV MatMul signature in block {block}")
        weight = initializers.get(qkv_matmul.input[1])
        if weight is None or list(weight.dims) != [hidden_size, hidden_size * 3]:
            actual = None if weight is None else list(weight.dims)
            raise ValueError(
                f"Unexpected QKV weight shape in block {block}: "
                f"expected={[hidden_size, hidden_size * 3]}, actual={actual}"
            )
        bias_inputs = [name for name in qkv_add.input if name in initializers]
        data_inputs = [name for name in qkv_add.input if name not in initializers]
        if len(bias_inputs) != 1 or list(initializers[bias_inputs[0]].dims) != [hidden_size * 3]:
            raise ValueError(f"Unexpected QKV bias in block {block}")
        if data_inputs != list(qkv_matmul.output):
            raise ValueError(f"QKV Add is not connected to QKV MatMul in block {block}")

        for node in model.graph.node:
            if not node.name.startswith(prefix) or node.name in {proj_matmul.name, proj_add.name}:
                continue
            for output_name in node.output:
                external = [
                    consumer.name
                    for consumer in consumers.get(output_name, [])
                    if not consumer.name.startswith(prefix)
                ]
                if external:
                    raise ValueError(
                        f"Cannot safely replace block {block}; {node.name} has "
                        f"external consumers {external}"
                    )


def fuse_attention(
    model: onnx.ModelProto,
    num_heads: int,
    hidden_size: int,
    operator: str = "mha",
    num_blocks: int = 24,
) -> int:
    if operator not in {"mha", "attention"}:
        raise ValueError(f"Unsupported fused attention operator: {operator}")
    validate_attention_graph(
        model,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_blocks=num_blocks,
    )

    split_name = "depthaccel_attention_split_sizes"
    if operator == "mha":
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.array([hidden_size, hidden_size, hidden_size], dtype="int64"),
                name=split_name,
            )
        )

    new_nodes = []
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    fused_blocks = set()
    fused_output_by_block = {}
    qkv_projection_by_block = {}
    for node in model.graph.node:
        match = BLOCK_ATTN.match(node.name)
        if not match:
            new_nodes.append(node)
            continue

        block = int(match.group(1))
        if node.name.endswith("qkv/MatMul"):
            if operator == "mha":
                new_nodes.append(node)
            else:
                qkv_projection_by_block[block] = node
            continue

        if node.name.endswith("qkv/Add"):
            output_name = f"/blocks.{block}/attn/FusedAttention_output_0"
            if operator == "mha":
                q_name = f"/blocks.{block}/attn/FusedAttention_q"
                k_name = f"/blocks.{block}/attn/FusedAttention_k"
                v_name = f"/blocks.{block}/attn/FusedAttention_v"
                split = helper.make_node(
                    "Split",
                    inputs=[node.output[0], split_name],
                    outputs=[q_name, k_name, v_name],
                    name=f"/blocks.{block}/attn/FusedAttention/Split",
                    axis=2,
                )
                fused = helper.make_node(
                    "MultiHeadAttention",
                    inputs=[q_name, k_name, v_name, ""],
                    outputs=[output_name],
                    name=f"/blocks.{block}/attn/FusedAttention",
                    domain="com.microsoft",
                    num_heads=num_heads,
                )
                new_nodes.extend([split, fused])
            else:
                qkv_projection = qkv_projection_by_block.get(block)
                if qkv_projection is None:
                    raise ValueError(f"Missing qkv projection for block {block}")
                bias_name = next(name for name in node.input if name in initializer_names)
                fused = helper.make_node(
                    "Attention",
                    inputs=[qkv_projection.input[0], qkv_projection.input[1], bias_name],
                    outputs=[output_name],
                    name=f"/blocks.{block}/attn/FusedAttention",
                    domain="com.microsoft",
                    num_heads=num_heads,
                )
                new_nodes.append(fused)
            fused_blocks.add(block)
            fused_output_by_block[block] = output_name
            continue

        if node.name.endswith("proj/MatMul"):
            node.input[0] = fused_output_by_block[block]
            new_nodes.append(node)
            continue

        if node.name.endswith("proj/Add"):
            new_nodes.append(node)
            continue

        # All other nodes in this block's attention subgraph are replaced by
        # the split + fused attention nodes above.

    if fused_blocks != set(range(num_blocks)):
        raise RuntimeError(f"Expected to fuse {num_blocks} blocks, fused {sorted(fused_blocks)}")
    if len(fused_blocks) != len(fused_output_by_block):
        raise RuntimeError("Some fused attention blocks have no output")

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    if not any(opset.domain == "com.microsoft" for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    return len(fused_blocks)


def fuse_attention_model(
    input_path: str | Path,
    output_path: str | Path,
    *,
    encoder: str | None = None,
    operator: str = "attention",
) -> Path:
    """Fuse attention subgraphs in an explicit ONNX graph and save the result.

    ``encoder`` is taken from the graph's ``depthaccel.encoder`` metadata when
    not given.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    started = time.perf_counter()
    print(f"[DepthAnythingAccel] Fuse: loading graph {input_path.name}...", flush=True)
    model = onnx.load(str(input_path), load_external_data=True)
    if encoder is None:
        metadata = {entry.key: entry.value for entry in model.metadata_props}
        encoder = metadata.get("depthaccel.encoder")
        if encoder is None:
            raise ValueError(
                "The ONNX graph carries no 'depthaccel.encoder' metadata; "
                "pass --encoder explicitly."
            )
    if encoder not in DA2_ENCODERS:
        raise ValueError(f"Unsupported DA2 encoder in graph metadata: {encoder!r}")
    config = DA2_ENCODERS[encoder]
    num_heads = config["num_heads"]
    hidden_size = config["hidden_size"]

    print(
        f"[DepthAnythingAccel] Fuse: validating {config['num_blocks']} attention "
        f"blocks and rewriting ({encoder}, attention)...",
        flush=True,
    )
    before = operator_counts(model)
    fused_blocks = fuse_attention(
        model,
        num_heads,
        hidden_size,
        operator,
        num_blocks=config["num_blocks"],
    )
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata.update(
        {
            "depthaccel.architecture": metadata.get(
                "depthaccel.architecture", "depth-anything-v2"
            ),
            "depthaccel.encoder": encoder,
            "depthaccel.attention": f"com.microsoft::{operator}",
            "depthaccel.hidden_size": str(hidden_size),
            "depthaccel.num_heads": str(num_heads),
            "depthaccel.patch_size": "14",
            "depthaccel.precision": metadata.get("depthaccel.precision", "fp32"),
        }
    )
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    print(
        f"[DepthAnythingAccel] Fuse: validating fused graph and saving "
        f"({time.perf_counter() - started:.1f}s so far)...",
        flush=True,
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))

    print(f"Input: {input_path.resolve()}")
    print(f"Output: {output_path.resolve()}")
    print(f"Fused blocks: {fused_blocks}")
    print(f"Fused operator: {operator}")
    print(f"Operators before: {dict(sorted(before.items()))}")
    print(f"Operators after: {dict(sorted(operator_counts(model).items()))}")
    print(
        f"[DepthAnythingAccel] Fuse: done in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return output_path.resolve()


def main() -> None:
    args = parse_args()
    if args.encoder is None and (args.num_heads or args.hidden_size):
        raise ValueError(
            "--encoder is required together with --num-heads/--hidden-size overrides."
        )
    if args.encoder is not None:
        config = DA2_ENCODERS[args.encoder]
        num_heads = args.num_heads or config["num_heads"]
        hidden_size = args.hidden_size or config["hidden_size"]
        if num_heads != config["num_heads"] or hidden_size != config["hidden_size"]:
            raise ValueError(
                f"Overrides do not match DA2 {args.encoder}: expected "
                f"num_heads={config['num_heads']}, hidden_size={config['hidden_size']}"
            )
    fuse_attention_model(
        args.input,
        args.output,
        encoder=args.encoder,
        operator=args.operator,
    )


if __name__ == "__main__":
    main()
