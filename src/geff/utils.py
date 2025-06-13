from __future__ import annotations

import os
from typing import TYPE_CHECKING

import zarr

from .metadata_schema import GeffMetadata

if TYPE_CHECKING:
    from pathlib import Path


def validate(path: str | Path):
    """Check that the structure of the zarr conforms to geff specification

    Args:
        path (str | Path): Path to geff zarr
    """
    # Check that directory exists
    assert os.path.exists(path), f"Directory {path} does not exist"

    # zarr python 3 doesn't support Path
    path = str(path)

    graph = zarr.open(path, mode="r")

    # graph attrs validation
    # Raises pydantic.ValidationError or ValueError
    meta = GeffMetadata(**graph.attrs)

    assert "nodes" in graph, "graph group must contain a nodes group"
    nodes = graph["nodes"]

    # ids and attrs/position are required and should be same length
    assert "ids" in nodes.array_keys(), "nodes group must contain an ids array"
    assert "attrs" in nodes.group_keys(), "nodes group must contain an attrs group"

    if meta.position_attr is not None:
        assert meta.position_attr in nodes["attrs"].group_keys(), (
            "nodes group must contain an attrs/position group"
        )
        assert "missing" not in nodes[f"attrs/{meta.position_attr}"].array_keys(), (
            "position group cannot have missing values"
        )

    # Attribute array length should match id length
    id_len = nodes["ids"].shape[0]
    for attr in nodes["attrs"].keys():
        attr_group = nodes["attrs"][attr]
        assert "values" in attr_group.array_keys(), (
            f"node attribute group {attr} must have values group"
        )
        attr_len = attr_group["values"].shape[0]
        assert attr_len == id_len, (
            f"Node attribute {attr} values has length {attr_len}, which does not match "
            f"id length {id_len}"
        )
        if "missing" in attr_group.array_keys():
            missing_len = attr_group["missing"].shape[0]
            assert missing_len == id_len, (
                f"Node attribute {attr} missing mask has length {missing_len}, which "
                f"does not match id length {id_len}"
            )

    if "edges" in graph.group_keys():
        edges = graph["edges"]

        # Edges only require ids which contain nodes for each edge
        assert "ids" in edges, "edge group must contain ids array"
        id_shape = edges["ids"].shape
        assert id_shape[-1] == 2, (
            f"edges ids must have a last dimension of size 2, received shape {id_shape}"
        )

        # Edge attribute array length should match edge id length
        edge_id_len = edges["ids"].shape[0]
        if "attrs" in edges:
            for attr in edges["attrs"].keys():
                attr_group = edges["attrs"][attr]
                assert "values" in attr_group.array_keys(), (
                    f"Edge attribute group {attr} must have values group"
                )
                attr_len = attr_group["values"].shape[0]
                assert attr_len == edge_id_len, (
                    f"Edge attribute {attr} values has length {attr_len}, which does not "
                    f"match id length {edge_id_len}"
                )
                if "missing" in attr_group.array_keys():
                    missing_len = attr_group["missing"].shape[0]
                    assert missing_len == edge_id_len, (
                        f"Edge attribute {attr} missing mask has length {missing_len}, "
                        f"which does not match id length {edge_id_len}"
                    )
