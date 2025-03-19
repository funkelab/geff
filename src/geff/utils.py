import os

import zarr

from .metadata_schema import GeffMetadata


def validate(path: str):
    """Check that the structure of the zarr conforms to geff specification

    Args:
        path (str): Path to geff zarr
    """
    # Check that directory exists
    assert os.path.exists(path), f"Directory {path} does not exist"

    z = zarr.open(path, mode="r")

    assert "graph" in z, "geff zarr must contain a graph group"
    graph = z["graph"]

    # graph attrs validation
    # Raises pydantic.ValidationError or ValueError
    GeffMetadata(**graph.attrs)

    assert "nodes" in graph, "graph group must contain a nodes group"
    nodes = graph["nodes"]

    # ids and attrs/position are required and should be same length
    assert "ids" in nodes, "nodes group must contain an ids array"
    assert "attrs/position" in nodes, "nodes group must contain an attrs/position array"

    # Attribute array length should match id length
    for attr in nodes["attrs"].keys():
        attr_len = nodes["attrs"][attr].shape[0]
        id_len = nodes["ids"].shape[0]
        assert (
            attr_len == id_len
        ), f"Node attribute {attr} has length {attr_len}, which does not match id length {id_len}"

    assert "edges" in graph, "graph group must contain an edge group"
    edges = graph["edges"]

    # Edges only require ids which contain nodes for each edge
    assert "ids" in edges, "edge group must contain ids array"
    id_shape = edges["ids"].shape
    assert (
        id_shape[-1] == 2
    ), f"edges ids must have a last dimension of size 2, received shape {id_shape}"
