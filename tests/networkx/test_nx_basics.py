import networkx as nx
import numpy as np
import pytest

import geff.networkx

node_dtypes = ["int8", "uint8", "int16", "uint16"]
node_attr_dtypes = [
    {"position": "double"},
    {"position": "int"},
]
edge_attr_dtypes = [
    {"score": "float64", "color": "uint8"},
    {"score": "float32", "color": "int16"},
]

# TODO: mixed dtypes?
# TODO: missing values?


@pytest.mark.parametrize("node_dtype", node_dtypes)
@pytest.mark.parametrize("node_attr_dtypes", node_attr_dtypes)
@pytest.mark.parametrize("edge_attr_dtypes", edge_attr_dtypes)
@pytest.mark.parametrize("directed", [True, False])
def test_read_write_consistency(tmpdir, node_dtype, node_attr_dtypes, edge_attr_dtypes, directed):
    graph = nx.DiGraph() if directed else nx.Graph()

    nodes = np.array([10, 2, 127, 4, 5], dtype=node_dtype)
    positions = np.array(
        [
            [0.1, 0.5, 100.0, 1.0],
            [0.2, 0.4, 200.0, 0.1],
            [0.3, 0.3, 300.0, 0.1],
            [0.4, 0.2, 400.0, 0.1],
            [0.5, 0.1, 500.0, 0.1],
        ],
        dtype=node_attr_dtypes["position"],
    )
    for node, pos in zip(nodes, positions):
        graph.add_node(node.item(), position=pos.tolist())

    edges = np.array(
        [
            [10, 2],
            [2, 127],
            [2, 4],
            [4, 5],
        ],
        dtype=node_dtype,
    )
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=edge_attr_dtypes["score"])
    colors = np.array([1, 2, 3, 4], dtype=edge_attr_dtypes["color"])
    for edge, score, color in zip(edges, scores, colors):
        graph.add_edge(*edge.tolist(), score=score.item(), color=color.item())

    path = tmpdir / "rw_consistency.zarr/graph"

    geff.networkx.write(graph, "position", path)

    compare = geff.networkx.read(path)

    assert set(graph.nodes) == set(compare.nodes)
    assert set(graph.edges) == set(compare.edges)
    for node in nodes:
        assert graph.nodes[node.item()]["position"] == compare.nodes[node.item()]["position"]

    for edge in edges:
        assert graph.edges[edge.tolist()]["score"] == compare.edges[edge.tolist()]["score"]
        assert graph.edges[edge.tolist()]["color"] == compare.edges[edge.tolist()]["color"]
