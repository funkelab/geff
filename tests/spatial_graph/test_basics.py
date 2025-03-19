import numpy as np
import pytest
import spatial_graph as sg

import geff.spatial_graph

node_dtypes = ["int8", "uint8", "int16", "uint16"]
node_attr_dtypes = [
    {"position": "double[4]"},
    {"position": "int[4]"},
]
edge_attr_dtypes = [
    {"score": "float64", "color": "uint8"},
    {"score": "float32", "color": "int16"},
]


@pytest.mark.parametrize("node_dtype", node_dtypes)
@pytest.mark.parametrize("node_attr_dtypes", node_attr_dtypes)
@pytest.mark.parametrize("edge_attr_dtypes", edge_attr_dtypes)
@pytest.mark.parametrize("directed", [True, False])
def test_read_write_consistency(node_dtype, node_attr_dtypes, edge_attr_dtypes, directed):
    graph = sg.SpatialGraph(
        ndims=4,
        node_dtype=node_dtype,
        node_attr_dtypes=node_attr_dtypes,
        edge_attr_dtypes=edge_attr_dtypes,
        position_attr="position",
        directed=directed,
    )

    nodes = np.array([10, 2, 127, 4, 5], dtype=graph.node_dtype)
    graph.add_nodes(
        nodes,
        position=np.array(
            [
                [0.1, 0.5, 100.0, 1.0],
                [0.2, 0.4, 200.0, 0.1],
                [0.3, 0.3, 300.0, 0.1],
                [0.4, 0.2, 400.0, 0.1],
                [0.5, 0.1, 500.0, 0.1],
            ],
            dtype=graph.coord_dtype,
        ),
    )

    edges = np.array(
        [
            [10, 2],
            [2, 127],
            [2, 4],
            [4, 5],
        ],
        dtype=graph.node_dtype,
    )
    graph.add_edges(
        edges,
        score=np.array([0.1, 0.2, 0.3, 0.4], dtype=edge_attr_dtypes["score"]),
        color=np.array([1, 2, 3, 4], dtype=edge_attr_dtypes["color"]),
    )

    geff.spatial_graph.write(graph, "rw_consistency.zarr/graph")

    compare = geff.spatial_graph.read("rw_consistency.zarr/graph")

    np.testing.assert_equal(graph.nodes, compare.nodes)
    np.testing.assert_equal(graph.edges, compare.edges)
    np.testing.assert_equal(graph.node_attrs[nodes].position, compare.node_attrs[nodes].position)
    np.testing.assert_equal(graph.edge_attrs[edges].score, compare.edge_attrs[edges].score)
    np.testing.assert_equal(graph.edge_attrs[edges].color, compare.edge_attrs[edges].color)
