from pathlib import Path

import numpy as np
import pytest
import spatial_graph as sg
import zarr

import geff.spatial_graph

node_dtypes = ["int8", "uint8", "int16", "uint16"]
node_attr_dtypes = [
    {"position": "double[4]", "score": "float32"},
    {"position": "int[4]", "score": "float32"},
]
edge_attr_dtypes = [
    {"score": "float64", "color": "uint8"},
    {"score": "float32", "color": "int16"},
]


@pytest.mark.parametrize("node_dtype", node_dtypes)
@pytest.mark.parametrize("node_attr_dtypes", node_attr_dtypes)
@pytest.mark.parametrize("edge_attr_dtypes", edge_attr_dtypes)
@pytest.mark.parametrize("directed", [True, False])
def test_read_missing_attrs(tmp_path, node_dtype, node_attr_dtypes, edge_attr_dtypes, directed):
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
        score=np.array(
            [0.1, 0.5, 100.0, 1.0, 3.2],
            dtype=node_attr_dtypes["score"],
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
    zarr_path = Path(tmp_path) / "test.zarr/graph"
    geff.spatial_graph.write(graph, zarr_path)
    zroot = zarr.open(zarr_path)

    # construct missing masks that correspond to specific node/edge ids (regardless of
    # storage order)
    node_ids = list(zroot["nodes/ids"][:])
    num_nodes = len(node_ids)
    missing_nodes = [2, 4]
    missing_mask = np.zeros(shape=(num_nodes), dtype=bool)
    for node_id in missing_nodes:
        missing_mask[node_ids.index(node_id)] = 1

    node_score_group = zroot["nodes/attrs/score"]
    node_score_group.create_dataset("missing", shape=(num_nodes), dtype=bool)
    node_score_group["missing"][:] = missing_mask

    edge_ids = zroot["edges/ids"][:].tolist()
    num_edges = len(edge_ids)
    missing_edges = [
        [10, 2],
        [2, 127],
    ]
    missing_mask = np.zeros(shape=(num_edges), dtype=bool)
    for edge_id in missing_edges:
        missing_mask[edge_ids.index(edge_id)] = 1
    edge_score_group = zroot["edges/attrs/score"]
    edge_score_group.create_dataset("missing", shape=(num_edges), dtype=bool)
    edge_score_group["missing"][:] = missing_mask

    with pytest.raises(
        ValueError,
        match="Nodes attribute score has missing values, must provide default value to "
        "read with spatial graph",
    ):
        geff.spatial_graph.read(f"{tmp_path}/test.zarr/graph")

    default_node_attr_values = {"score": 0}
    with pytest.raises(
        ValueError,
        match="Edges attribute score has missing values, must provide default value to "
        "read with spatial graph",
    ):
        geff.spatial_graph.read(
            f"{tmp_path}/test.zarr/graph",
            default_node_attr_values=default_node_attr_values,
        )

    default_node_attr_values = {"score": 0}
    default_edge_attr_values = {"score": 0}
    compare = geff.spatial_graph.read(
        f"{tmp_path}/test.zarr/graph",
        default_node_attr_values=default_node_attr_values,
        default_edge_attr_values=default_edge_attr_values,
    )

    np.testing.assert_almost_equal([0.1, 0.0, 100.0, 0.0, 3.2], compare.node_attrs[nodes].score)
    np.testing.assert_almost_equal([0.0, 0.0, 0.3, 0.4], compare.edge_attrs[edges].score)
