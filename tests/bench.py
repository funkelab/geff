from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import geff.networkx.io as geff_nx
from geff.utils import validate

ROUNDS = 3
N_NODES = 2000


@pytest.fixture(scope="session")
def big_graph():
    graph = nx.DiGraph()

    nodes = np.arange(N_NODES)  # int
    positions = np.random.uniform(size=(N_NODES, 4))  # float
    for node, pos in zip(nodes, positions):
        graph.add_node(node.item(), position=pos.tolist())

    float_attr = np.random.uniform(size=(N_NODES * N_NODES))
    int_attr = np.arange(N_NODES * N_NODES)
    for i, (source, target) in enumerate(product(nodes, nodes)):
        if source != target:
            graph.add_edge(source, target, float_attr=float_attr[i], int_attr=int_attr[i])

    print("N nodes", len(graph.nodes))
    print("N edges", len(graph.edges))

    return graph


@pytest.fixture(scope="session")
def big_graph_path(tmpdir_factory, big_graph):
    tmp_path = Path(tmpdir_factory.mktemp("data").join("test.zarr"))
    geff_nx.write_nx(graph=big_graph, path=tmp_path, position_attr="position")
    return tmp_path


def test_write(benchmark, tmp_path, big_graph):
    path = tmp_path / "test_write.zarr"

    benchmark.pedantic(
        geff_nx.write_nx,
        kwargs={"graph": big_graph, "position_attr": "position", "path": path},
        rounds=ROUNDS,
    )


def test_validate(benchmark, big_graph_path):
    benchmark.pedantic(validate, kwargs={"path": big_graph_path}, rounds=ROUNDS)


def test_read(benchmark, big_graph_path):
    benchmark.pedantic(
        geff_nx.read_nx, kwargs={"path": big_graph_path, "validate": False}, rounds=ROUNDS
    )
