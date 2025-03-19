import warnings
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import zarr

import geff


def get_roi(graph: nx.Graph, position_attr: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Get the roi of a networkx graph.

    Args:
        graph (nx.Graph): A non-empty networkx graph
        position_attr (str): All nodes on graph have this attribute holding their position

    Returns:
        tuple[tuple[float, ...], tuple[float, ...]]: A tuple with the min values in each
            spatial dim, and a tuple with the max values in each spatial dim
    """
    _min = None
    _max = None
    for _, data in graph.nodes(data=True):
        pos = np.array(data[position_attr])
        if _min is None:
            _min = pos
            _max = pos
        else:
            _min = np.min([_min, pos], axis=0)
            _max = np.max([_max, pos], axis=0)

    return tuple(_min.tolist()), tuple(_max.tolist())  # type: ignore


def get_node_attrs(graph: nx.Graph) -> list[str]:
    """Get the attribute keys present on any node in the networkx graph. Does not imply
    that the attributes are present on all nodes.

    Args:
        graph (nx.Graph): a networkx graph

    Returns:
        list[str]: A list of all unique node attribute keys
    """
    return list({k for n in graph.nodes for k in graph.nodes[n]})


def get_edge_attrs(graph: nx.Graph) -> list[str]:
    """Get the attribute keys present on any edge in the networkx graph. Does not imply
    that the attributes are present on all edges.

    Args:
        graph (nx.Graph): a networkx graph

    Returns:
        list[str]: A list of all unique edge attribute keys
    """
    return list({k for e in graph.edges for k in graph.edges[e]})


def write(
    graph: nx.Graph,
    position_attr: str,
    path: str | Path,
    axis_names: Optional[list[str]] = None,
    axis_units: Optional[list[str]] = None,
):
    """Write a networkx graph to the geff file format

    Args:
        graph (nx.Graph): a networkx graph where every node has a position attribute
        position_attr (str): the name of the position attribute present on every node
        path (str | Path): the path to the output zarr. Opens in append mode,
            so will only overwrite geff-controlled groups.
        axis_names (Optional[list[str]], optional): The names of the spatial dims
            represented in position attribute. Defaults to None.
        axis_units (Optional[list[str]], optional): The units of the spatial dims
            represented in position attribute. Defaults to None.
    """
    if nx.is_empty(graph):
        warnings.warn(f"Graph is empty - not writing anything to {path}", stacklevel=2)
        return
    # open/create zarr container
    group = zarr.open(path, "a")

    # write meta-data
    group.attrs["geff_spec"] = geff.__version__
    group.attrs["position_attr"] = position_attr
    group.attrs["directed"] = isinstance(graph, nx.DiGraph)
    group.attrs["roi"] = get_roi(graph, position_attr=position_attr)
    if axis_names:
        graph.attrs["axis_names"] = axis_names
    if axis_units:
        graph.attrs["axis_units"] = axis_units

    # get node and edge IDs
    nodes_list = list(graph.nodes())
    nodes_arr = np.array(nodes_list)
    edges_list = list(graph.edges())
    edges_arr = np.array(edges_list)

    # write nodes
    group["nodes/ids"] = nodes_arr

    # write node attributes
    for name in get_node_attrs(graph):
        # TODO: handle missing values
        group[f"nodes/attrs/{name}"] = np.array([graph.nodes[node][name] for node in nodes_list])

    # write edges
    group["edges/ids"] = edges_arr

    # write edge attributes
    for name in get_edge_attrs(graph):
        group[f"edges/attrs/{name}"] = np.array([graph.edges[edge][name] for edge in edges_list])


def read(path: Path) -> nx.Graph:
    """Read a geff file into a networkx graph.

    Args:
        path (Path): The path to the root of the geff zarr, where the .attrs contains
            the geff  metadata
    Returns:
        nx.Graph: The graph that was stored in the geff file format
    """
    # open zarr container
    path = Path(path)
    assert path.exists(), f"Cannot read graph from {path} because it does not exist"
    group = zarr.open(path, "r")

    # TODO: validate the metadata using the schema

    # read meta-data
    directed = group.attrs["directed"]
    graph = nx.DiGraph() if directed else nx.Graph()

    nodes = group["nodes/ids"][:]
    graph.add_nodes_from(nodes.tolist())
    edges = group["edges/ids"][:]
    graph.add_edges_from(edges.tolist())

    # collect node attributes
    for name in group["nodes/attrs"]:
        ds = group[f"nodes/attrs/{name}"]
        for node, val in zip(nodes, ds[:]):
            val = val.tolist() if val.size > 1 else val.item()
            graph.nodes[node.item()][name] = val

    # collect edge attributes]
    for name in group["edges/attrs"]:
        ds = group[f"edges/attrs/{name}"]
        for edge, val in zip(edges, ds[:]):
            val = val.tolist() if val.size > 1 else val.item()
            graph.edges[*edge.tolist()][name] = val

    return graph
