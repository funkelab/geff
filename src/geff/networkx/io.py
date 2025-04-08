from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import zarr

import geff
import geff.utils
from geff.metadata_schema import GeffMetadata

if TYPE_CHECKING:
    from pathlib import Path


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
    axis_names: list[str] | None = None,
    axis_units: list[str] | None = None,
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
    if graph.number_of_nodes() == 0:
        warnings.warn(f"Graph is empty - not writing anything to {path}", stacklevel=2)
        return
    # open/create zarr container
    group = zarr.open(path, "a")

    # write meta-data
    group.attrs["geff_version"] = geff.__version__
    group.attrs["position_attr"] = position_attr
    group.attrs["directed"] = isinstance(graph, nx.DiGraph)
    roi_min, roi_max = get_roi(graph, position_attr=position_attr)
    group.attrs["roi_min"] = roi_min
    group.attrs["roi_max"] = roi_max
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
        values = []
        missing = []
        for node in nodes_list:
            if name in graph.nodes[node]:
                value = graph.nodes[node][name]
                mask = 0
            else:
                if name == position_attr:
                    raise ValueError(f"Missing position attr for node {node}")
                value = 0
                mask = 1
            values.append(value)
            missing.append(mask)
        if name == position_attr:
            name = "position"
        else:
            group[f"nodes/attrs/{name}/missing"] = np.array(missing, dtype=bool)
        group[f"nodes/attrs/{name}/values"] = np.array(values)

    # write edges
    if len(edges_list) > 0:
        group["edges/ids"] = edges_arr

        # write edge attributes
        for name in get_edge_attrs(graph):
            values = []
            missing = []
            for edge in edges_list:
                if name in graph.edges[edge]:
                    value = graph.edges[edge][name]
                    mask = 0
                else:
                    if name == position_attr:
                        raise ValueError(f"Missing position attr for edge {edge}")
                    value = 0
                    mask = 1
                values.append(value)
                missing.append(mask)
            group[f"edges/attrs/{name}/missing"] = np.array(missing, dtype=bool)
            group[f"edges/attrs/{name}/values"] = np.array(values)


def read(path: Path | str, validate: bool = True) -> nx.Graph:
    """Read a geff file into a networkx graph.

    Args:
        path (Path): The path to the root of the geff zarr, where the .attrs contains
            the geff  metadata
    Returns:
        nx.Graph: The graph that was stored in the geff file format
    """
    # open zarr container
    if validate:
        geff.utils.validate(path)

    group = zarr.open(path, "r")
    metadata = GeffMetadata(**group.attrs)

    # read meta-data
    graph = nx.DiGraph() if metadata.directed else nx.Graph()

    nodes = group["nodes/ids"][:]
    graph.add_nodes_from(nodes.tolist())

    # collect node attributes
    for name in group["nodes/attrs"]:
        attr_group = group[f"nodes/attrs/{name}"]
        values = attr_group["values"][:]
        sparse = "missing" in attr_group.array_keys()
        if sparse:
            missing = attr_group["missing"][:]
        for idx in range(len(nodes)):
            node = nodes[idx]
            val = values[idx]
            ignore = missing[idx] if sparse else False
            if not ignore:
                val = val.tolist() if val.size > 1 else val.item()
                graph.nodes[node.item()][name] = val

    if "edges" in group.group_keys():
        edges = group["edges/ids"][:]
        graph.add_edges_from(edges.tolist())

        # collect edge attributes
        for name in group["edges/attrs"]:
            attr_group = group[f"edges/attrs/{name}"]
            values = attr_group["values"][:]
            if sparse:
                missing = attr_group["missing"][:]
            for idx in range(edges.shape[0]):
                edge = edges[idx]
                val = values[idx]
                ignore = missing[idx] if sparse else False
                if not ignore:
                    val = val.tolist() if val.size > 1 else val.item()
                    source, target = edge.tolist()
                    graph.edges[source, target][name] = val

    return graph
