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


def write_nx(
    graph: nx.Graph,
    position_attr: str,
    path: str | Path,
    axis_names: list[str] | None = None,
    axis_units: list[str] | None = None,
    zarr_format: int = 3,
    validate: bool = True,
):
    """Write a networkx graph to the geff file format

    Args:
        graph (nx.Graph): a networkx graph where every node has a position attribute
        position_attr (str): the name of the position attribute present on every node
        path (str | Path): the path to the output zarr. Opens in append mode,
            so will only overwrite geff-controlled groups.
        axis_names (Optional[list[str]], optional): The names of the spatial dims
            represented in position attribute. Defaults to None. Will override
            value in graph attributes if provided.
        axis_units (Optional[list[str]], optional): The units of the spatial dims
            represented in position attribute. Defaults to None. Will override value
            in graph attributes if provided.
        zarr_format (Optional[int], optional): The version of zarr to write.
            Defaults to 3.
        validate (bool, optional): Flag indicating whether to perform validation on the
            networkx graph before writing anything to disk. If set to False and there are
            missing attributes, will likely fail with a KeyError, leading to an incomplete
            graph written to disk. Defaults to True.
    """
    if graph.number_of_nodes() == 0:
        warnings.warn(f"Graph is empty - not writing anything to {path}", stacklevel=2)
        return

    # open/create zarr container
    if zarr.__version__.startswith("3"):
        group = zarr.open(path, mode="a", zarr_format=zarr_format)
    else:
        group = zarr.open(path, mode="a")

    node_attrs = get_node_attrs(graph)
    if validate:
        if position_attr not in node_attrs:
            raise ValueError(f"Position attribute {position_attr} not found in graph")
        for node, data in graph.nodes(data=True):
            if position_attr not in data:
                raise ValueError(f"Node {node} does not have position attribute {position_attr}")

    # write metadata
    roi_min, roi_max = get_roi(graph, position_attr=position_attr)
    metadata = GeffMetadata(
        geff_version=geff.__version__,
        directed=isinstance(graph, nx.DiGraph),
        roi_min=roi_min,
        roi_max=roi_max,
        position_attr=position_attr,
        axis_names=axis_names if axis_names is not None else graph.graph.get("axis_names", None),
        axis_units=axis_units if axis_units is not None else graph.graph.get("axis_units", None),
    )
    metadata.write(group)

    # get node and edge IDs
    nodes_list = list(graph.nodes())
    nodes_arr = np.array(nodes_list)
    edges_list = list(graph.edges())
    edges_arr = np.array(edges_list)

    # write nodes
    group["nodes/ids"] = nodes_arr

    # write node attributes
    for name in node_attrs:
        values = []
        missing = []
        for node in nodes_list:
            if name in graph.nodes[node]:
                value = graph.nodes[node][name]
                mask = 0
            else:
                value = 0
                mask = 1
            values.append(value)
            missing.append(mask)
        # Set position attribute to default "position", original stored in metadata
        if name == position_attr:
            name = "position"
        else:
            # Always store missing array even if all values are present
            group[f"nodes/attrs/{name}/missing"] = np.array(missing, dtype=bool)
        group[f"nodes/attrs/{name}/values"] = np.array(values)

    # write edges
    # Edge group is only created if edges are present on graph
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
                    value = 0
                    mask = 1
                values.append(value)
                missing.append(mask)
            group[f"edges/attrs/{name}/missing"] = np.array(missing, dtype=bool)
            group[f"edges/attrs/{name}/values"] = np.array(values)


def _set_attribute_values(
    graph: nx.DiGraph, ids: np.ndarray, graph_group: zarr.Group, name: str, nodes: bool = True
) -> None:
    """Add attributes in-place to a networkx graph's nodes or edges.

    Args:
        graph (nx.DiGraph): The networkx graph, already populated with nodes or edges,
            that needs attributes added
        ids (np.ndarray): Node or edge ids from Geff. If nodes, 1D. If edges, 2D.
        graph_group (zarr.Group): A zarr group holding the geff graph.
        name (str): The name of the attribute
        nodes (bool, optional): If True, extract and set node attributes.  If False,
            extract and set edge attributes. Defaults to True.
    """
    element = "nodes" if nodes else "edges"
    attr_group = graph_group[f"{element}/attrs/{name}"]
    values = attr_group["values"][:]
    sparse = "missing" in attr_group.array_keys()
    if sparse:
        missing = attr_group["missing"][:]
    for idx in range(len(ids)):
        _id = ids[idx]
        val = values[idx]
        # If attribute is sparse and missing for this node, skip setting attribute
        ignore = missing[idx] if sparse else False
        if not ignore:
            # Get either individual item or list instead of setting with np.array
            val = val.tolist() if val.size > 1 else val.item()
            if nodes:
                if name == "position" and "position_attr" in graph.graph:
                    graph_attr = graph.graph["position_attr"]
                else:
                    graph_attr = name
                graph.nodes[_id.item()][graph_attr] = val
            else:
                source, target = _id.tolist()
                graph.edges[source, target][name] = val


def read_nx(path: Path | str, validate: bool = True) -> nx.Graph:
    """Read a geff file into a networkx graph. Metadata attributes will be stored in
    the graph attributes, accessed via `G.graph[key]` where G is a networkx graph.

    Args:
        path (Path | str): The path to the root of the geff zarr, where the .attrs contains
            the geff  metadata
    Returns:
        nx.Graph: The graph that was stored in the geff file format
    """
    # zarr python 3 doesn't support Path
    path = str(path)

    # open zarr container
    if validate:
        geff.utils.validate(path)

    group = zarr.open(path, mode="r")
    metadata = GeffMetadata.read(group)

    # read meta-data
    graph = nx.DiGraph() if metadata.directed else nx.Graph()
    for key, val in metadata:
        graph.graph[key] = val

    nodes = group["nodes/ids"][:]
    graph.add_nodes_from(nodes.tolist())

    # collect node attributes
    for name in group["nodes/attrs"]:
        _set_attribute_values(graph, nodes, group, name, nodes=True)

    if "edges" in group.group_keys():
        edges = group["edges/ids"][:]
        graph.add_edges_from(edges.tolist())

        # collect edge attributes if they exist
        if "edges/attrs" in group:
            for name in group["edges/attrs"]:
                _set_attribute_values(graph, edges, group, name, nodes=False)

    return graph
