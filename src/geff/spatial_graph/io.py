from pathlib import Path
from typing import Optional

import numpy as np
import spatial_graph as sg
import zarr

import geff
import geff.utils


def write(
    graph: sg.SpatialGraph,
    path: str | Path,
    axis_names: Optional[list[str]] = None,
    axis_units: Optional[list[str]] = None,
):
    # open/create zarr container
    group = zarr.open(path, "a")

    # write meta-data
    group.attrs["geff_version"] = geff.__version__
    group.attrs["position_attr"] = graph.position_attr
    group.attrs["directed"] = graph.directed
    group.attrs["roi_min"] = tuple(graph.roi[0].tolist())
    group.attrs["roi_max"] = tuple(graph.roi[1].tolist())
    if axis_names:
        group.attrs["axis_names"] = axis_names
    if axis_units:
        group.attrs["axis_units"] = axis_units

    # get node and edge IDs
    nodes = graph.nodes
    edges = graph.edges

    # write nodes
    group["nodes/ids"] = nodes

    # write node attributes
    for name in graph.node_attr_dtypes.keys():
        group[f"nodes/attrs/{name}/values"] = graph.node_attrs[nodes].__getattr__(name)

    # write edges
    group["edges/ids"] = graph.edges

    # write edge attributes
    for name in graph.edge_attr_dtypes.keys():
        group[f"edges/attrs/{name}/values"] = graph.edge_attrs[edges].__getattr__(name)


def _read_attr_values(
    geff_group: zarr.Group, attr_type: str, default_attr_values: dict[str, float] | None
) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    """Read the values for a set of node or edge attributes, including checking for
    missing values and either throwing an error or using the provided default.

    Args:
        geff_group (zarr.Group): The zarr group containing the geff_version in .zattrs
        attr_type (str): "nodes" or "edges"
        default_attr_values (dict[str, float]): A dictionary mapping attribute names
            to default values to use if the attribute is missing

    Raises:
        ValueError: If an attribute has missing values and no default value for it is
            provided

    Returns:
        tuple[dict[str, str], dict[str, np.ndarray]]: A mapping from attribute names to
            dtypes, and a mapping rom attribute names to values as numpy arrays
    """
    attr_dtypes = {}
    attrs = {}
    assert attr_type in ["nodes", "edges"]
    for name in geff_group[f"{attr_type}/attrs"]:
        attrs_group = geff_group[f"{attr_type}/attrs/{name}"]
        values = attrs_group["values"][:]
        if "missing" in attrs_group.array_keys():
            missing = attrs_group["missing"][:]
            if np.any(missing):
                if default_attr_values is None or name not in default_attr_values:
                    raise ValueError(
                        f"{attr_type.capitalize()} attribute {name} has missing values, "
                        "must provide default value to read with spatial graph"
                    )
                values[missing] = default_attr_values[name]
        dtype = values.dtype
        size = values.shape
        dtype_str = str(dtype) + ("" if len(size) == 1 else f"[{size[1]}]")
        attr_dtypes[name] = dtype_str
        attrs[name] = values
    return attr_dtypes, attrs


def read(
    path: Path | str,
    validate: bool = True,
    default_node_attr_values: dict[str, float] | None = None,
    default_edge_attr_values: dict[str, float] | None = None,
):
    if validate:
        geff.utils.validate(path)
    # open zarr container
    group = zarr.open(path, "r")

    # read meta-data
    position_attr = group.attrs["position_attr"]
    directed = group.attrs["directed"]

    ndims = group[f"nodes/attrs/{position_attr}/values"].shape[1]
    nodes = group["nodes/ids"][:]
    edges = group["edges/ids"][:]
    node_dtype = str(nodes.dtype)

    # collect node attributes
    node_attr_dtypes, node_attrs = _read_attr_values(group, "nodes", default_node_attr_values)
    # collect edge attributes
    edge_attr_dtypes, edge_attrs = _read_attr_values(group, "edges", default_edge_attr_values)
    # create graph
    graph = sg.SpatialGraph(
        ndims=ndims,
        node_dtype=node_dtype,
        node_attr_dtypes=node_attr_dtypes,
        edge_attr_dtypes=edge_attr_dtypes,
        position_attr=position_attr,
        directed=directed,
    )

    # add nodes
    graph.add_nodes(nodes, **node_attrs)

    # add edges
    graph.add_edges(edges, **edge_attrs)

    return graph
