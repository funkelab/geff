from pathlib import Path
from typing import Optional

import spatial_graph as sg
import zarr

import geff


def write(
    graph: sg.SpatialGraph,
    path: str | Path,
    axis_names: Optional[list[str]] = None,
    axis_units: Optional[list[str]] = None,
):
    # open/create zarr container
    group = zarr.open(path, "a")

    # write meta-data
    group.attrs["geff_spec"] = geff.__version__
    group.attrs["position_attr"] = graph.position_attr
    group.attrs["directed"] = graph.directed
    group.attrs["roi"] = (
        tuple(graph.roi[0].tolist()),
        tuple(graph.roi[1].tolist()),
    )
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
        group[f"nodes/attrs/{name}"] = graph.node_attrs[nodes].__getattr__(name)

    # write edges
    group["edges/ids"] = graph.edges

    # write edge attributes
    for name in graph.edge_attr_dtypes.keys():
        group[f"edges/attrs/{name}"] = graph.edge_attrs[edges].__getattr__(name)


def read(path):
    # open zarr container
    group = zarr.open(path, "r")

    # read meta-data
    position_attr = group.attrs["position_attr"]
    directed = group.attrs["directed"]

    ndims = group[f"nodes/attrs/{position_attr}"].shape[1]
    nodes = group["nodes/ids"][:]
    edges = group["edges/ids"][:]
    node_dtype = str(nodes.dtype)

    # collect node attributes
    node_attr_dtypes = {}
    node_attrs = {}
    for name in group["nodes/attrs"]:
        ds = group[f"nodes/attrs/{name}"]
        dtype = ds.dtype
        size = ds.shape
        dtype_str = str(dtype) + ("" if len(size) == 1 else f"[{size[1]}]")
        node_attr_dtypes[name] = dtype_str
        node_attrs[name] = ds[:]

    # collect edge attributes
    edge_attr_dtypes = {}
    edge_attrs = {}
    for name in group["edges/attrs"]:
        ds = group[f"edges/attrs/{name}"]
        dtype = ds.dtype
        size = ds.shape
        dtype_str = str(dtype) + ("" if len(size) == 1 else f"[{size[1]}]")
        edge_attr_dtypes[name] = dtype_str
        edge_attrs[name] = ds[:]

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
