import re

import pydantic
import pytest
import zarr

from geff.utils import validate


def test_validate(tmpdir):
    # Does not exist
    with pytest.raises(AssertionError, match=r"Directory .* does not exist"):
        validate("does-not-exist")

    zpath = tmpdir / "test.zarr"
    z = zarr.open(zpath)

    # Missing metadata
    with pytest.raises(pydantic.ValidationError):
        validate(zpath)
    z.attrs["geff_version"] = "v0.0.1"
    z.attrs["directed"] = True
    z.attrs["roi_min"] = [0, 0]
    z.attrs["roi_max"] = [100, 100]

    # No nodes
    with pytest.raises(AssertionError, match="graph group must contain a nodes group"):
        validate(zpath)
    z.create_group("nodes")

    # Nodes missing ids
    with pytest.raises(AssertionError, match="nodes group must contain an ids array"):
        validate(zpath)
    n_node = 10
    z["nodes"].create_dataset("ids", shape=(n_node))

    # Nodes missing position attrs
    with pytest.raises(AssertionError, match="nodes group must contain an attrs group"):
        validate(zpath)
    z["nodes"].create_group("attrs")
    with pytest.raises(AssertionError, match="nodes group must contain an attrs/position group"):
        validate(zpath)
    z["nodes"].create_group("attrs/position")
    with pytest.raises(
        AssertionError, match="node attribute group position must have values group"
    ):
        validate(zpath)
    z["nodes"].create_dataset("attrs/position/values", shape=(n_node))
    validate(zpath)

    # valid and invalid "missing" arrays for position attribute
    z["nodes"].create_dataset("attrs/position/missing", shape=(n_node), dtype=bool)
    with pytest.raises(AssertionError, match="position group cannot have missing values"):
        validate(zpath)
    del z["nodes/attrs/position"]["missing"]

    # Attr shape mismatch
    z["nodes"].create_dataset("attrs/badshape/values", shape=(n_node * 2))
    with pytest.raises(
        AssertionError,
        match=(
            f"Node attribute badshape values has length {n_node * 2}, "
            f"which does not match id length {n_node}"
        ),
    ):
        validate(zpath)

    del z["nodes/attrs"]["badshape"]
    # Attr missing shape mismatch
    z["nodes"].create_dataset("attrs/badshape/values", shape=(n_node))
    z["nodes"].create_dataset("attrs/badshape/missing", shape=(n_node * 2))
    with pytest.raises(
        AssertionError,
        match=(
            f"Node attribute badshape missing mask has length {n_node * 2}, "
            f"which does not match id length {n_node}"
        ),
    ):
        validate(zpath)
    del z["nodes/attrs"]["badshape"]

    # No edge group is okay, if the graph has no edges
    z.create_group("edges")

    # Missing edge ids
    with pytest.raises(AssertionError, match="edge group must contain ids array"):
        validate(zpath)

    # ids array must have last dim size 2
    n_edges = 5
    badshape = (n_edges, 3)
    z["edges"].create_dataset("ids", shape=badshape)
    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"edges ids must have a last dimension of size 2, received shape {badshape}"
        ),
    ):
        validate(zpath)
    del z["edges"]["ids"]
    z["edges"].create_dataset("ids", shape=(n_edges, 2))

    # Attr shape mismatch
    z["edges"].create_dataset("attrs/badshape/values", shape=(n_edges * 2, 2))
    with pytest.raises(
        AssertionError,
        match=(
            f"Edge attribute badshape values has length {n_edges * 2}, "
            f"which does not match id length {n_edges}"
        ),
    ):
        validate(zpath)
    del z["edges/attrs"]["badshape"]

    # Attr shape mismatch
    z["edges"].create_dataset("attrs/badshape/values", shape=(n_edges, 2))
    z["edges"].create_dataset("attrs/badshape/missing", shape=(n_edges * 2, 2))
    with pytest.raises(
        AssertionError,
        match=(
            f"Edge attribute badshape missing mask has length {n_edges * 2}, "
            f"which does not match id length {n_edges}"
        ),
    ):
        validate(zpath)
    del z["edges/attrs"]["badshape"]

    # everything passes
    validate(zpath)
