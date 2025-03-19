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

    # No graph group
    with pytest.raises(AssertionError, match="geff zarr must contain a graph group"):
        validate(zpath)
    z.create_group("graph")

    # Missing metadata
    with pytest.raises(pydantic.ValidationError):
        validate(zpath)
    z["graph"].attrs["geff_version"] = "v0.0.1"
    z["graph"].attrs["directed"] = True
    z["graph"].attrs["roi_min"] = [0, 0]
    z["graph"].attrs["roi_max"] = [100, 100]

    # No nodes
    with pytest.raises(AssertionError, match="graph group must contain a nodes group"):
        validate(zpath)
    z["graph"].create_group("nodes")

    # Nodes missing ids
    with pytest.raises(AssertionError, match="nodes group must contain an ids array"):
        validate(zpath)
    n_node = 10
    z["graph/nodes"].create_dataset("ids", shape=(n_node))

    # Nodes missing position attrs
    with pytest.raises(AssertionError, match="nodes group must contain an attrs/position array"):
        validate(zpath)
    z["graph/nodes"].create_dataset("attrs/position", shape=(n_node))

    # Attr shape mismatch
    z["graph/nodes"].create_dataset("attrs/badshape", shape=(n_node * 2))
    with pytest.raises(
        AssertionError,
        match=(
            f"Node attribute badshape has length {n_node * 2}, "
            f"which does not match id length {n_node}"
        ),
    ):
        validate(zpath)
    del z["graph/nodes/attrs"]["badshape"]

    # Edges missing
    with pytest.raises(AssertionError, match="graph group must contain an edge group"):
        validate(zpath)
    z["graph"].create_group("edges")

    # Missing edge ids
    with pytest.raises(AssertionError, match="edge group must contain ids array"):
        validate(zpath)

    # ids array must have last dim size 2
    badshape = (5, 3)
    z["graph/edges"].create_dataset("ids", shape=(5, 3))
    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"edges ids must have a last dimension of size 2, received shape {badshape}"
        ),
    ):
        validate(zpath)
    del z["graph/edges"]["ids"]
    z["graph/edges"].create_dataset("ids", shape=(5, 2))

    # everything passes
    validate(zpath)
