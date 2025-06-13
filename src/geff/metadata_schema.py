from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path

import yaml
import zarr
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

import geff

with (files(geff) / "supported_versions.yml").open() as f:
    SUPPORTED_VERSIONS = yaml.safe_load(f)["versions"]


def _get_versions_regex(versions: list[str]):
    return r"|".join([rf"({re.escape(version)})" for version in versions])


SUPPORTED_VERSIONS_REGEX = _get_versions_regex(SUPPORTED_VERSIONS)


class GeffMetadata(BaseModel):
    """
    Geff metadata schema to validate the attributes json file in a geff zarr
    """

    # this determines the title of the generated json schema
    model_config = ConfigDict(title="geff_metadata")

    geff_version: str = Field(pattern=SUPPORTED_VERSIONS_REGEX)
    directed: bool
    roi_min: tuple[float, ...] | None = None
    roi_max: tuple[float, ...] | None = None
    position_attr: str | None = None
    axis_names: tuple[str, ...] | None = None
    axis_units: tuple[str, ...] | None = None

    def model_post_init(self, *args, **kwargs):
        # Check spatial metadata only if position is provided
        if self.position_attr is not None:
            # Check that rois are there if position provided
            if self.roi_min is None or self.roi_max is None:
                raise ValueError(
                    f"Position attribute {self.position_attr} has been specified, "
                    "but roi_min and/or roi_max are not specified."
                )

            if len(self.roi_min) != len(self.roi_max):
                raise ValueError(
                    f"Roi min {self.roi_min} and roi max {self.roi_max} have different lengths."
                )
            ndim = len(self.roi_min)
            for dim in range(ndim):
                if self.roi_min[dim] > self.roi_max[dim]:
                    raise ValueError(
                        f"Roi min {self.roi_min} is greater than "
                        f"max {self.roi_max} in dimension {dim}"
                    )

            if self.axis_names is not None and len(self.axis_names) != ndim:
                raise ValueError(
                    f"Length of axis names ({len(self.axis_names)}) does not match number of"
                    f" dimensions in roi ({ndim})"
                )
            if self.axis_units is not None and len(self.axis_units) != ndim:
                raise ValueError(
                    f"Length of axis units ({len(self.axis_units)}) does not match number of"
                    f" dimensions in roi ({ndim})"
                )
        # If no position, check that other spatial metadata is not provided
        else:
            if any([self.roi_min, self.roi_max, self.axis_names, self.axis_units]):
                raise ValueError(
                    "Spatial metadata (roi_min, roi_max, axis_names or axis_units) provided without"
                    " position_attr"
                )

    def write(self, group: zarr.Group | Path):
        """Helper function to write GeffMetadata into the zarr geff group.

        Args:
            group (zarr.Group | Path): The geff group to write the metadata to
        """
        if isinstance(group, Path):
            group = zarr.open(group)
        for key, value in self:
            group.attrs[key] = value

    @classmethod
    def read(cls, group: zarr.Group | Path) -> GeffMetadata:
        """Helper function to read GeffMetadata from a zarr geff group.

        Args:
            group (zarr.Group | Path): The zarr group containing the geff metadata

        Returns:
            GeffMetadata: The GeffMetadata object
        """
        if isinstance(group, Path):
            group = zarr.open(group)
        return cls(**group.attrs)


def write_metadata_schema(outpath: Path):
    metadata_schema = GeffMetadata.model_json_schema()
    with open(outpath, "w") as f:
        f.write(json.dumps(metadata_schema, indent=2))
