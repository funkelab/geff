import re

import pydantic
import pytest
import zarr

from geff.metadata_schema import GeffMetadata, _get_versions_regex, write_metadata_schema


class TestVersionRegex:
    def test_get_versions_regex_simple(self):
        version_str = "0.0.1-a"
        versions = ["0.0"]
        regex = _get_versions_regex(versions)
        assert re.match(regex, version_str) is not None

    def test_get_versions_regex_complex(self):
        version_str = "0.1.1-a"
        versions = ["0.0", "0.1"]
        regex = _get_versions_regex(versions)
        assert re.match(regex, version_str) is not None

    def test_invalid_version_regex(self):
        version_str = "v1.0.1-a"
        versions = ["0.0", "0.1"]
        regex = _get_versions_regex(versions)
        assert re.match(regex, version_str) is None

    def test_invalid_prefix_regex(self):
        version_str = "9810.0.1"
        versions = ["0.0", "0.1"]
        regex = _get_versions_regex(versions)
        assert re.match(regex, version_str) is None


class TestMetadataModel:
    def test_valid_init(self):
        model = GeffMetadata(
            geff_version="0.0.1",
            directed=True,
            position_attr="position",
            roi_min=[0, 0, 0],
            roi_max=[100, 100, 100],
            axis_names=["t", "y", "x"],
            axis_units=["min", "nm", "nm"],
        )
        assert model.geff_version == "0.0.1"

        model = GeffMetadata(
            geff_version="0.0.1",
            directed=True,
            position_attr="position",
            roi_min=[0, 0, 0],
            roi_max=[100, 100, 100],
        )
        assert model.axis_names is None
        assert model.axis_units is None

        model = GeffMetadata(
            geff_version="0.0.1",
            directed=True
        )
        assert model.position_attr is None
        assert model.roi_min is None
        assert model.roi_max is None

    def test_invalid_version(self):
        with pytest.raises(pydantic.ValidationError, match="String should match pattern"):
            GeffMetadata(
                geff_version="aljkdf",
                directed=True,
                roi_min=[0, 0, 0],
                roi_max=[100, 100, 100],
                axis_names=["t", "y", "x"],
                axis_units=["min", "nm", "nm"],
            )

    def test_invalid_roi(self):
        with pytest.raises(ValueError, match="Roi min .* is greater than max .* in dimension 0"):
            GeffMetadata(
                geff_version="0.0.1-a",
                directed=False,
                position_attr="position",
                roi_min=[1000, 0, 0],
                roi_max=[100, 100, 100],
            )
        with pytest.raises(ValueError, match="Roi min .* and roi max .* have different lengths"):
            GeffMetadata(
                geff_version="0.0.1-a",
                directed=False,
                position_attr="position",
                roi_min=[1000, 0],
                roi_max=[100, 100, 100],
            )

    def test_invalid_axis_annotations(self):
        with pytest.raises(
            ValueError,
            match="Length of axis names",
        ):
            GeffMetadata(
                geff_version="0.0.1-a",
                directed=False,
                position_attr="position",
                roi_min=[0, 0, 0],
                roi_max=[100, 100, 100],
                axis_names=["t", "y"],
                axis_units=["min", "nm", "nm"],
            )

        with pytest.raises(
            ValueError,
            match="Length of axis units",
        ):
            GeffMetadata(
                geff_version="0.0.1-a",
                directed=False,
                position_attr="position",
                roi_min=[0, 0, 0],
                roi_max=[100, 100, 100],
                axis_names=["t", "y", "x"],
                axis_units=["nm", "nm"],
            )

    def test_invalid_spatial_metadata(self):
        with pytest.raises(
            ValueError,
            match="Spatial metadata"
        ):
            GeffMetadata(
                geff_version="0.0.1-a",
                directed=False,
                axis_units=["nm", "nm"],
            )

    def test_extra_attrs(self):
        # Should not fail
        GeffMetadata(
            geff_version="0.0.1",
            directed=True,
            position_attr="position",
            roi_min=[0, 0, 0],
            roi_max=[100, 100, 100],
            axis_names=["t", "y", "x"],
            axis_units=["min", "nm", "nm"],
            extra=True,
        )

    def test_read_write(self, tmp_path):
        meta = GeffMetadata(
            geff_version="0.0.1",
            directed=True,
            position_attr="position",
            roi_min=[0, 0, 0],
            roi_max=[100, 100, 100],
            axis_names=["t", "y", "x"],
            axis_units=["min", "nm", "nm"],
            extra=True,
        )
        zpath = tmp_path / "test.zarr"
        group = zarr.open(zpath, "a")
        meta.write(group)
        compare = GeffMetadata.read(group)
        assert compare == meta

        meta.directed = False
        meta.write(zpath)
        compare = GeffMetadata.read(zpath)
        assert compare == meta


def test_write_schema(tmp_path):
    schema_path = tmp_path / "schema.json"
    write_metadata_schema(schema_path)
    assert schema_path.is_file()
    assert schema_path.stat().st_size > 0
