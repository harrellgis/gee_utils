"""Tests for the WDPA protected-areas module.

The export wrappers are tested with mocks (no real export tasks fire, and the export
guard raises before any Earth Engine call). The real-asset loaders/filters are checked
against the live WDPA collection and marked slow.
"""

from unittest.mock import patch

import pytest

from eetools.sensors.wdpa import preprocessing


# --------------------------------------------------------------------------- #
# Pure / mock tests (no Earth Engine session)
# --------------------------------------------------------------------------- #
def test_export_wdpa_to_drive_requires_country_or_identifier():
    with pytest.raises(ValueError, match="at least one"):
        preprocessing.export_wdpa_to_drive(folder="folder", file_prefix="prefix")


def test_export_wdpa_to_drive_wires_filter_and_export():
    with (
        patch.object(preprocessing, "get_wdpa_collection", return_value="FC") as get_fc,
        patch.object(preprocessing, "export_table_to_drive") as export_tbl,
    ):
        out = preprocessing.export_wdpa_to_drive(
            folder="folder",
            file_prefix="bwa_pas",
            country="BWA",
            file_format="GeoJSON",
        )

    get_fc.assert_called_once_with(country="BWA", identifier=None)
    _, kwargs = export_tbl.call_args
    assert kwargs["collection"] == "FC"
    assert kwargs["folder"] == "folder"
    assert kwargs["fileNamePrefix"] == "bwa_pas"
    assert kwargs["fileFormat"] == "GeoJSON"
    assert out == "FC"


def test_export_wdpa_in_aoi_to_drive_wires_loader_and_export():
    with (
        patch.object(preprocessing, "get_wdpa_in_aoi", return_value="FC") as get_fc,
        patch.object(preprocessing, "export_table_to_drive") as export_tbl,
    ):
        out = preprocessing.export_wdpa_in_aoi_to_drive(
            aoi="AOI", folder="folder", file_prefix="aoi_pas"
        )

    get_fc.assert_called_once_with("AOI")
    _, kwargs = export_tbl.call_args
    assert kwargs["collection"] == "FC"
    assert kwargs["fileNamePrefix"] == "aoi_pas"
    assert kwargs["fileFormat"] == "SHP"  # default vector format
    assert out == "FC"


# --------------------------------------------------------------------------- #
# Real-asset integration (live WDPA collection)
# --------------------------------------------------------------------------- #
@pytest.mark.ee
@pytest.mark.slow
def test_get_wdpa_collection_filters_by_country(ee_session):
    col = preprocessing.get_wdpa_collection(country="BWA")
    assert isinstance(col, ee_session.FeatureCollection)
    assert col.size().getInfo() > 0
    # Every returned feature should be in the requested country.
    iso3_values = col.aggregate_array("ISO3").distinct().getInfo()
    assert iso3_values == ["BWA"]


@pytest.mark.ee
@pytest.mark.slow
def test_get_wdpa_collection_filters_by_identifier(ee_session):
    # Pull a real site id from Botswana, then confirm filtering returns that site.
    bwa = preprocessing.get_wdpa_collection(country="BWA")
    site_id = bwa.first().get("SITE_ID").getInfo()

    one = preprocessing.get_wdpa_collection(identifier=site_id)
    assert one.size().getInfo() >= 1
    assert one.aggregate_array("SITE_ID").distinct().getInfo() == [site_id]


@pytest.mark.ee
@pytest.mark.slow
def test_get_wdpa_in_aoi_returns_feature_collection(ee_session, small_aoi):
    col = preprocessing.get_wdpa_in_aoi(small_aoi)
    assert isinstance(col, ee_session.FeatureCollection)
    # Size is queryable (may be 0 if the small AOI has no PA); just confirm it resolves.
    assert col.size().getInfo() >= 0
