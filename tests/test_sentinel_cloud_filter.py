"""Pure tests for the Sentinel-2 scene-level cloud filter option (no EE session)."""

from unittest.mock import patch

import pytest

from eetools.constants import CLOUD_FILTER
from eetools.sensors.sentinel import masking, preprocessing


@pytest.mark.parametrize("cloud_filter, expected", [(CLOUD_FILTER, 50), (20, 20)])
def test_get_s2_sr_cld_col_applies_scene_filter(cloud_filter, expected):
    with patch.object(masking, "ee") as mock_ee:
        masking.get_s2_sr_cld_col("AOI", "S", "E", cloud_filter=cloud_filter)
    mock_ee.Filter.lte.assert_called_once_with("CLOUDY_PIXEL_PERCENTAGE", expected)
    dated = mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate
    dated.return_value.filter.assert_any_call(mock_ee.Filter.lte.return_value)


def test_get_s2_sr_cld_col_none_keeps_every_scene():
    with patch.object(masking, "ee") as mock_ee:
        masking.get_s2_sr_cld_col("AOI", "S", "E", cloud_filter=None)
    mock_ee.Filter.lte.assert_not_called()
    dated = mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate
    dated.return_value.filter.assert_not_called()
    dated.return_value.map.assert_called_once_with(masking.mask_edges)


def test_default_scene_filter_unchanged():
    with patch.object(masking, "ee") as mock_ee:
        masking.get_s2_sr_cld_col("AOI", "S", "E")
    mock_ee.Filter.lte.assert_called_once_with("CLOUDY_PIXEL_PERCENTAGE", 50)


@pytest.mark.parametrize("cloud_filter", [None, 30])
def test_get_s2_sr_collection_passes_cloud_filter(cloud_filter):
    with (
        patch.object(preprocessing, "validate_s2_sr_date_range"),
        patch.object(preprocessing, "build_cloudfree_s2sr_col") as build,
    ):
        preprocessing.get_s2_sr_collection(
            "AOI", "S", "E", apply_water_masking=False, cloud_filter=cloud_filter
        )
    build.assert_called_once_with("AOI", "S", "E", cloud_filter=cloud_filter)


def test_build_cloudfree_s2sr_col_passes_cloud_filter():
    with patch.object(masking, "get_s2_sr_cld_col") as get_col:
        masking.build_cloudfree_s2sr_col("AOI", "S", "E", cloud_filter=None)
    get_col.assert_called_once_with("AOI", "S", "E", cloud_filter=None)
