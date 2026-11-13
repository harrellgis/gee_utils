"""Tests for the GEDI L4A/L2A monthly builders and footprint counts."""

from unittest.mock import patch

import pytest

from eetools import constants
from eetools.sensors.gedi import preprocessing as gedi


def test_gedi_constants():
    assert constants.GEDI_L4A_MONTHLY_COLLECTION == "LARSE/GEDI/GEDI04_A_002_MONTHLY"
    assert constants.GEDI_L2A_MONTHLY_COLLECTION == "LARSE/GEDI/GEDI02_A_002_MONTHLY"
    assert constants.GEDI_L4A_DEFAULT_BANDS == ["agbd", "agbd_se"]


@pytest.mark.parametrize(
    "builder, collection_id",
    [
        (gedi.get_gedi_l4a_monthly, "LARSE/GEDI/GEDI04_A_002_MONTHLY"),
        (gedi.get_gedi_l2a_monthly, "LARSE/GEDI/GEDI02_A_002_MONTHLY"),
    ],
)
def test_gedi_builders_validate_and_filter(builder, collection_id):
    with (
        patch.object(gedi, "ee") as mock_ee,
        patch.object(gedi, "validate_collection_date_range") as validate,
    ):
        builder("AOI", "2020-01-01", "2021-01-01")
    assert validate.call_args.kwargs["collection_id"] == collection_id
    mock_ee.ImageCollection.assert_called_once_with(collection_id)


def test_gedi_builders_reject_empty_bands():
    with pytest.raises(ValueError, match="at least one band"):
        gedi.get_gedi_l4a_monthly("AOI", "2020-01-01", "2021-01-01", bands=[])


@pytest.mark.parametrize(
    "args, kwargs, match",
    [((2021, 2020), {"scale": 25}, "end_year"), ((2020, 2021), {"scale": 0}, "scale")],
)
def test_gedi_footprint_counts_rejects_bad_args(args, kwargs, match):
    with pytest.raises(ValueError, match=match):
        gedi.gedi_footprint_counts("COL", "REGIONS", *args, **kwargs)


def test_gedi_footprint_counts_unweighted_sum_on_native_grid():
    with patch.object(gedi, "ee") as mock_ee:
        gedi.gedi_footprint_counts("COL", "REGIONS", 2020, 2021, scale=25)
    col = mock_ee.ImageCollection.return_value
    assert [c.args for c in col.filterDate.call_args_list] == [
        ("2020-01-01", "2021-01-01"),
        ("2021-01-01", "2022-01-01"),
    ]
    counts = col.filterDate.return_value.map.return_value.count.return_value
    counts.unmask.assert_called_with(0, False)
    pinned = counts.unmask.return_value.setDefaultProjection
    pinned.assert_called_with(mock_ee.Image.return_value.projection.return_value)
    kwargs = pinned.return_value.reduceRegions.call_args.kwargs
    assert kwargs["scale"] == 25
    assert kwargs["reducer"] is mock_ee.Reducer.sum.return_value.unweighted.return_value


@pytest.mark.ee
@pytest.mark.parametrize(
    "quality, degrade, kept", [(1, 0, True), (0, 0, False), (1, 1, False)]
)
def test_gedi_quality_mask(ee_session, first_value, quality, degrade, kept):
    import ee

    image = ee.Image.constant([120.5, 10.0, quality, degrade]).rename(
        ["agbd", "agbd_se", "l4_quality_flag", "degrade_flag"]
    )
    out = gedi.mask_gedi_quality(image, "l4_quality_flag", ["agbd"])
    assert out.bandNames().getInfo() == ["agbd"]
    value = first_value(out, "agbd")
    assert (value == pytest.approx(120.5)) if kept else value is None


@pytest.mark.ee
@pytest.mark.slow
def test_gedi_l4a_and_counts_real(ee_session):
    import ee

    box = ee.Geometry.Rectangle([36.55, -0.45, 36.65, -0.35])  # Aberdare forest
    regions = ee.FeatureCollection([ee.Feature(box, {"site_id": "s1"})])
    col = gedi.get_gedi_l4a_monthly(box, "2020-01-01", "2021-01-01")
    rows = gedi.gedi_footprint_counts(col, regions, 2020, 2020, scale=25).getInfo()
    (props,) = [f["properties"] for f in rows["features"]]
    assert props["region_id"] == "s1" and props["year"] == 2020
    assert props["n_footprints"] > 0 and float(props["n_footprints"]).is_integer()
    stats = (
        col.select("agbd")
        .mean()
        .reduceRegion(ee.Reducer.minMax(), box, 25, maxPixels=1e8)
        .getInfo()
    )
    assert stats["agbd_min"] >= 0
