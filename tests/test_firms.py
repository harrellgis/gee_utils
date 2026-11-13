"""Tests for the FIRMS active-fire builder and per-region annual counts."""

from unittest.mock import patch

import pytest

from eetools import constants
from eetools.sensors.firms import preprocessing as firms


def test_firms_constants():
    assert constants.FIRMS_COLLECTION == "FIRMS"
    assert constants.FIRMS_BANDS == ["T21", "confidence"]
    assert constants.FIRMS_DEFAULT_MIN_CONFIDENCE == 30
    assert constants.FIRMS_NATIVE_SCALE_M == 1000


@pytest.mark.parametrize("bad", [-1, 101])
def test_get_firms_rejects_bad_confidence(bad):
    with pytest.raises(ValueError, match="min_confidence"):
        firms.get_firms("AOI", "2020-01-01", "2021-01-01", min_confidence=bad)


def test_get_firms_builds_filtered_collection():
    with (
        patch.object(firms, "ee") as mock_ee,
        patch.object(firms, "validate_collection_date_range") as validate,
    ):
        firms.get_firms("AOI", "2020-01-01", "2021-01-01", min_confidence=80)
    assert validate.call_args.kwargs["collection_id"] == "FIRMS"
    mock_ee.ImageCollection.assert_called_once_with("FIRMS")
    col = mock_ee.ImageCollection.return_value.filterDate.return_value
    col.filterBounds.assert_called_once_with("AOI")
    col.filterBounds.return_value.select.assert_called_once_with(["T21", "confidence"])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"start_year": 2021, "end_year": 2020, "scale": 1000}, "end_year"),
        ({"start_year": 2020, "end_year": 2021, "scale": 0}, "scale must be positive"),
    ],
)
def test_firms_counts_rejects_bad_args(kwargs, match):
    with patch.object(firms, "get_firms") as get_firms:
        with pytest.raises(ValueError, match=match):
            firms.firms_counts("REGIONS", **kwargs)
    get_firms.assert_not_called()


def test_firms_counts_one_unweighted_sum_per_year():
    with (
        patch.object(firms, "ee") as mock_ee,
        patch.object(firms, "get_firms") as get_firms,
    ):
        firms.firms_counts("REGIONS", 2019, 2021, scale=1000, min_confidence=50)
    regions = mock_ee.FeatureCollection.return_value
    get_firms.assert_called_once_with(
        regions.geometry.return_value.bounds.return_value,
        "2019-01-01",
        "2022-01-01",
        50,
    )
    fires = get_firms.return_value.select.return_value
    assert [c.args for c in fires.filterDate.call_args_list] == [
        ("2019-01-01", "2020-01-01"),
        ("2020-01-01", "2021-01-01"),
        ("2021-01-01", "2022-01-01"),
    ]
    total = fires.filterDate.return_value.sum.return_value
    total.unmask.assert_called_with(0, False)
    reduce_kwargs = (
        total.unmask.return_value.rename.return_value.reduceRegions.call_args.kwargs
    )
    assert reduce_kwargs["scale"] == 1000
    assert (
        reduce_kwargs["reducer"]
        is mock_ee.Reducer.sum.return_value.unweighted.return_value
    )


@pytest.mark.ee
@pytest.mark.parametrize("confidence, expected", [(29, 0), (30, 1), (95, 1)])
def test_add_fire_band_threshold(ee_session, first_value, confidence, expected):
    import ee

    image = ee.Image.constant([330, confidence]).rename(["T21", "confidence"])
    assert first_value(firms.add_fire_band(image), "fire") == expected


@pytest.mark.ee
@pytest.mark.slow
def test_firms_counts_real(ee_session):
    """Real FIRMS over a fire-prone savanna window: integer counts, one row per year."""
    import ee

    region = ee.Geometry.Rectangle([34.5, -1.6, 34.7, -1.4])  # Mara savanna, Kenya
    regions = ee.FeatureCollection([ee.Feature(region, {"site_id": "s1"})])
    rows = firms.firms_counts(regions, 2019, 2020, scale=1000).getInfo()["features"]
    props = sorted((f["properties"] for f in rows), key=lambda p: p["year"])
    assert [p["year"] for p in props] == [2019, 2020]
    assert all(p["region_id"] == "s1" for p in props)
    assert all(
        p["fire_count"] >= 0 and float(p["fire_count"]).is_integer() for p in props
    )
    assert sum(p["fire_count"] for p in props) > 0
