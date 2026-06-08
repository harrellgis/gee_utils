"""Tests for eetools.visualization.summaries.

``_validate_composite_stat`` is a pure guard and runs without Earth Engine;
the remaining functions build server-side graphs and are exercised on small
synthetic constant collections under the ``ee`` marker.
"""

import pytest


# --------------------------------------------------------------------------- #
# Pure validation (no Earth Engine)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stat", ["mean", "median", "sum"])
def test_validate_composite_stat_accepts_valid(stat):
    from eetools.visualization.summaries import _validate_composite_stat

    # Returns None and does not raise for supported stats.
    assert _validate_composite_stat(stat) is None


def test_validate_composite_stat_rejects_invalid():
    from eetools.visualization.summaries import _validate_composite_stat

    with pytest.raises(ValueError, match="composite_stat"):
        _validate_composite_stat("max")


# --------------------------------------------------------------------------- #
# Earth Engine graph builders
# --------------------------------------------------------------------------- #
@pytest.fixture
def timed_collection(ee_session):
    """A 2-image constant collection with one image in 2020 and one in 2021.

    The constants are cast to a common float type — otherwise each
    ``ee.Image.constant`` carries a distinct typed value-range and EE rejects
    reductions across the collection as "inhomogeneous".
    """
    return ee_session.ImageCollection(
        [
            ee_session.Image.constant(0.2)
            .toFloat()
            .rename("b")
            .set("system:time_start", ee_session.Date("2020-06-01").millis()),
            ee_session.Image.constant(0.6)
            .toFloat()
            .rename("b")
            .set("system:time_start", ee_session.Date("2021-06-01").millis()),
        ]
    )


@pytest.mark.ee
def test_apply_stat_mean(ee_session, timed_collection, first_value):
    from eetools.visualization.summaries import _apply_stat

    out = _apply_stat(timed_collection, "mean")
    assert first_value(out, "b") == pytest.approx(0.4)


@pytest.mark.ee
def test_apply_stat_rejects_unknown(ee_session, timed_collection):
    from eetools.visualization.summaries import _apply_stat

    with pytest.raises(ValueError):
        _apply_stat(timed_collection, "bogus")


@pytest.mark.ee
def test_time_windows_annual_count(ee_session):
    from eetools.visualization.summaries import _time_windows

    windows = _time_windows(
        ee_session.Date("2020-01-01"), ee_session.Date("2022-01-01"), "annual"
    )
    assert windows.size().getInfo() == 2


@pytest.mark.ee
def test_time_windows_monthly_count(ee_session):
    from eetools.visualization.summaries import _time_windows

    # Jan, Feb, Mar -> 3 monthly windows (Apr 1 is the exclusive end). This
    # range spans 31-day months that previously truncated to 2 windows.
    windows = _time_windows(
        ee_session.Date("2020-01-01"), ee_session.Date("2020-04-01"), "monthly"
    )
    assert windows.size().getInfo() == 3


@pytest.mark.ee
def test_time_windows_monthly_includes_partial_end_month(ee_session):
    from eetools.visualization.summaries import _time_windows

    # A mid-month exclusive end still includes that month's window: the data
    # before the end date falls inside the April window.
    windows = _time_windows(
        ee_session.Date("2020-01-01"), ee_session.Date("2020-04-15"), "monthly"
    )
    months = windows.aggregate_array("month").getInfo()
    assert months == [1, 2, 3, 4]


@pytest.mark.ee
def test_time_windows_monthly_single_31_day_month(ee_session):
    from eetools.visualization.summaries import _time_windows

    # The shortest case: one 31-day month must yield exactly one window.
    windows = _time_windows(
        ee_session.Date("2020-01-01"), ee_session.Date("2020-02-01"), "monthly"
    )
    assert windows.size().getInfo() == 1


@pytest.mark.ee
def test_time_windows_rejects_bad_scale(ee_session):
    from eetools.visualization.summaries import _time_windows

    with pytest.raises(ValueError, match="annual"):
        _time_windows(
            ee_session.Date("2020-01-01"), ee_session.Date("2021-01-01"), "weekly"
        )


@pytest.mark.ee
def test_build_period_composites_annual(ee_session, timed_collection):
    from eetools.visualization.summaries import build_period_composites

    composites = build_period_composites(
        timed_collection,
        bands=["b"],
        start_date="2020-01-01",
        end_date="2022-01-01",
        temporal_scale="annual",
        composite_stat="median",
    )
    assert composites.size().getInfo() == 2
    first = composites.sort("system:time_start").first()
    assert first.get("year").getInfo() == 2020
    assert first.get("temporal_scale").getInfo() == "annual"
    assert first.get("composite_stat").getInfo() == "median"


@pytest.mark.ee
def test_reduce_image_over_region_combines_props_and_stats(ee_session, small_aoi):
    from eetools.visualization.summaries import reduce_image_over_region

    img = (
        ee_session.Image.constant(0.5)
        .rename("b")
        .clip(small_aoi)
        .set("year", 2021)
        .set("date", "2021-06-01")
        .set("temporal_scale", "annual")
    )
    feat = reduce_image_over_region(img, region=small_aoi, bands=["b"], scale=100)
    props = feat.toDictionary().getInfo()
    assert props["b"] == pytest.approx(0.5)
    assert props["year"] == 2021
    assert props["temporal_scale"] == "annual"


@pytest.mark.ee
def test_collection_to_region_timeseries_one_feature_per_image(
    ee_session, timed_collection, small_aoi
):
    from eetools.visualization.summaries import collection_to_region_timeseries

    # Give each image the metadata reduce_image_over_region reads.
    annotated = timed_collection.map(
        lambda img: ee_session.Image(img).set("year", 2020)
    )
    fc = collection_to_region_timeseries(
        annotated, region=small_aoi, bands=["b"], scale=100
    )
    assert isinstance(fc, ee_session.FeatureCollection)
    assert fc.size().getInfo() == 2


@pytest.mark.ee
def test_image_collection_to_region_stats_fc(ee_session, small_aoi):
    from eetools.visualization.summaries import image_collection_to_region_stats_fc

    image = (
        ee_session.Image.constant(0.5)
        .rename("b")
        .set("year", 2021)
        .set("product", "test")
        .set("param_set", "default")
    )
    collection = ee_session.ImageCollection([image])
    regions = ee_session.FeatureCollection(
        [ee_session.Feature(small_aoi, {"site_name": "A"})]
    )
    fc = image_collection_to_region_stats_fc(
        collection, regions_fc=regions, band_names=["b"], scale=100
    )
    assert isinstance(fc, ee_session.FeatureCollection)
    assert fc.size().getInfo() == 1
    props = fc.first().toDictionary().getInfo()
    assert props["site_name"] == "A"
    assert props["year"] == 2021


@pytest.mark.ee
def test_summarize_collection_histograms(ee_session, small_aoi):
    from eetools.visualization.summaries import summarize_collection_histograms

    image = (
        ee_session.Image.constant(0.5)
        .rename("b")
        .clip(small_aoi)
        .set("site_name", "A")
    )
    collection = ee_session.ImageCollection([image])
    result = summarize_collection_histograms(
        collection,
        band_name="b",
        min_value=0.0,
        max_value=1.0,
        steps=4,
        scale=100,
    )
    assert isinstance(result, list)
    assert result[0]["site_name"] == "A"
    assert result[0]["histogram"] is not None
