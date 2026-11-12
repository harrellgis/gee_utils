"""Tests for eetools.compositing.

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
    from eetools.compositing import _validate_composite_stat

    assert _validate_composite_stat(stat) is None


def test_validate_composite_stat_rejects_invalid():
    from eetools.compositing import _validate_composite_stat

    with pytest.raises(ValueError, match="composite_stat"):
        _validate_composite_stat("max")


# --------------------------------------------------------------------------- #
# Earth Engine graph builders
# --------------------------------------------------------------------------- #
@pytest.mark.ee
def test_apply_stat_mean(ee_session, timed_collection, first_value):
    from eetools.compositing import _apply_stat

    out = _apply_stat(timed_collection, "mean")
    assert first_value(out, "b") == pytest.approx(0.4)


@pytest.mark.ee
def test_apply_stat_rejects_unknown(ee_session, timed_collection):
    from eetools.compositing import _apply_stat

    with pytest.raises(ValueError):
        _apply_stat(timed_collection, "bogus")


@pytest.mark.ee
def test_time_windows_annual_count(ee_session):
    from eetools.compositing import _time_windows

    windows = _time_windows(
        ee_session.Date("2020-01-01"), ee_session.Date("2022-01-01"), "annual"
    )
    assert windows.size().getInfo() == 2


@pytest.mark.ee
def test_time_windows_monthly_count(ee_session):
    from eetools.compositing import _time_windows

    # Jan, Feb, Mar -> 3 monthly windows (Apr 1 is the exclusive end). This
    # range spans 31-day months that previously truncated to 2 windows.
    windows = _time_windows(
        ee_session.Date("2020-01-01"), ee_session.Date("2020-04-01"), "monthly"
    )
    assert windows.size().getInfo() == 3


@pytest.mark.ee
def test_time_windows_monthly_includes_partial_end_month(ee_session):
    from eetools.compositing import _time_windows

    # A mid-month exclusive end still includes that month's window: the data
    # before the end date falls inside the April window.
    windows = _time_windows(
        ee_session.Date("2020-01-01"), ee_session.Date("2020-04-15"), "monthly"
    )
    months = windows.aggregate_array("month").getInfo()
    assert months == [1, 2, 3, 4]


@pytest.mark.ee
def test_time_windows_monthly_single_31_day_month(ee_session):
    from eetools.compositing import _time_windows

    # The shortest case: one 31-day month must yield exactly one window.
    windows = _time_windows(
        ee_session.Date("2020-01-01"), ee_session.Date("2020-02-01"), "monthly"
    )
    assert windows.size().getInfo() == 1


@pytest.mark.ee
def test_time_windows_rejects_bad_scale(ee_session):
    from eetools.compositing import _time_windows

    with pytest.raises(ValueError, match="annual"):
        _time_windows(
            ee_session.Date("2020-01-01"), ee_session.Date("2021-01-01"), "weekly"
        )


@pytest.mark.ee
def test_build_period_composites_annual(ee_session, timed_collection):
    from eetools.compositing import build_period_composites

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


# --------------------------------------------------------------------------- #
# build_seasonal_composites
# --------------------------------------------------------------------------- #
@pytest.fixture
def seasonal_collection(ee_session):
    """Collection with one image in the wet season (Apr) of 2020 and 2021, and
    one image outside the season (Jan) in 2022 — used to verify per-year
    filtering and empty-year exclusion."""
    return ee_session.ImageCollection(
        [
            ee_session.Image.constant(0.3)
            .toFloat()
            .rename("b")
            .set("system:time_start", ee_session.Date("2020-04-15").millis()),
            ee_session.Image.constant(0.7)
            .toFloat()
            .rename("b")
            .set("system:time_start", ee_session.Date("2021-04-15").millis()),
            # 2022 image falls outside the (3, 5) season window
            ee_session.Image.constant(0.9)
            .toFloat()
            .rename("b")
            .set("system:time_start", ee_session.Date("2022-01-15").millis()),
        ]
    )


@pytest.mark.ee
def test_build_seasonal_composites_happy_path(ee_session, seasonal_collection):
    from eetools.compositing import build_seasonal_composites

    composites = build_seasonal_composites(
        seasonal_collection,
        bands=["b"],
        start_year=2020,
        end_year=2021,
        season_months=(3, 5),
        season_name="wet",
        composite_stat="median",
    )
    assert composites.size().getInfo() == 2
    first = composites.sort("system:time_start").first()
    assert first.get("year").getInfo() == 2020
    assert first.get("season").getInfo() == "wet"
    assert first.get("season_months").getInfo() == "3-5"
    assert first.get("composite_stat").getInfo() == "median"


@pytest.mark.ee
def test_build_seasonal_composites_excludes_empty_years(
    ee_session, seasonal_collection
):
    from eetools.compositing import build_seasonal_composites

    # 2022 has no images in months 3-5, so the output should have only 2020 and 2021.
    composites = build_seasonal_composites(
        seasonal_collection,
        bands=["b"],
        start_year=2020,
        end_year=2022,
        season_months=(3, 5),
        season_name="wet",
    )
    assert composites.size().getInfo() == 2


def test_build_seasonal_composites_rejects_invalid_stat():
    from eetools.compositing import build_seasonal_composites

    with pytest.raises(ValueError, match="composite_stat"):
        build_seasonal_composites(
            None,  # type: ignore[arg-type]
            bands=["b"],
            start_year=2020,
            end_year=2021,
            season_months=(3, 5),
            season_name="wet",
            composite_stat="max",
        )


def test_build_seasonal_composites_rejects_invalid_months():
    from eetools.compositing import build_seasonal_composites

    with pytest.raises(ValueError, match="season_months"):
        build_seasonal_composites(
            None,  # type: ignore[arg-type]
            bands=["b"],
            start_year=2020,
            end_year=2021,
            season_months=(5, 3),  # start > end
            season_name="wet",
        )


# --------------------------------------------------------------------------- #
# build_composite
# --------------------------------------------------------------------------- #
def test_build_composite_rejects_invalid_stat():
    from eetools.compositing import build_composite

    with pytest.raises(ValueError, match="composite_stat"):
        build_composite(None, bands=["b"], composite_stat="max")  # type: ignore[arg-type]


@pytest.mark.ee
def test_build_composite_median_value(ee_session, timed_collection):
    from eetools.compositing import build_composite

    # timed_collection has two images: constant 0.2 and constant 0.6.
    # Median of two values = mean = 0.4.
    composite = build_composite(timed_collection, bands=["b"], composite_stat="median")
    val = (
        composite.reduceRegion(
            reducer=ee_session.Reducer.first(),
            geometry=ee_session.Geometry.Point([0, 0]),
            scale=1000,
        )
        .get("b")
        .getInfo()
    )
    assert val == pytest.approx(0.4)


@pytest.mark.ee
def test_build_composite_selects_bands(ee_session):
    from eetools.compositing import build_composite

    col = ee_session.ImageCollection(
        [ee_session.Image.constant([1, 2]).toFloat().rename(["a", "b"])]
    )
    composite = build_composite(col, bands=["a"])
    assert composite.bandNames().getInfo() == ["a"]


# --------------------------------------------------------------------------- #
# clear_count band
# --------------------------------------------------------------------------- #
def test_composite_without_clear_count_is_unchanged():
    from unittest.mock import MagicMock

    from eetools.compositing import _composite

    col = MagicMock(name="collection")
    out = _composite(col, ["NDVI", "EVI"], "median", add_clear_count=False)
    assert out is col.median.return_value
    col.select.assert_not_called()
    col.median.return_value.addBands.assert_not_called()


def test_composite_with_clear_count_counts_first_band_as_uint16():
    from unittest.mock import MagicMock

    from eetools.compositing import _composite
    from eetools.constants import CLEAR_COUNT_BAND

    col = MagicMock(name="collection")
    out = _composite(col, ["NDVI", "EVI"], "mean", add_clear_count=True)

    col.select.assert_called_once_with("NDVI")
    count = col.select.return_value.count
    count.assert_called_once_with()
    count.return_value.toUint16.assert_called_once_with()
    rename = count.return_value.toUint16.return_value.rename
    rename.assert_called_once_with(CLEAR_COUNT_BAND)
    col.mean.return_value.addBands.assert_called_once_with(rename.return_value)
    assert out is col.mean.return_value.addBands.return_value
    assert CLEAR_COUNT_BAND == "clear_count"


@pytest.fixture
def masked_triplet(ee_session):
    """Three 2021 images of band 'b'; the first is masked west of lon 0, so a
    western pixel has 2 clear observations and an eastern pixel has 3."""
    ee = ee_session
    west_masked = ee.Image.pixelLonLat().select("longitude").gte(0)
    images = []
    for i, (value, date) in enumerate(
        [(0.2, "2021-04-01"), (0.4, "2021-04-10"), (0.6, "2021-04-20")]
    ):
        img = ee.Image.constant(value).toFloat().rename("b")
        if i == 0:
            img = img.updateMask(west_masked)
        images.append(img.set("system:time_start", ee.Date(date).millis()))
    return ee.ImageCollection(images)


def _value_at(ee, image, band, lon):
    return (
        image.select(band)
        .reduceRegion(ee.Reducer.first(), ee.Geometry.Point([lon, 0]), scale=1000)
        .get(band)
        .getInfo()
    )


@pytest.mark.ee
def test_build_composite_clear_count_values_and_type(ee_session, masked_triplet):
    from eetools.compositing import build_composite

    composite = build_composite(masked_triplet, ["b"], add_clear_count=True)
    assert composite.bandNames().getInfo() == ["b", "clear_count"]
    assert _value_at(ee_session, composite, "clear_count", -0.5) == 2
    assert _value_at(ee_session, composite, "clear_count", 0.5) == 3
    band_type = composite.select("clear_count").bandTypes().getInfo()["clear_count"]
    assert band_type["precision"] == "int"
    assert (band_type["min"], band_type["max"]) == (0, 65535)


@pytest.mark.ee
def test_period_and_seasonal_composites_clear_count(ee_session, masked_triplet):
    from eetools.compositing import (
        build_period_composites,
        build_seasonal_composites,
    )

    period = build_period_composites(
        masked_triplet,
        ["b"],
        "2020-01-01",
        "2022-01-01",
        temporal_scale="annual",
        add_clear_count=True,
    )
    assert period.size().getInfo() == 1  # empty 2020 still dropped
    first = ee_session.Image(period.first())
    assert first.bandNames().getInfo() == ["b", "clear_count"]
    assert _value_at(ee_session, first, "clear_count", -0.5) == 2

    seasonal = build_seasonal_composites(
        masked_triplet,
        ["b"],
        start_year=2021,
        end_year=2021,
        season_months=(3, 5),
        season_name="wet",
        add_clear_count=True,
    )
    s_first = ee_session.Image(seasonal.first())
    assert _value_at(ee_session, s_first, "clear_count", 0.5) == 3

    default = build_seasonal_composites(
        masked_triplet,
        ["b"],
        start_year=2021,
        end_year=2021,
        season_months=(3, 5),
        season_name="wet",
    )
    assert ee_session.Image(default.first()).bandNames().getInfo() == ["b"]
