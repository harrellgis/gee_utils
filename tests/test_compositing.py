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
    out = _composite(col, ["NDVI", "EVI"], "median", [])
    assert out is col.median.return_value
    col.select.assert_not_called()
    col.median.return_value.addBands.assert_not_called()


def test_composite_with_clear_count_counts_first_band_as_uint16():
    from unittest.mock import MagicMock

    from eetools.compositing import _composite
    from eetools.constants import CLEAR_COUNT_BAND

    col = MagicMock(name="collection")
    out = _composite(col, ["NDVI", "EVI"], "mean", ["clear_count"])

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


# --------------------------------------------------------------------------- #
# build_period_stack
# --------------------------------------------------------------------------- #
def test_period_stack_band_names():
    from eetools.compositing import period_stack_band_names

    assert period_stack_band_names(["a", "b"], ["p1", "p2"]) == [
        "a_p1",
        "b_p1",
        "a_p2",
        "b_p2",
    ]
    assert period_stack_band_names(["a"], ["p1"], add_clear_count=True) == [
        "a_p1",
        "clear_count_p1",
    ]
    assert period_stack_band_names(["a"], ["x"], band_name="{label}|{band}") == ["x|a"]


def test_period_stack_band_names_qa_layers():
    from eetools.compositing import period_stack_band_names

    # layer order is fixed (clear_count, then doy) whatever order is passed
    assert period_stack_band_names(["a"], ["p1"], qa_layers=("doy", "clear_count")) == [
        "a_p1",
        "clear_count_p1",
        "obs_doy_median_p1",
        "obs_doy_min_p1",
        "obs_doy_max_p1",
    ]
    assert period_stack_band_names(["a"], ["p1"], add_clear_count=True) == (
        period_stack_band_names(["a"], ["p1"], qa_layers=("clear_count",))
    )
    assert period_stack_band_names(["a"], ["p1"], qa_layers=("doy",))[1:] == [
        "obs_doy_median_p1",
        "obs_doy_min_p1",
        "obs_doy_max_p1",
    ]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"bands": []}, "bands"),
        ({"periods": []}, "periods"),
        ({"periods": [("p", "2021-01-01", "2022-01-01")] * 2}, "unique"),
        ({"band_name": "{band}"}, "band_name"),
        ({"composite_stat": "max"}, "composite_stat"),
        ({"qa_layers": ("cloud_distance",)}, "qa_layers"),
        ({"qa_layers": "doy"}, "qa_layers"),
    ],
)
def test_build_period_stack_validates_before_ee(kwargs, match):
    from eetools.compositing import build_period_stack

    args = {
        "collection": None,
        "bands": ["b"],
        "periods": [("p", "2021-01-01", "2022-01-01")],
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        build_period_stack(**args)


@pytest.mark.ee
def test_build_period_stack_keeps_empty_periods(ee_session, masked_triplet):
    from eetools.compositing import build_period_stack

    periods = [
        ("apr", "2021-04-01", "2021-05-01"),
        ("none", "2022-01-01", "2022-02-01"),
    ]
    stack = build_period_stack(masked_triplet, ["b"], periods, add_clear_count=True)
    assert stack.bandNames().getInfo() == [
        "b_apr",
        "clear_count_apr",
        "b_none",
        "clear_count_none",
    ]
    assert _value_at(ee_session, stack, "b_apr", 0.5) == pytest.approx(0.4)
    assert _value_at(ee_session, stack, "b_apr", -0.5) == pytest.approx(0.5)
    assert _value_at(ee_session, stack, "clear_count_apr", 0.5) == 3
    assert _value_at(ee_session, stack, "clear_count_apr", -0.5) == 2
    assert _value_at(ee_session, stack, "b_none", 0.5) is None  # masked
    assert _value_at(ee_session, stack, "clear_count_none", 0.5) == 0
    types = stack.bandTypes().getInfo()
    assert types["clear_count_none"]["precision"] == "int"


@pytest.mark.ee
def test_build_period_stack_merges_typed_range_bands(ee_session):
    """Index bands are typed Float<-1, 1>; without the cast EE refuses the merge."""
    from eetools.compositing import build_period_stack

    ee = ee_session
    images = [
        ee.Image.constant([0.5, 0.1])
        .rename(["x", "y"])
        .normalizedDifference(["x", "y"])  # Float<-1, 1>, value 0.6667
        .rename("nd")
        .set("system:time_start", ee.Date("2021-06-01").millis())
    ]
    stack = build_period_stack(
        ee.ImageCollection(images), ["nd"], [("jun", "2021-06-01", "2021-07-01")]
    )
    assert _value_at(ee, stack, "nd_jun", 0.5) == pytest.approx(2 / 3, rel=1e-5)


@pytest.mark.ee
def test_build_period_stack_doy_layers(ee_session, masked_triplet):
    """Clear observations on 1, 10 and 20 April 2021 (days 91, 100, 110); the first is
    masked west of lon 0."""
    from eetools.compositing import build_period_stack

    periods = [
        ("apr", "2021-04-01", "2021-05-01"),
        ("none", "2022-01-01", "2022-02-01"),
    ]
    stack = build_period_stack(
        masked_triplet, ["b"], periods, qa_layers=("clear_count", "doy")
    )
    assert stack.bandNames().getInfo()[:5] == [
        "b_apr",
        "clear_count_apr",
        "obs_doy_median_apr",
        "obs_doy_min_apr",
        "obs_doy_max_apr",
    ]
    ee = ee_session
    assert _value_at(ee, stack, "b_apr", 0.5) == pytest.approx(0.4)  # unchanged
    assert _value_at(ee, stack, "obs_doy_median_apr", 0.5) == pytest.approx(100)
    assert _value_at(ee, stack, "obs_doy_min_apr", 0.5) == pytest.approx(91)
    assert _value_at(ee, stack, "obs_doy_max_apr", 0.5) == pytest.approx(110)
    assert _value_at(ee, stack, "obs_doy_median_apr", -0.5) == pytest.approx(105)
    assert _value_at(ee, stack, "obs_doy_min_apr", -0.5) == pytest.approx(100)
    assert _value_at(ee, stack, "obs_doy_median_none", 0.5) is None  # empty: masked
    assert _value_at(ee, stack, "clear_count_none", 0.5) == 0


@pytest.mark.ee
def test_build_period_stack_doy_crosses_year_end(ee_session):
    from eetools.compositing import build_period_stack

    ee = ee_session
    images = [
        ee.Image.constant(0.3)
        .toFloat()
        .rename("b")
        .set("system:time_start", ee.Date(d).millis())
        for d in ("2021-12-31", "2022-01-02")
    ]
    stack = build_period_stack(
        ee.ImageCollection(images),
        ["b"],
        [("djf", "2021-12-01", "2022-03-01")],
        qa_layers=("doy",),
    )
    assert _value_at(ee, stack, "obs_doy_min_djf", 0.5) == pytest.approx(365)
    assert _value_at(ee, stack, "obs_doy_max_djf", 0.5) == pytest.approx(367)


@pytest.mark.ee
def test_build_period_stack_default_bands_unchanged(ee_session, masked_triplet):
    from eetools.compositing import build_period_stack

    periods = [("apr", "2021-04-01", "2021-05-01")]
    stack = build_period_stack(masked_triplet, ["b"], periods, add_clear_count=True)
    assert stack.bandNames().getInfo() == ["b_apr", "clear_count_apr"]


def test_collection_lineage_validates_before_ee():
    from eetools.compositing import collection_lineage

    with pytest.raises(ValueError, match="periods"):
        collection_lineage(None, [])
    with pytest.raises(ValueError, match="properties"):
        collection_lineage(None, [("p", "2021-01-01", "2021-02-01")], properties=[])


@pytest.mark.ee
def test_collection_lineage(ee_session, masked_triplet):
    from eetools.compositing import collection_lineage

    out = collection_lineage(
        masked_triplet,
        [("apr", "2021-04-01", "2021-04-15"), ("none", "2022-01-01", "2022-02-01")],
    )
    apr = out["apr"]
    assert (apr["start"], apr["end"], apr["n_scenes"]) == (
        "2021-04-01",
        "2021-04-15",
        2,
    )
    assert [s["datetime"] for s in apr["scenes"]] == [
        "2021-04-01T00:00:00Z",
        "2021-04-10T00:00:00Z",
    ]
    assert apr["scenes"][0]["system:time_start"] == 1617235200000  # system: props kept
    assert set(apr["scenes"][0]) == {"system:index", "system:time_start", "datetime"}
    assert out["none"] == {
        "start": "2022-01-01",
        "end": "2022-02-01",
        "n_scenes": 0,
        "scenes": [],
    }
    with pytest.raises(ValueError, match="max_scenes"):
        collection_lineage(
            masked_triplet, [("apr", "2021-04-01", "2021-05-01")], max_scenes=2
        )


# --------------------------------------------------------------------------- #
# QA layers: gap, and the other builders (0.15.0)
# --------------------------------------------------------------------------- #
def test_build_composite_rejects_gap():
    from eetools.compositing import build_composite

    with pytest.raises(ValueError, match="needs a window"):
        build_composite(None, ["b"], qa_layers=("gap",))


@pytest.mark.ee
def test_build_period_stack_gap(ee_session, masked_triplet):
    """Window 1 March (day 60) to 1 May (121): east obs 91, 100, 110 -> longest
    interval 31 (60 -> 91); west obs 100, 110 -> 40 (60 -> 100). Empty: 31 days."""
    from eetools.compositing import build_period_stack

    periods = [
        ("ma", "2021-03-01", "2021-05-01"),
        ("jan22", "2022-01-01", "2022-02-01"),
    ]
    stack = build_period_stack(
        masked_triplet, ["b"], periods, qa_layers=("clear_count", "doy", "gap")
    )
    assert stack.bandNames().getInfo()[:6] == [
        "b_ma",
        "clear_count_ma",
        "obs_doy_median_ma",
        "obs_doy_min_ma",
        "obs_doy_max_ma",
        "obs_gap_max_ma",
    ]
    ee = ee_session
    assert _value_at(ee, stack, "obs_gap_max_ma", 0.5) == pytest.approx(31)
    assert _value_at(ee, stack, "obs_gap_max_ma", -0.5) == pytest.approx(40)
    assert _value_at(ee, stack, "obs_gap_max_jan22", 0.5) == pytest.approx(31)
    assert _value_at(ee, stack, "obs_doy_median_ma", 0.5) == pytest.approx(100)


@pytest.mark.ee
def test_build_period_stack_gap_only(ee_session, masked_triplet):
    from eetools.compositing import build_period_stack

    stack = build_period_stack(
        masked_triplet, ["b"], [("ma", "2021-03-01", "2021-05-01")], qa_layers=("gap",)
    )
    assert stack.bandNames().getInfo() == ["b_ma", "obs_gap_max_ma"]
    assert _value_at(ee_session, stack, "obs_gap_max_ma", -0.5) == pytest.approx(40)


@pytest.mark.ee
def test_build_composite_doy_layers(ee_session, masked_triplet):
    from eetools.compositing import build_composite

    out = build_composite(masked_triplet, ["b"], qa_layers=("clear_count", "doy"))
    assert out.bandNames().getInfo() == [
        "b",
        "clear_count",
        "obs_doy_median",
        "obs_doy_min",
        "obs_doy_max",
    ]
    assert _value_at(ee_session, out, "b", 0.5) == pytest.approx(0.4)
    assert _value_at(ee_session, out, "obs_doy_median", -0.5) == pytest.approx(105)
    assert _value_at(ee_session, out, "obs_doy_min", 0.5) == pytest.approx(91)


@pytest.mark.ee
def test_build_period_composites_qa_layers(ee_session, masked_triplet):
    """Annual 2021 window: days 1 to 366 (first day after); last obs day 110 -> 256."""
    from eetools.compositing import build_period_composites

    col = build_period_composites(
        masked_triplet,
        ["b"],
        "2021-01-01",
        "2022-01-01",
        qa_layers=("doy", "gap"),
    )
    img = ee_session.Image(col.first())
    assert img.bandNames().getInfo() == [
        "b",
        "obs_doy_median",
        "obs_doy_min",
        "obs_doy_max",
        "obs_gap_max",
    ]
    assert _value_at(ee_session, img, "obs_doy_median", 0.5) == pytest.approx(100)
    assert _value_at(ee_session, img, "obs_gap_max", 0.5) == pytest.approx(256)


@pytest.mark.ee
def test_build_seasonal_composites_qa_layers(ee_session, masked_triplet):
    """March-May 2021: days 60 to 152 (first day after); last obs 110 -> gap 42."""
    from eetools.compositing import build_seasonal_composites

    col = build_seasonal_composites(
        masked_triplet,
        ["b"],
        2021,
        2021,
        (3, 5),
        "mam",
        qa_layers=("clear_count", "gap"),
    )
    img = ee_session.Image(col.first())
    assert img.bandNames().getInfo() == ["b", "clear_count", "obs_gap_max"]
    assert _value_at(ee_session, img, "clear_count", -0.5) == 2
    assert _value_at(ee_session, img, "obs_gap_max", -0.5) == pytest.approx(42)
