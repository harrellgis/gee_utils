"""Tests for the Sentinel-1 timescan module.

The statistics are checked on a synthetic 3-image constant collection in dB, against
values computed here in plain Python (population stdDev, as ee.Reducer.stdDev). The
seasonal builder is checked for date filtering, band suffixing, and its properties.
The real-collection run is marked slow.
"""

import statistics

import pytest

# (VV, VH) backscatter in dB for each synthetic in-season image.
IN_SEASON_DB = [(-10.0, -20.0), (-12.0, -20.0), (-20.0, -14.0)]


def _linear(db: float) -> float:
    return 10 ** (db / 10)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def s1_image(ee_session):
    """Return a builder for a constant VV+VH dB image dated ``date``."""

    def _s1_image(vv_db: float, vh_db: float, date: str):
        return (
            ee_session.Image.cat(
                [
                    ee_session.Image.constant(vv_db).rename("VV"),
                    ee_session.Image.constant(vh_db).rename("VH"),
                ]
            )
            .toFloat()
            .set("system:time_start", ee_session.Date(date).millis())
        )

    return _s1_image


@pytest.fixture
def s1_collection(ee_session, s1_image):
    """Three images inside the Mar-May 2024 window, plus one in Jan 2024 and one in
    Apr 2023 carrying extreme values that must be filtered out by the seasonal
    builder."""
    dates = ["2024-03-15", "2024-04-15", "2024-05-15"]
    images = [s1_image(vv, vh, d) for (vv, vh), d in zip(IN_SEASON_DB, dates)]
    images += [s1_image(-2.0, -3.0, "2024-01-15"), s1_image(-2.0, -3.0, "2023-04-15")]
    return ee_session.ImageCollection(images)


# --------------------------------------------------------------------------- #
# Pure tests                                                                   #
# --------------------------------------------------------------------------- #
def test_s1_timescan_bands_order():
    from eetools.constants import S1_TIMESCAN_BANDS

    assert len(S1_TIMESCAN_BANDS) == 25
    assert len(set(S1_TIMESCAN_BANDS)) == 25
    assert S1_TIMESCAN_BANDS[:5] == [
        "VV_min",
        "VV_max",
        "VV_mean",
        "VV_stdDev",
        "VV_median",
    ]
    assert S1_TIMESCAN_BANDS[-5:] == [
        "VV_median_VH_median_nat",
        "VV_median_VH_median_db",
        "VV_CV",
        "VH_CV",
        "NDCV",
    ]


@pytest.mark.parametrize("season_months", [(5, 3), (0, 4), (11, 13)])
def test_build_seasonal_s1_timescan_rejects_invalid_months(season_months):
    from eetools.sensors.sentinel1.timescan import build_seasonal_s1_timescan

    with pytest.raises(ValueError, match="season_months"):
        build_seasonal_s1_timescan(
            None,  # type: ignore[arg-type]
            year=2024,
            season_months=season_months,
            season_name="wet",
        )


# --------------------------------------------------------------------------- #
# Backscatter conversions                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.ee
def test_db_to_linear(ee_session, first_value, s1_image):
    from eetools.sensors.sentinel1.timescan import db_to_linear

    out = db_to_linear(s1_image(-10.0, -20.0, "2024-01-01"))
    assert out.bandNames().getInfo() == ["VV", "VH"]
    assert out.get("system:time_start").getInfo() is not None
    assert first_value(out, "VV") == pytest.approx(0.1, rel=1e-5)
    assert first_value(out, "VH") == pytest.approx(0.01, rel=1e-5)


@pytest.mark.ee
def test_add_polarization_ratios(ee_session, first_value, s1_image):
    from eetools.sensors.sentinel1.timescan import add_polarization_ratios

    out = add_polarization_ratios(s1_image(-10.0, -20.0, "2024-01-01"))
    assert out.bandNames().getInfo() == ["VV", "VH", "VVVH_ratio", "VVVH_ND_ratio"]
    # Linear VV = 0.1, VH = 0.01 -> ratio 10, ND (0.09 / 0.11).
    assert first_value(out, "VVVH_ratio") == pytest.approx(10.0, rel=1e-5)
    assert first_value(out, "VVVH_ND_ratio") == pytest.approx(0.09 / 0.11, rel=1e-5)


# --------------------------------------------------------------------------- #
# Timescan                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.ee
def test_build_s1_timescan_values(ee_session, point, s1_collection):
    from eetools.constants import S1_TIMESCAN_BANDS
    from eetools.sensors.sentinel1.timescan import build_s1_timescan

    in_season = s1_collection.filterDate("2024-03-01", "2024-06-01")
    timescan = build_s1_timescan(in_season)
    assert timescan.bandNames().getInfo() == S1_TIMESCAN_BANDS

    vv = [vv for vv, _ in IN_SEASON_DB]
    vh = [vh for _, vh in IN_SEASON_DB]
    ratio = [_linear(a) / _linear(b) for a, b in IN_SEASON_DB]
    nd_ratio = [
        (_linear(a) - _linear(b)) / (_linear(a) + _linear(b)) for a, b in IN_SEASON_DB
    ]
    vv_cv = statistics.pstdev(map(_linear, vv)) / statistics.mean(map(_linear, vv))
    vh_cv = statistics.pstdev(map(_linear, vh)) / statistics.mean(map(_linear, vh))

    expected = {}
    for band, values in [
        ("VV", vv),
        ("VH", vh),
        ("VVVH_ratio", ratio),
        ("VVVH_ND_ratio", nd_ratio),
    ]:
        expected[f"{band}_min"] = min(values)
        expected[f"{band}_max"] = max(values)
        expected[f"{band}_mean"] = statistics.mean(values)
        expected[f"{band}_stdDev"] = statistics.pstdev(values)
        expected[f"{band}_median"] = statistics.median(values)
    expected["VV_median_VH_median_nat"] = -12.0 - -20.0
    expected["VV_median_VH_median_db"] = -12.0 / -20.0
    expected["VV_CV"] = vv_cv
    expected["VH_CV"] = vh_cv
    expected["NDCV"] = (vv_cv - vh_cv) / (vv_cv + vh_cv)

    # One reduceRegion for all 25 bands instead of a first_value() call per band.
    values = timescan.reduceRegion(
        reducer=ee_session.Reducer.first(), geometry=point, scale=1
    ).getInfo()
    for band, value in expected.items():
        assert values[band] == pytest.approx(value, rel=1e-4, abs=1e-6), band


@pytest.mark.ee
def test_build_seasonal_s1_timescan_filters_and_suffixes(
    ee_session, first_value, s1_collection
):
    from eetools.constants import S1_TIMESCAN_BANDS
    from eetools.sensors.sentinel1.timescan import build_seasonal_s1_timescan

    timescan = build_seasonal_s1_timescan(
        s1_collection, year=2024, season_months=(3, 5), season_name="wet"
    )
    assert timescan.bandNames().getInfo() == [f"{b}_wet" for b in S1_TIMESCAN_BANDS]
    # The Jan 2024 and Apr 2023 images (VV = -2 dB) are outside the window.
    assert first_value(timescan, "VV_max_wet") == pytest.approx(-10.0, rel=1e-5)
    assert timescan.get("image_count").getInfo() == 3
    assert timescan.get("season").getInfo() == "wet"
    assert timescan.get("season_months").getInfo() == "3-5"
    assert timescan.get("year").getInfo() == 2024
    assert (
        timescan.get("system:time_start").getInfo()
        == ee_session.Date("2024-03-01").millis().getInfo()
    )


@pytest.mark.ee
def test_build_seasonal_s1_timescan_empty_season(
    ee_session, first_value, s1_collection
):
    from eetools.constants import S1_TIMESCAN_BANDS
    from eetools.sensors.sentinel1.timescan import build_seasonal_s1_timescan

    # No synthetic images fall in Jul-Oct 2024: bands are kept but fully masked.
    timescan = build_seasonal_s1_timescan(
        s1_collection, year=2024, season_months=(7, 10), season_name="dry"
    )
    assert timescan.bandNames().getInfo() == [f"{b}_dry" for b in S1_TIMESCAN_BANDS]
    assert timescan.get("image_count").getInfo() == 0
    assert first_value(timescan, "VV_min_dry") is None
    assert first_value(timescan, "NDCV_dry") is None


@pytest.mark.ee
@pytest.mark.slow
def test_build_seasonal_s1_timescan_real(ee_session, small_aoi):
    from eetools.constants import S1_TIMESCAN_BANDS, SEASON_DRY
    from eetools.sensors.sentinel1.preprocessing import get_s1_grd_collection
    from eetools.sensors.sentinel1.timescan import build_seasonal_s1_timescan

    # Only ascending passes cover the small_aoi (coastal Kenya) in 2024.
    col = get_s1_grd_collection(
        small_aoi,
        ee_session.Date("2024-01-01"),
        ee_session.Date("2025-01-01"),
        orbit_pass="ASCENDING",
    )
    timescan = build_seasonal_s1_timescan(
        col, year=2024, season_months=SEASON_DRY, season_name="dry"
    )
    if timescan.get("image_count").getInfo() == 0:
        pytest.skip("no Sentinel-1 scenes for the AOI/season")
    assert timescan.bandNames().getInfo() == [f"{b}_dry" for b in S1_TIMESCAN_BANDS]
    stats = timescan.reduceRegion(
        reducer=ee_session.Reducer.mean(), geometry=small_aoi, scale=100, tileScale=4
    ).getInfo()
    # Land backscatter in dB is negative; CVs are positive; NDCV is in [-1, 1].
    assert stats["VV_median_dry"] < 0
    assert stats["VV_CV_dry"] > 0
    assert -1 <= stats["NDCV_dry"] <= 1
