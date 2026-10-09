"""Tests for the NASA SMAP L4 soil moisture preprocessing module.

The daily-mean aggregation is checked on a synthetic 3-hourly collection; the
real-collection builder is marked ``slow``.
"""

import pytest

pytestmark = pytest.mark.ee


def _timestep(ee, iso, value):
    return (
        ee.Image.constant(value)
        .toFloat()
        .rename("sm_surface")
        .set("system:time_start", ee.Date(iso).millis())
    )


@pytest.fixture
def three_hourly_collection(ee_session):
    """Two timesteps on 2024-06-01, one on 2024-06-02, none on 2024-06-03."""
    return ee_session.ImageCollection(
        [
            _timestep(ee_session, "2024-06-01T01:30:00", 0.1),
            _timestep(ee_session, "2024-06-01T04:30:00", 0.3),
            _timestep(ee_session, "2024-06-02T01:30:00", 0.5),
        ]
    )


# --------------------------------------------------------------------------- #
# Daily aggregation                                                            #
# --------------------------------------------------------------------------- #


def test_daily_means_averages_each_day(
    ee_session, first_value, three_hourly_collection
):
    from eetools.sensors.smap.preprocessing import _daily_means

    out = _daily_means(
        three_hourly_collection,
        ee_session.Date("2024-06-01"),
        ee_session.Date("2024-06-04"),
    )
    images = out.toList(10)
    # The empty 2024-06-03 is dropped rather than returned bandless.
    assert out.size().getInfo() == 2
    assert first_value(images.get(0), "sm_surface") == pytest.approx(0.2)
    assert first_value(images.get(1), "sm_surface") == pytest.approx(0.5)


def test_daily_means_sets_date_properties(ee_session, three_hourly_collection):
    from eetools.sensors.smap.preprocessing import _daily_means

    out = _daily_means(
        three_hourly_collection,
        ee_session.Date("2024-06-01"),
        ee_session.Date("2024-06-03"),
    )
    props = ee_session.Image(out.first()).toDictionary(
        ["year", "month", "day", "image_count", "system:time_start"]
    )
    info = props.getInfo()
    assert (info["year"], info["month"], info["day"]) == (2024, 6, 1)
    assert info["image_count"] == 2
    assert info["system:time_start"] == ee_session.Date("2024-06-01").millis().getInfo()


# --------------------------------------------------------------------------- #
# Real-collection builder                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_get_smap_collection_daily_real(ee_session, small_aoi):
    from eetools.constants import SMAP_SM_BANDS
    from eetools.sensors.smap.preprocessing import get_smap_collection

    col = get_smap_collection(small_aoi, "2024-06-01", "2024-06-04")
    assert col.size().getInfo() == 3  # one image per day
    first = ee_session.Image(col.first())
    assert first.bandNames().getInfo() == SMAP_SM_BANDS
    assert first.get("image_count").getInfo() == 8  # 3-hourly timesteps
    sm = (
        first.reduceRegion(ee_session.Reducer.mean(), small_aoi.centroid(), 9000)
        .get("sm_surface")
        .getInfo()
    )
    assert sm is None or 0.0 <= sm <= 1.0


@pytest.mark.slow
def test_get_smap_collection_native_real(ee_session, small_aoi):
    from eetools.sensors.smap.preprocessing import get_smap_collection

    col = get_smap_collection(
        small_aoi, "2024-06-01", "2024-06-02", bands=["sm_rootzone"], daily=False
    )
    assert col.size().getInfo() == 8
    assert col.first().bandNames().getInfo() == ["sm_rootzone"]


@pytest.mark.slow
def test_get_smap_collection_rejects_pre_mission(ee_session, small_aoi):
    from eetools.sensors.smap.preprocessing import get_smap_collection

    with pytest.raises(ValueError):
        get_smap_collection(small_aoi, "2010-01-01", "2011-01-01")
