"""Tests for the CHIRPS preprocessing module."""

import pytest

pytestmark = pytest.mark.ee


def test_process_chirps_image_selects_precip_and_keeps_props(ee_session, first_value):
    from eetools.constants import CHIRPS_PRECIP_BAND
    from eetools.sensors.chirps.preprocessing import process_chirps_image

    raw = (
        ee_session.Image.cat(
            [
                ee_session.Image.constant(12.5).rename(CHIRPS_PRECIP_BAND),
                ee_session.Image.constant(1).rename("extra"),
            ]
        )
        .set("system:time_start", ee_session.Date("2021-06-01").millis())
        .set("year", 2021)
        .set("month", 6)
        .set("day", 1)
    )
    out = ee_session.Image(process_chirps_image(raw))
    assert out.bandNames().getInfo() == [CHIRPS_PRECIP_BAND]
    assert first_value(out, CHIRPS_PRECIP_BAND) == pytest.approx(12.5)
    assert out.get("year").getInfo() == 2021
    assert out.get("month").getInfo() == 6


@pytest.mark.slow
def test_get_chirps_collection_real(ee_session, small_aoi):
    from eetools.constants import CHIRPS_PRECIP_BAND
    from eetools.sensors.chirps.preprocessing import get_chirps_collection

    col = get_chirps_collection(small_aoi, "2020-01-01", "2020-01-15")
    assert isinstance(col, ee_session.ImageCollection)
    # CHIRPS is daily and global, so this window always has images.
    assert col.size().getInfo() > 0
    assert col.first().bandNames().getInfo() == [CHIRPS_PRECIP_BAND]
