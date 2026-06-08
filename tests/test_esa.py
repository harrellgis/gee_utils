"""Tests for the ESA WorldCover land-mask module.

Both functions construct the real ESA WorldCover mosaic internally, so these
are marked ``slow``.
"""

import pytest

pytestmark = [pytest.mark.ee, pytest.mark.slow]


def test_get_esa_land_mask_band(ee_session):
    from eetools.sensors.esa.preprocessing import get_esa_land_mask

    mask = get_esa_land_mask()
    assert mask.bandNames().getInfo() == ["land_mask"]


def test_apply_land_mask_preserves_band_and_props(ee_session, small_aoi):
    from eetools.sensors.esa.preprocessing import apply_land_mask

    img = (
        ee_session.Image.constant(0.5)
        .rename("b")
        .clip(small_aoi)
        .set("custom_prop", "kept")
    )
    out = ee_session.Image(apply_land_mask(img))
    assert out.bandNames().getInfo() == ["b"]
    assert out.get("custom_prop").getInfo() == "kept"
