"""Tests for the Meta/WRI 1m Global Canopy Height module.

The function constructs the real canopy-height mosaic internally, so this is
marked ``slow``.
"""

import pytest

pytestmark = [pytest.mark.ee, pytest.mark.slow]


def test_get_canopy_height_band(ee_session, small_aoi):
    from eetools.sensors.canopy_height.preprocessing import get_canopy_height

    img = get_canopy_height(aoi=small_aoi)
    assert img.bandNames().getInfo() == ["canopy_height"]
