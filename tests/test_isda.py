"""Tests for the iSDA Africa Total Soil Carbon module.

The function constructs the real iSDA asset internally, so this is marked
``slow``.
"""

import pytest

pytestmark = [pytest.mark.ee, pytest.mark.slow]


def test_get_soil_carbon_band(ee_session, small_aoi):
    from eetools.sensors.isda.preprocessing import get_soil_carbon

    img = get_soil_carbon(aoi=small_aoi)
    assert img.bandNames().getInfo() == ["soil_carbon"]


def test_get_soil_carbon_no_aoi_returns_unclipped_band(ee_session):
    from eetools.sensors.isda.preprocessing import get_soil_carbon

    img = get_soil_carbon()
    assert img.bandNames().getInfo() == ["soil_carbon"]
