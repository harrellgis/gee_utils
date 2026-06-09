"""Tests for the Copernicus DEM / terrain module.

The terrain derivation (``ee.Terrain.products``) is exercised on a synthetic
elevation image so the band wiring is checked without the network. A ``slow`` tier
confirms the real Copernicus DEM GLO-30 loads and derives end to end.
"""

import pytest

pytestmark = pytest.mark.ee


def test_get_terrain_band_names_on_synthetic_elevation(ee_session, small_aoi):
    from eetools.sensors.dem.preprocessing import get_terrain

    elevation = ee_session.Image.constant(100).rename("elevation")
    terrain = get_terrain(elevation=elevation, aoi=small_aoi)

    bands = terrain.bandNames().getInfo()
    # ee.Terrain.products adds slope/aspect/hillshade and copies elevation through.
    assert {"slope", "aspect", "hillshade"} <= set(bands)
    assert "elevation" in bands


def test_get_terrain_can_drop_elevation(ee_session, small_aoi):
    from eetools.sensors.dem.preprocessing import get_terrain

    elevation = ee_session.Image.constant(100).rename("elevation")
    terrain = get_terrain(elevation=elevation, aoi=small_aoi, add_elevation=False)

    assert sorted(terrain.bandNames().getInfo()) == ["aspect", "hillshade", "slope"]


@pytest.mark.slow
def test_get_copernicus_dem_single_elevation_band(ee_session, small_aoi):
    from eetools.sensors.dem.preprocessing import get_copernicus_dem

    dem = get_copernicus_dem(aoi=small_aoi)
    assert dem.bandNames().getInfo() == ["elevation"]


@pytest.mark.slow
def test_get_terrain_from_canonical_dem(ee_session, small_aoi):
    from eetools.sensors.dem.preprocessing import get_terrain

    terrain = get_terrain(aoi=small_aoi)
    bands = terrain.bandNames().getInfo()
    assert {"elevation", "slope", "aspect", "hillshade"} <= set(bands)
