"""Tests for the Hansen Global Forest Change module.

The period-range validation runs without Earth Engine (it raises before any ``ee``
call). Band-name / selection behaviour is confirmed against the real Hansen GFC asset
over a small AOI, so those tests are marked ``slow``.
"""

import pytest


# --------------------------------------------------------------------------- #
# Pure validation (no Earth Engine)
# --------------------------------------------------------------------------- #
def test_get_forest_loss_in_period_rejects_reversed_range():
    from eetools.sensors.hansen.preprocessing import get_forest_loss_in_period

    with pytest.raises(ValueError, match="must not exceed"):
        get_forest_loss_in_period(aoi=None, start_year=2020, end_year=2015)


def test_get_forest_loss_in_period_rejects_out_of_coverage():
    from eetools.sensors.hansen.preprocessing import get_forest_loss_in_period

    with pytest.raises(ValueError, match="outside the Hansen GFC coverage"):
        get_forest_loss_in_period(aoi=None, start_year=1999, end_year=2025)


# --------------------------------------------------------------------------- #
# Real-asset integration (small AOI)
# --------------------------------------------------------------------------- #
@pytest.mark.ee
@pytest.mark.slow
def test_get_forest_2000_band_name(ee_session, small_aoi):
    from eetools.sensors.hansen.preprocessing import get_forest_2000

    img = get_forest_2000(aoi=small_aoi)
    assert img.bandNames().getInfo() == ["treecover2000"]


@pytest.mark.ee
@pytest.mark.slow
def test_get_forest_loss_image_band_name(ee_session, small_aoi):
    from eetools.sensors.hansen.preprocessing import get_forest_loss_image

    img = get_forest_loss_image(aoi=small_aoi)
    assert img.bandNames().getInfo() == ["forest_loss"]


@pytest.mark.ee
@pytest.mark.slow
def test_get_forest_loss_year_image_band_name(ee_session, small_aoi):
    from eetools.sensors.hansen.preprocessing import get_forest_loss_year_image

    img = get_forest_loss_year_image(aoi=small_aoi)
    assert img.bandNames().getInfo() == ["lossyear"]


@pytest.mark.ee
@pytest.mark.slow
def test_get_forest_gain_image_band_name(ee_session, small_aoi):
    from eetools.sensors.hansen.preprocessing import get_forest_gain_image

    img = get_forest_gain_image(aoi=small_aoi)
    assert img.bandNames().getInfo() == ["gain"]


@pytest.mark.ee
@pytest.mark.slow
def test_get_tree_cover_mask_is_binary(ee_session, small_aoi):
    from eetools.sensors.hansen.preprocessing import get_tree_cover_mask

    mask = get_tree_cover_mask(aoi=small_aoi, threshold=10)
    stats = mask.reduceRegion(
        reducer=ee_session.Reducer.minMax(),
        geometry=small_aoi,
        scale=30,
        maxPixels=int(1e9),
        tileScale=4,
    ).getInfo()
    values = {v for v in stats.values() if v is not None}
    assert values <= {0, 1}


@pytest.mark.ee
@pytest.mark.slow
def test_get_forest_loss_in_period_band_name(ee_session, small_aoi):
    from eetools.sensors.hansen.preprocessing import get_forest_loss_in_period

    img = get_forest_loss_in_period(
        aoi=small_aoi, start_year=2015, end_year=2025, tree_cover_threshold=10
    )
    assert img.bandNames().getInfo() == ["forest_loss"]
