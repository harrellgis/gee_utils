"""Tests for the Dynamic World sensor module.

cover_types validation raises before any Earth Engine call, so those checks are pure.
The cover-type mask is checked on constant images; the real-collection builder is slow.
"""

import pytest


# --------------------------------------------------------------------------- #
# Pure validation (no Earth Engine)
# --------------------------------------------------------------------------- #
def test_get_dynamic_world_collection_rejects_empty_cover_types():
    from eetools.sensors.dynamicworld.preprocessing import get_dynamic_world_collection

    with pytest.raises(ValueError, match="non-empty"):
        get_dynamic_world_collection(
            aoi=None, start_date=None, end_date=None, cover_types=[]
        )


def test_get_dynamic_world_collection_rejects_invalid_cover_types():
    from eetools.sensors.dynamicworld.preprocessing import get_dynamic_world_collection

    with pytest.raises(ValueError, match="invalid Dynamic World class"):
        get_dynamic_world_collection(
            aoi=None, start_date=None, end_date=None, cover_types=[1, 9]
        )


# --------------------------------------------------------------------------- #
# Earth Engine behaviour (synthetic constant images)
# --------------------------------------------------------------------------- #
@pytest.mark.ee
def test_mask_to_cover_types_keeps_and_drops(ee_session, first_value):
    from eetools.sensors.dynamicworld.masking import mask_to_cover_types

    # A "water" (label 0) pixel with a water probability band.
    img = ee_session.Image.cat(
        [
            ee_session.Image.constant(0).toInt().rename("label"),
            ee_session.Image.constant(0.9).rename("water"),
        ]
    )
    # Class 0 requested -> kept; only {1, 2} requested -> label 0 masked out.
    assert first_value(mask_to_cover_types(img, [0]), "water") == pytest.approx(0.9)
    assert first_value(mask_to_cover_types(img, [1, 2]), "water") is None


@pytest.mark.ee
def test_mask_to_cover_types_rejects_empty(ee_session):
    from eetools.sensors.dynamicworld.masking import mask_to_cover_types

    img = ee_session.Image.constant(0).toInt().rename("label")
    with pytest.raises(ValueError):
        mask_to_cover_types(img, [])


@pytest.mark.ee
@pytest.mark.slow
def test_get_dynamic_world_collection_real(ee_session, small_aoi):
    from eetools.sensors.dynamicworld.preprocessing import get_dynamic_world_collection

    col = get_dynamic_world_collection(
        small_aoi,
        ee_session.Date("2022-01-01"),
        ee_session.Date("2022-04-01"),
    )
    assert isinstance(col, ee_session.ImageCollection)
    if col.size().getInfo() == 0:
        pytest.skip("no Dynamic World scenes for the AOI/window")
    assert "label" in col.first().bandNames().getInfo()


@pytest.mark.ee
@pytest.mark.slow
def test_get_dynamic_world_collection_cover_types_real(ee_session, small_aoi):
    from eetools.sensors.dynamicworld.preprocessing import get_dynamic_world_collection

    # water + flooded vegetation only; builder should wire the mask without error.
    col = get_dynamic_world_collection(
        small_aoi,
        ee_session.Date("2022-01-01"),
        ee_session.Date("2022-04-01"),
        cover_types=[0, 3],
    )
    assert isinstance(col, ee_session.ImageCollection)
    if col.size().getInfo() == 0:
        pytest.skip("no Dynamic World scenes for the AOI/window")
    assert "label" in col.first().bandNames().getInfo()
