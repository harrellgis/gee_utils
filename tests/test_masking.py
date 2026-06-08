"""Tests for the shared sensor-agnostic masking helpers.

The per-sensor masking tests exercise these via delegation; these cover the
generic API directly (arbitrary NIR band name and mask band name).
"""

import pytest

pytestmark = pytest.mark.ee


def test_build_non_water_mask_flags_water(ee_session, first_value):
    from eetools.sensors.masking import build_non_water_mask

    col = ee_session.ImageCollection(
        [
            ee_session.Image.cat(
                [
                    ee_session.Image.constant(0.5).rename("MNDWI"),
                    ee_session.Image.constant(0.1).rename("NDVI"),
                    ee_session.Image.constant(0.05).rename("CUSTOM_NIR"),
                ]
            )
        ]
    )
    mask = build_non_water_mask(col, nir_band="CUSTOM_NIR")
    assert first_value(mask, "non_water") == 0


def test_build_non_water_mask_keeps_land(ee_session, first_value):
    from eetools.sensors.masking import build_non_water_mask

    col = ee_session.ImageCollection(
        [
            ee_session.Image.cat(
                [
                    ee_session.Image.constant(-0.2).rename("MNDWI"),
                    ee_session.Image.constant(0.6).rename("NDVI"),
                    ee_session.Image.constant(0.4).rename("CUSTOM_NIR"),
                ]
            )
        ]
    )
    mask = build_non_water_mask(col, nir_band="CUSTOM_NIR")
    assert first_value(mask, "non_water") == 1


def test_apply_cloud_mask_drops_flagged(ee_session, first_value):
    from eetools.sensors.masking import apply_cloud_mask

    img = ee_session.Image.cat(
        [
            ee_session.Image.constant(0.5).rename("b"),
            ee_session.Image.constant(1).rename("cloudmask"),
        ]
    )
    assert first_value(apply_cloud_mask(img), "b") is None


def test_apply_cloud_mask_keeps_clear(ee_session, first_value):
    from eetools.sensors.masking import apply_cloud_mask

    img = ee_session.Image.cat(
        [
            ee_session.Image.constant(0.5).rename("b"),
            ee_session.Image.constant(0).rename("cloudmask"),
        ]
    )
    assert first_value(apply_cloud_mask(img), "b") == pytest.approx(0.5)


def test_apply_cloud_mask_custom_band(ee_session, first_value):
    from eetools.sensors.masking import apply_cloud_mask

    img = ee_session.Image.cat(
        [
            ee_session.Image.constant(0.5).rename("b"),
            ee_session.Image.constant(1).rename("qa"),
        ]
    )
    assert first_value(apply_cloud_mask(img, mask_band="qa"), "b") is None


def test_apply_water_mask_drops_water(ee_session, first_value):
    from eetools.sensors.masking import apply_water_mask

    img = ee_session.Image.constant(0.5).rename("b")
    non_water = ee_session.Image.constant(0).rename("non_water")
    assert first_value(apply_water_mask(img, non_water), "b") is None
