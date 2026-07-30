"""Tests for the Sentinel-2 masking and preprocessing modules.

The s2cloudless/shadow projection in ``add_cld_shdw_mask`` involves
``reproject`` + ``directionalDistanceTransform`` which is expensive to
evaluate per-pixel, so it is checked structurally (band added) rather than by
reducing values. The water mask and band processing are checked on constant
images, and the real-collection builder is marked ``slow``.
"""

import pytest

pytestmark = pytest.mark.ee


def test_s2_mask_edges_keeps_bands(ee_session):
    from eetools.sensors.sentinel.masking import mask_edges

    img = ee_session.Image.cat(
        [
            ee_session.Image.constant(0.3).rename("B8A"),
            ee_session.Image.constant(0.3).rename("B9"),
            ee_session.Image.constant(0.3).rename("B8"),
        ]
    )
    out = mask_edges(img)
    assert set(["B8A", "B9", "B8"]).issubset(set(out.bandNames().getInfo()))


def test_s2_add_cld_shdw_mask_adds_band(ee_session):
    from eetools.sensors.sentinel.masking import add_cld_shdw_mask

    prob = ee_session.Image.constant(80).rename("probability")
    img = (
        ee_session.Image.cat(
            [
                ee_session.Image.constant(0.05).rename("B8"),
                ee_session.Image.constant(4).toInt().rename("SCL"),
            ]
        )
        .set("s2cloudless", prob)
        .set("MEAN_SOLAR_AZIMUTH_ANGLE", 135)
    )
    out = add_cld_shdw_mask(img)
    # bandNames is graph metadata: confirms the band is wired without evaluating
    # the heavy reproject/DDT pixel computation.
    assert "cloudmask" in out.bandNames().getInfo()


def test_s2_non_water_mask(ee_session, first_value):
    from eetools.sensors.sentinel.masking import build_s2_non_water_mask

    col = ee_session.ImageCollection(
        [
            ee_session.Image.cat(
                [
                    ee_session.Image.constant(0.5).rename("MNDWI"),
                    ee_session.Image.constant(0.1).rename("NDVI"),
                    ee_session.Image.constant(0.05).rename("B8"),
                ]
            )
        ]
    )
    assert first_value(build_s2_non_water_mask(col), "non_water") == 0


def test_s2_apply_water_mask(ee_session, first_value):
    from eetools.sensors.sentinel.masking import apply_water_mask

    img = ee_session.Image.constant(0.5).rename("b")
    keep = ee_session.Image.constant(1).rename("non_water")
    drop = ee_session.Image.constant(0).rename("non_water")
    assert first_value(apply_water_mask(img, keep), "b") == pytest.approx(0.5)
    assert first_value(apply_water_mask(img, drop), "b") is None


def test_process_s2_image_scales_and_adds_indices(ee_session, first_value):
    from eetools.constants import S2_SCALE_FACTOR
    from eetools.sensors.sentinel.preprocessing import process_s2_image

    raw = (
        ee_session.Image.constant(
            [800, 1000, 2000, 3000, 4000, 4500, 5000, 5000, 5200, 2500, 1500, 4]
        )
        .rename(
            ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "SCL"]
        )
        .set("system:time_start", ee_session.Date("2021-06-01").millis())
    )
    out = ee_session.Image(process_s2_image(raw))
    names = out.bandNames().getInfo()
    assert "NDVI" in names
    assert "NDRE" in names  # NDRE auto-included: the S2 band map has a red edge
    assert (
        "B1" in names
    )  # coastal aerosol carried through even though unused by indices
    # Reflectance is scaled by S2_SCALE_FACTOR.
    assert first_value(out, "B4") == pytest.approx(3000 * S2_SCALE_FACTOR)
    assert first_value(out, "B1") == pytest.approx(800 * S2_SCALE_FACTOR)
    assert out.get("system:time_start").getInfo() is not None


@pytest.mark.slow
def test_get_s2_sr_collection_real(ee_session, small_aoi):
    from eetools.sensors.sentinel.preprocessing import get_s2_sr_collection

    col = get_s2_sr_collection(
        small_aoi,
        ee_session.Date("2021-06-01"),
        ee_session.Date("2021-07-01"),
        apply_water_masking=False,
    )
    assert isinstance(col, ee_session.ImageCollection)
    if col.size().getInfo() == 0:
        pytest.skip("no Sentinel-2 scenes for the AOI/window")
    assert "NDVI" in col.first().bandNames().getInfo()
