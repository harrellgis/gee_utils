"""Tests for the Landsat 8 masking and preprocessing modules.

QA bit logic and the spectral water mask are verified deterministically on
synthetic constant images. The collection builders that hit the real
``LANDSAT/LC08/C02/T1_L2`` archive are marked ``slow``.
"""

import pytest

pytestmark = pytest.mark.ee


def _qa_image(ee, qa_pixel, qa_radsat, data=0.5):
    return ee.Image.cat(
        [
            ee.Image.constant(qa_pixel).toInt().rename("QA_PIXEL"),
            ee.Image.constant(qa_radsat).toInt().rename("QA_RADSAT"),
            ee.Image.constant(data).rename("SR_B5"),
        ]
    )


@pytest.mark.parametrize(
    "qa_pixel, qa_radsat, expected",
    [
        (0, 0, 0),  # clear
        (1, 0, 1),  # QA_PIXEL bit 0 set (fill) -> flagged
        (0b10000, 0, 1),  # bit 4 (cloud) set -> flagged
        (0b100000, 0, 0),  # bit 5 (outside 0-4 window) -> not flagged
        (0, 1, 1),  # saturated band -> flagged
    ],
)
def test_l8_add_cld_shdw_mask_bits(
    ee_session, first_value, qa_pixel, qa_radsat, expected
):
    from eetools.sensors.landsat.masking import add_cld_shdw_mask

    out = add_cld_shdw_mask(_qa_image(ee_session, qa_pixel, qa_radsat))
    assert "cloudmask" in out.bandNames().getInfo()
    assert first_value(out, "cloudmask") == expected


def test_l8_apply_cld_shdw_mask_drops_flagged_pixels(ee_session, first_value):
    from eetools.sensors.landsat.masking import (
        add_cld_shdw_mask,
        apply_cld_shdw_mask,
    )

    flagged = apply_cld_shdw_mask(add_cld_shdw_mask(_qa_image(ee_session, 1, 0)))
    clear = apply_cld_shdw_mask(add_cld_shdw_mask(_qa_image(ee_session, 0, 0)))

    # Flagged pixel is masked out (no value); clear pixel survives.
    assert first_value(flagged, "SR_B5") is None
    assert first_value(clear, "SR_B5") == pytest.approx(0.5)


def test_l8_mask_edges_is_identity(ee_session):
    from eetools.sensors.landsat.masking import mask_edges

    img = ee_session.Image.constant(1).rename("b")
    assert mask_edges(img).bandNames().getInfo() == ["b"]


def _water_composite(ee, mndwi, ndvi, nir):
    return ee.ImageCollection(
        [
            ee.Image.cat(
                [
                    ee.Image.constant(mndwi).rename("MNDWI"),
                    ee.Image.constant(ndvi).rename("NDVI"),
                    ee.Image.constant(nir).rename("SR_B5"),
                ]
            )
        ]
    )


def test_l8_non_water_mask_flags_water(ee_session, first_value):
    from eetools.sensors.landsat.masking import build_l8_non_water_mask

    water = build_l8_non_water_mask(_water_composite(ee_session, 0.5, 0.1, 0.05))
    assert first_value(water, "non_water") == 0


def test_l8_non_water_mask_keeps_land(ee_session, first_value):
    from eetools.sensors.landsat.masking import build_l8_non_water_mask

    land = build_l8_non_water_mask(_water_composite(ee_session, -0.2, 0.6, 0.4))
    assert first_value(land, "non_water") == 1


def test_apply_water_mask_drops_water_pixels(ee_session, first_value):
    from eetools.sensors.landsat.masking import apply_water_mask

    img = ee_session.Image.constant(0.5).rename("b")
    non_water = ee_session.Image.constant(0).rename("non_water")  # everything water
    masked = apply_water_mask(img, non_water)
    assert first_value(masked, "b") is None


def test_process_l8_image_adds_indices_and_keeps_time(ee_session):
    from eetools.sensors.landsat.preprocessing import process_l8_image

    raw = (
        ee_session.Image.constant([8000, 9000, 10000, 20000, 15000, 12000])
        .rename(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
        .set("system:time_start", ee_session.Date("2021-06-01").millis())
    )
    out = ee_session.Image(process_l8_image(raw))
    names = out.bandNames().getInfo()
    assert "NDVI" in names
    # Landsat keeps its native SR_B* band names (only HLS renames to a logical
    # schema); the band map just tells calc_indices which band is which.
    assert "SR_B4" in names
    assert out.get("system:time_start").getInfo() is not None


@pytest.mark.slow
def test_get_l8_sr_collection_real(ee_session, small_aoi):
    from eetools.sensors.landsat.preprocessing import get_l8_sr_collection

    col = get_l8_sr_collection(
        small_aoi,
        ee_session.Date("2021-01-01"),
        ee_session.Date("2021-04-01"),
        apply_water_masking=False,
    )
    assert isinstance(col, ee_session.ImageCollection)
    if col.size().getInfo() == 0:
        pytest.skip("no Landsat scenes for the AOI/window")
    assert "NDVI" in col.first().bandNames().getInfo()


# --------------------------------------------------------------------------- #
# Landsat 9 (OLI) — identical band layout to Landsat 8
# --------------------------------------------------------------------------- #
def test_process_l9_image_adds_indices_and_keeps_time(ee_session):
    from eetools.sensors.landsat.preprocessing import process_l9_image

    raw = (
        ee_session.Image.constant([8000, 9000, 10000, 20000, 15000, 12000])
        .rename(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
        .set("system:time_start", ee_session.Date("2022-06-01").millis())
    )
    out = ee_session.Image(process_l9_image(raw))
    names = out.bandNames().getInfo()
    assert "NDVI" in names
    assert "SR_B4" in names  # OLI red
    assert out.get("system:time_start").getInfo() is not None


# --------------------------------------------------------------------------- #
# Landsat 5 / 7 (TM / ETM+) — shifted band numbering vs OLI
# --------------------------------------------------------------------------- #
def test_process_l5_image_uses_tm_band_map(ee_session, first_value):
    # The point of the TM band map: NDVI must come from SR_B4 (NIR) / SR_B3 (Red),
    # not the OLI SR_B5/SR_B4. Verify the computed value against the TM bands.
    from eetools.constants import LANDSAT_C2_ADD_OFFSET as OFF
    from eetools.constants import LANDSAT_C2_SCALE_FACTOR as SF
    from eetools.sensors.landsat.preprocessing import process_l5_image

    # TM bands: SR_B1, SR_B2, SR_B3(red), SR_B4(nir), SR_B5, SR_B7. Raw values kept
    # high enough that scaled reflectance stays positive (normalizedDifference masks
    # negative inputs).
    raw = (
        ee_session.Image.constant([10000, 10000, 8000, 20000, 15000, 9000])
        .rename(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"])
        .set("system:time_start", ee_session.Date("2005-06-01").millis())
    )
    out = ee_session.Image(process_l5_image(raw))
    assert "NDVI" in out.bandNames().getInfo()

    red = 8000 * SF + OFF
    nir = 20000 * SF + OFF
    expected_ndvi = (nir - red) / (nir + red)
    assert first_value(out, "NDVI") == pytest.approx(expected_ndvi, rel=1e-6)


def test_process_l7_image_adds_indices_and_keeps_time(ee_session):
    from eetools.sensors.landsat.preprocessing import process_l7_image

    raw = (
        ee_session.Image.constant([10000, 10000, 8000, 20000, 15000, 9000])
        .rename(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"])
        .set("system:time_start", ee_session.Date("2001-06-01").millis())
    )
    out = ee_session.Image(process_l7_image(raw))
    names = out.bandNames().getInfo()
    assert "NDVI" in names
    assert "SR_B3" in names  # TM/ETM+ red
    assert out.get("system:time_start").getInfo() is not None


@pytest.mark.slow
def test_get_l9_sr_collection_real(ee_session, small_aoi):
    from eetools.sensors.landsat.preprocessing import get_l9_sr_collection

    col = get_l9_sr_collection(
        small_aoi,
        ee_session.Date("2022-01-01"),
        ee_session.Date("2022-06-01"),
        apply_water_masking=False,
    )
    assert isinstance(col, ee_session.ImageCollection)
    if col.size().getInfo() == 0:
        pytest.skip("no Landsat 9 scenes for the AOI/window")
    assert "NDVI" in col.first().bandNames().getInfo()


@pytest.mark.slow
def test_get_l7_sr_collection_real(ee_session, small_aoi):
    from eetools.sensors.landsat.preprocessing import get_l7_sr_collection

    col = get_l7_sr_collection(
        small_aoi,
        ee_session.Date("2001-01-01"),
        ee_session.Date("2001-06-01"),
        apply_water_masking=False,
    )
    assert isinstance(col, ee_session.ImageCollection)
    if col.size().getInfo() == 0:
        pytest.skip("no Landsat 7 scenes for the AOI/window")
    assert "NDVI" in col.first().bandNames().getInfo()


@pytest.mark.slow
def test_get_l5_sr_collection_real(ee_session, small_aoi):
    from eetools.sensors.landsat.preprocessing import get_l5_sr_collection

    col = get_l5_sr_collection(
        small_aoi,
        ee_session.Date("2010-01-01"),
        ee_session.Date("2010-06-01"),
        apply_water_masking=False,
    )
    assert isinstance(col, ee_session.ImageCollection)
    if col.size().getInfo() == 0:
        pytest.skip("no Landsat 5 scenes for the AOI/window")
    assert "NDVI" in col.first().bandNames().getInfo()
