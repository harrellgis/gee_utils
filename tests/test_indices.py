"""Tests for eetools.sensors.indices.

The index math is verified deterministically against a synthetic constant
reflectance image (see the ``synthetic_reflectance_image`` fixture):
    blue=0.10 green=0.20 red=0.30 red_edge=0.40 nir=0.50 swir1=0.25 swir2=0.15

so e.g. NDVI = (0.50 - 0.30) / (0.50 + 0.30) = 0.25.
"""

import pytest

pytestmark = pytest.mark.ee


def test_calc_ndvi_value(synthetic_reflectance_image, first_value):
    from eetools.sensors.indices import calc_ndvi

    out = calc_ndvi(synthetic_reflectance_image, nir_band="nir", red_band="red")
    assert first_value(out, "NDVI") == pytest.approx(0.25)


def test_calc_ndvi_custom_output_band(synthetic_reflectance_image):
    from eetools.sensors.indices import calc_ndvi

    out = calc_ndvi(
        synthetic_reflectance_image,
        nir_band="nir",
        red_band="red",
        output_band="ndvi_x",
    )
    assert out.bandNames().getInfo() == ["ndvi_x"]


def test_calc_ndwi_value(synthetic_reflectance_image, first_value):
    from eetools.sensors.indices import calc_ndwi

    # (green - nir) / (green + nir) = (0.20 - 0.50) / 0.70
    out = calc_ndwi(synthetic_reflectance_image, green_band="green", nir_band="nir")
    assert first_value(out, "NDWI") == pytest.approx((0.20 - 0.50) / (0.20 + 0.50))


def test_calc_savi_reduces_to_ndvi_scaled(synthetic_reflectance_image, first_value):
    from eetools.sensors.indices import calc_savi

    # ((nir - red) / (nir + red + L)) * (1 + L), L=0.5
    out = calc_savi(synthetic_reflectance_image, nir_band="nir", red_band="red")
    expected = ((0.50 - 0.30) / (0.50 + 0.30 + 0.5)) * 1.5
    assert first_value(out, "SAVI") == pytest.approx(expected)


def test_calc_evi_value(synthetic_reflectance_image, first_value):
    from eetools.sensors.indices import calc_evi

    out = calc_evi(
        synthetic_reflectance_image,
        nir_band="nir",
        red_band="red",
        blue_band="blue",
    )
    expected = 2.5 * ((0.50 - 0.30) / (0.50 + 6 * 0.30 - 7.5 * 0.10 + 1))
    assert first_value(out, "EVI") == pytest.approx(expected)


def test_calc_nirv_value(synthetic_reflectance_image, first_value):
    from eetools.sensors.indices import calc_nirv

    # NIR * NDVI = 0.50 * 0.25
    out = calc_nirv(synthetic_reflectance_image, nir_band="nir", red_band="red")
    assert first_value(out, "NIRv") == pytest.approx(0.50 * 0.25)


def test_calc_bsi_value(synthetic_reflectance_image, first_value):
    from eetools.sensors.indices import calc_bsi

    # ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))
    out = calc_bsi(
        synthetic_reflectance_image,
        swir1_band="swir1",
        red_band="red",
        nir_band="nir",
        blue_band="blue",
    )
    num = (0.25 + 0.30) - (0.50 + 0.10)
    den = (0.25 + 0.30) + (0.50 + 0.10)
    assert first_value(out, "BSI") == pytest.approx(num / den)


def test_calc_fpar_clamped(synthetic_reflectance_image, first_value):
    from eetools.sensors.indices import calc_fpar

    # NDVI=0.25, soil=0.15, veg=0.80 -> (0.25-0.15)/(0.80-0.15)*0.95
    out = calc_fpar(synthetic_reflectance_image, nir_band="nir", red_band="red")
    expected = (0.25 - 0.15) / (0.80 - 0.15) * 0.95
    assert first_value(out, "Fpar") == pytest.approx(expected)


def test_calc_fpar_upper_clamp(ee_session, first_value):
    from eetools.sensors.indices import calc_fpar

    # Force NDVI above the veg endpoint so the result clamps at max_fpar.
    img = ee_session.Image.constant([0.9, 0.01]).rename(["nir", "red"])
    out = calc_fpar(img, nir_band="nir", red_band="red")
    assert first_value(out, "Fpar") == pytest.approx(0.95)


def test_calc_kndvi_fixed_sigma_value(synthetic_reflectance_image, first_value):
    import math

    from eetools.constants import SIGMA
    from eetools.sensors.indices import calc_kndvi_fixed_sigma

    out = calc_kndvi_fixed_sigma(
        synthetic_reflectance_image, red_band="red", nir_band="nir"
    )
    d2 = (0.50 - 0.30) ** 2
    expected = math.tanh(d2 / (4.0 * SIGMA * SIGMA))
    assert first_value(out, "kNDVI_fixed") == pytest.approx(expected)


def test_calc_indices_appends_full_band_set(
    synthetic_reflectance_image, reflectance_band_map
):
    from eetools.sensors.indices import calc_indices

    out = calc_indices(
        synthetic_reflectance_image, band_map=reflectance_band_map, include_ndre=True
    )
    names = out.bandNames().getInfo()
    for band in [
        "NDVI",
        "kNDVI_fixed",
        "Fpar",
        "EVI",
        "NDWI",
        "MNDWI",
        "SAVI",
        "NDMI",
        "NBR",
        "NIRv",
        "BSI",
        "NDRE",
    ]:
        assert band in names


def test_calc_indices_excludes_ndre_by_default(
    synthetic_reflectance_image, reflectance_band_map
):
    from eetools.sensors.indices import calc_indices

    out = calc_indices(synthetic_reflectance_image, band_map=reflectance_band_map)
    assert "NDRE" not in out.bandNames().getInfo()


def test_calc_veg_indices_band_set(synthetic_reflectance_image, reflectance_band_map):
    from eetools.sensors.indices import calc_veg_indices

    out = calc_veg_indices(synthetic_reflectance_image, band_map=reflectance_band_map)
    names = out.bandNames().getInfo()
    for band in ["NDVI", "kNDVI_fixed", "Fpar", "EVI", "NDWI", "MNDWI", "SAVI"]:
        assert band in names
    # calc_veg_indices does not add the disturbance/moisture indices.
    assert "NBR" not in names


def test_select_base_bands_selects_subset(synthetic_reflectance_image):
    from eetools.sensors.indices import select_base_bands

    out = select_base_bands(synthetic_reflectance_image, input_bands=["red", "nir"])
    assert out.bandNames().getInfo() == ["red", "nir"]


def test_select_base_bands_renames(synthetic_reflectance_image):
    from eetools.sensors.indices import select_base_bands

    out = select_base_bands(
        synthetic_reflectance_image,
        input_bands=["red", "nir"],
        output_bands=["RED", "NIR"],
    )
    assert out.bandNames().getInfo() == ["RED", "NIR"]
