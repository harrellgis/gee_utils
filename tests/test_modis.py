"""Tests for the MODIS LAI/FPAR preprocessing module.

QA bit logic, band scaling, and the fPAR override are checked on synthetic
constant images. The real-collection builder is marked ``slow``.
"""

import pytest

pytestmark = pytest.mark.ee


def _qa_image(ee, fpar_lai_qc, fpar_extra_qc, fpar=50, lai=30):
    return ee.Image.cat(
        [
            ee.Image.constant(fpar).toInt().rename("Fpar"),
            ee.Image.constant(lai).toInt().rename("Lai"),
            ee.Image.constant(5).toInt().rename("FparStdDev"),
            ee.Image.constant(4).toInt().rename("LaiStdDev"),
            ee.Image.constant(fpar_lai_qc).toInt().rename("FparLai_QC"),
            ee.Image.constant(fpar_extra_qc).toInt().rename("FparExtra_QC"),
        ]
    )


def test_modis_qa_keeps_good_pixel(ee_session, first_value):
    from eetools.sensors.modis.preprocessing import _mask_modis_lai_fpar_qa

    # FparLai_QC = 0 -> bit0 clear, bits5-7 == 0 (<=1). FparExtra_QC = 0 -> clear.
    out = _mask_modis_lai_fpar_qa(_qa_image(ee_session, 0, 0))
    assert first_value(out, "Fpar") == 50


def test_modis_qa_drops_bad_quality_pixel(ee_session, first_value):
    from eetools.sensors.modis.preprocessing import _mask_modis_lai_fpar_qa

    # FparLai_QC bit0 set -> MODLAND poor quality -> masked.
    out = _mask_modis_lai_fpar_qa(_qa_image(ee_session, 1, 0))
    assert first_value(out, "Fpar") is None


def test_modis_qa_drops_high_qc_score(ee_session, first_value):
    from eetools.sensors.modis.preprocessing import _mask_modis_lai_fpar_qa

    # bits 5-7 of FparLai_QC = 2 (>1) -> masked.
    out = _mask_modis_lai_fpar_qa(_qa_image(ee_session, 2 << 5, 0))
    assert first_value(out, "Fpar") is None


def test_modis_scale_bands(ee_session, first_value):
    from eetools.constants import FPAR_SCALE_FACTOR, LAI_SCALE_FACTOR
    from eetools.sensors.modis.preprocessing import _scale_modis_lai_fpar_bands

    # copyProperties() returns an ee.Element; in production .map() casts it back
    # to an Image, so the test wraps it explicitly.
    out = ee_session.Image(
        _scale_modis_lai_fpar_bands(_qa_image(ee_session, 0, 0, fpar=50, lai=30))
    )
    assert first_value(out, "Fpar") == pytest.approx(50 * FPAR_SCALE_FACTOR)
    assert first_value(out, "Lai") == pytest.approx(30 * LAI_SCALE_FACTOR)
    # QA bands are carried through unscaled.
    assert "FparLai_QC" in out.bandNames().getInfo()


def test_overwrite_fpar_band(ee_session, first_value):
    from eetools.sensors.modis.preprocessing import overwrite_fpar_band

    img = ee_session.Image.cat(
        [
            ee_session.Image.constant(0.5).rename("nir"),
            ee_session.Image.constant(0.3).rename("red"),
            ee_session.Image.constant(0.0).rename("fpar"),  # to be overwritten
        ]
    )
    out = overwrite_fpar_band(
        img, red_band="red", nir_band="nir", ndvi_soil=0.15, ndvi_veg=0.80
    )
    # NDVI = 0.25 -> (0.25-0.15)/(0.80-0.15)*0.95
    expected = (0.25 - 0.15) / (0.80 - 0.15) * 0.95
    assert first_value(out, "fpar") == pytest.approx(expected)


@pytest.mark.slow
def test_get_modis_lai_fpar_col_real(ee_session, small_aoi):
    from eetools.sensors.modis.preprocessing import get_modis_lai_fpar_col

    col = get_modis_lai_fpar_col(small_aoi, "2021-01-01", "2021-02-01")
    assert isinstance(col, ee_session.ImageCollection)
    if col.size().getInfo() == 0:
        pytest.skip("no MODIS images for the AOI/window")
    assert "Fpar" in col.first().bandNames().getInfo()
