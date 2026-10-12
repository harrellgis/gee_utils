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


# --------------------------------------------------------------------------- #
# MOD13Q1 vegetation indices                                                   #
# --------------------------------------------------------------------------- #


def _vi_image(ee, summary_qa, ndvi=5000):
    return ee.Image.cat(
        [
            ee.Image.constant(ndvi).toInt().rename("NDVI"),
            ee.Image.constant(summary_qa).toInt().rename("SummaryQA"),
        ]
    )


@pytest.mark.parametrize(
    "summary_qa, kept", [(0, True), (1, True), (2, False), (3, False), (-1, False)]
)
def test_modis_vi_qa_mask(ee_session, first_value, summary_qa, kept):
    from eetools.sensors.modis.masking import mask_modis_vi_qa

    out = mask_modis_vi_qa(_vi_image(ee_session, summary_qa))
    assert (first_value(out, "NDVI") is not None) is kept


def test_modis_vi_qa_mask_strict_threshold(ee_session, first_value):
    from eetools.sensors.modis.masking import mask_modis_vi_qa

    # max_summary_qa=0 keeps only "good" retrievals, so marginal is dropped.
    out = mask_modis_vi_qa(_vi_image(ee_session, 1), max_summary_qa=0)
    assert first_value(out, "NDVI") is None


def test_modis_scale_bands_helper(ee_session, first_value):
    from eetools.constants import MODIS_VI_SCALE_FACTOR
    from eetools.sensors.modis.preprocessing import _scale_bands

    img = _vi_image(ee_session, 0, ndvi=5509)
    out = _scale_bands(img, ["NDVI"], MODIS_VI_SCALE_FACTOR)
    assert first_value(out, "NDVI") == pytest.approx(0.5509)
    # Unscaled bands are carried through untouched.
    assert first_value(out, "SummaryQA") == 0


# --------------------------------------------------------------------------- #
# MCD64A1 burned area                                                          #
# --------------------------------------------------------------------------- #


def _burn_image(ee, qa, burn_date=None):
    burn = ee.Image.constant(burn_date or 0).toInt().rename("BurnDate")
    if burn_date is None:
        # GEE masks BurnDate over unburned land rather than storing 0.
        burn = burn.updateMask(0)
    return ee.Image.cat([burn, ee.Image.constant(qa).toInt().rename("QA")])


@pytest.mark.parametrize(
    "qa, burn_date, expected",
    [
        (3, 120, 1),  # land + valid, burned on DOY 120
        (11, 120, 1),  # relabeled-cell bit set, still valid
        (3, None, 0),  # land + valid, BurnDate masked -> unburned
        (0, None, None),  # water -> masked
        (1, None, None),  # land but insufficient data -> masked
    ],
)
def test_add_modis_burned_band(ee_session, first_value, qa, burn_date, expected):
    from eetools.sensors.modis.masking import add_modis_burned_band

    out = add_modis_burned_band(_burn_image(ee_session, qa, burn_date))
    assert first_value(out, "burned") == expected


# --------------------------------------------------------------------------- #
# MOD17A3HGF annual NPP                                                        #
# --------------------------------------------------------------------------- #


def _npp_image(ee, npp, gpp):
    return ee.Image.cat(
        [
            ee.Image.constant(npp).toInt().rename("Npp"),
            ee.Image.constant(gpp).toInt().rename("Gpp"),
            ee.Image.constant(44).toInt().rename("Npp_QC"),
        ]
    )


def test_modis_npp_fill_keeps_valid(ee_session, first_value):
    from eetools.sensors.modis.masking import mask_modis_npp_fill

    out = mask_modis_npp_fill(_npp_image(ee_session, 8574, 13337))
    assert first_value(out, "Npp") == 8574
    assert first_value(out, "Gpp") == 13337


def test_modis_npp_fill_masks_fill_codes(ee_session, first_value):
    from eetools.sensors.modis.masking import mask_modis_npp_fill

    # 32766 / 65534 are the MOD17 water fill codes.
    out = mask_modis_npp_fill(_npp_image(ee_session, 32766, 65534))
    assert first_value(out, "Npp") is None
    assert first_value(out, "Gpp") is None
    assert first_value(out, "Npp_QC") == 44


# --------------------------------------------------------------------------- #
# MOD44B vegetation continuous fields                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value, kept", [(0, True), (42, True), (100, True), (200, False), (253, False)]
)
def test_modis_vcf_fill_mask(ee_session, first_value, value, kept):
    from eetools.constants import MODIS_VCF_BANDS
    from eetools.sensors.modis.masking import mask_modis_vcf_fill

    img = (
        ee_session.Image.constant([value] * len(MODIS_VCF_BANDS))
        .toInt()
        .rename(MODIS_VCF_BANDS)
    )
    out = mask_modis_vcf_fill(img)
    assert (first_value(out, "Percent_Tree_Cover") is not None) is kept


# --------------------------------------------------------------------------- #
# Real-collection builders                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_get_modis_vi_collection_real(ee_session, small_aoi):
    from eetools.constants import MODIS_VI_BANDS
    from eetools.sensors.modis.preprocessing import get_modis_vi_collection

    col = get_modis_vi_collection(small_aoi, "2021-01-01", "2021-03-01")
    assert col.size().getInfo() > 0
    assert col.first().bandNames().getInfo() == MODIS_VI_BANDS
    ndvi_max = (
        col.select("NDVI")
        .max()
        .reduceRegion(ee_session.Reducer.max(), small_aoi, 250)
        .get("NDVI")
        .getInfo()
    )
    assert ndvi_max is None or -0.2 <= ndvi_max <= 1.0


@pytest.mark.slow
def test_get_modis_burned_area_collection_real(ee_session, small_aoi):
    from eetools.sensors.modis.preprocessing import get_modis_burned_area_collection

    col = get_modis_burned_area_collection(small_aoi, "2021-01-01", "2021-04-01")
    assert col.size().getInfo() == 3  # one image per month
    assert "burned" in col.first().bandNames().getInfo()


@pytest.mark.slow
def test_get_modis_npp_collection_real(ee_session, small_aoi):
    from eetools.sensors.modis.preprocessing import get_modis_npp_collection

    col = get_modis_npp_collection(small_aoi, "2020-01-01", "2022-01-01")
    assert col.size().getInfo() == 2  # annual
    npp = (
        col.first()
        .reduceRegion(ee_session.Reducer.mean(), small_aoi, 500)
        .get("Npp")
        .getInfo()
    )
    assert npp is None or -3.0 <= npp <= 3.27  # kg C / m^2 / yr after scaling


@pytest.mark.slow
def test_get_modis_vcf_collection_real(ee_session, small_aoi):
    from eetools.constants import MODIS_VCF_BANDS
    from eetools.sensors.modis.preprocessing import get_modis_vcf_collection

    col = get_modis_vcf_collection(small_aoi, "2020-01-01", "2022-01-01")
    assert col.size().getInfo() == 2
    assert col.first().bandNames().getInfo() == MODIS_VCF_BANDS


@pytest.mark.slow
def test_get_modis_npp_collection_rejects_out_of_range(ee_session, small_aoi):
    from eetools.sensors.modis.preprocessing import get_modis_npp_collection

    with pytest.raises(ValueError):
        get_modis_npp_collection(small_aoi, "1990-01-01", "1995-01-01")
