"""Tests for the NASA Black Marble VIIRS VNP46A2 nighttime-lights module.

The quality/snow masking is checked on synthetic constant images; the
real-collection builder is marked ``slow``.
"""

import pytest

pytestmark = pytest.mark.ee


def _ntl_image(ee, quality_flag, snow_flag=0, ntl=42.0, gap_filled=40.0):
    return ee.Image.cat(
        [
            ee.Image.constant(ntl).toFloat().rename("DNB_BRDF_Corrected_NTL"),
            ee.Image.constant(gap_filled)
            .toFloat()
            .rename("Gap_Filled_DNB_BRDF_Corrected_NTL"),
            ee.Image.constant(quality_flag).toInt().rename("Mandatory_Quality_Flag"),
            ee.Image.constant(snow_flag).toInt().rename("Snow_Flag"),
        ]
    )


# --------------------------------------------------------------------------- #
# QA masking                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "quality_flag, kept", [(0, True), (1, True), (2, False), (255, False)]
)
def test_viirs_ntl_qa_quality_flag(ee_session, first_value, quality_flag, kept):
    from eetools.sensors.viirs.masking import mask_viirs_ntl_qa

    out = mask_viirs_ntl_qa(_ntl_image(ee_session, quality_flag))
    assert (first_value(out, "DNB_BRDF_Corrected_NTL") is not None) is kept


def test_viirs_ntl_qa_persistent_only(ee_session, first_value):
    from eetools.sensors.viirs.masking import mask_viirs_ntl_qa

    # max_quality_flag=0 drops ephemeral lights (flag 1).
    out = mask_viirs_ntl_qa(_ntl_image(ee_session, 1), max_quality_flag=0)
    assert first_value(out, "DNB_BRDF_Corrected_NTL") is None


def test_viirs_ntl_qa_snow(ee_session, first_value):
    from eetools.sensors.viirs.masking import mask_viirs_ntl_qa

    snowy = _ntl_image(ee_session, 0, snow_flag=1)
    assert first_value(mask_viirs_ntl_qa(snowy), "DNB_BRDF_Corrected_NTL") is None
    kept = mask_viirs_ntl_qa(snowy, mask_snow=False)
    assert first_value(kept, "DNB_BRDF_Corrected_NTL") == pytest.approx(42.0)


def test_viirs_ntl_qa_leaves_gap_filled_band(ee_session, first_value):
    from eetools.sensors.viirs.masking import mask_viirs_ntl_qa

    out = mask_viirs_ntl_qa(_ntl_image(ee_session, 2))
    assert first_value(out, "DNB_BRDF_Corrected_NTL") is None
    assert first_value(out, "Gap_Filled_DNB_BRDF_Corrected_NTL") == pytest.approx(40.0)


# --------------------------------------------------------------------------- #
# Real-collection builder                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_get_viirs_ntl_collection_real(ee_session):
    from eetools.constants import VIIRS_NTL_BANDS
    from eetools.sensors.viirs.preprocessing import get_viirs_ntl_collection

    nairobi = ee_session.Geometry.Rectangle([36.78, -1.32, 36.86, -1.26])
    col = get_viirs_ntl_collection(nairobi, "2024-06-04", "2024-06-11")
    assert col.size().getInfo() > 0
    first = ee_session.Image(col.first())
    assert first.bandNames().getInfo() == VIIRS_NTL_BANDS
    radiance = (
        col.select("Gap_Filled_DNB_BRDF_Corrected_NTL")
        .mean()
        .reduceRegion(ee_session.Reducer.mean(), nairobi, 500)
        .get("Gap_Filled_DNB_BRDF_Corrected_NTL")
        .getInfo()
    )
    assert radiance is not None and radiance > 1  # an urban core is lit


@pytest.mark.slow
def test_get_viirs_ntl_collection_rejects_pre_mission(ee_session, small_aoi):
    from eetools.sensors.viirs.preprocessing import get_viirs_ntl_collection

    with pytest.raises(ValueError):
        get_viirs_ntl_collection(small_aoi, "2010-01-01", "2011-01-01")
