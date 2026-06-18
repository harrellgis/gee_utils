"""Tests for the OPERA DSWx masking and preprocessing modules.

Invalid-class masking is checked on synthetic constant images — the key behaviour
is that the valid-class threshold differs between the HLS and S1 products. The
real-collection builders are marked slow.
"""

import pytest

pytestmark = pytest.mark.ee


def _wtr_image(ee, wtr_value):
    """Two-band DSWx-like image: a WTR class plus a binary-water band."""
    return ee.Image.cat(
        [
            ee.Image.constant(wtr_value).toInt().rename("WTR_Water_classification"),
            ee.Image.constant(1).toInt().rename("BWTR_Binary_water"),
        ]
    )


def test_dswx_mask_invalid_drops_mask_classes(ee_session, first_value):
    from eetools.constants import DSWX_HLS_VALID_MAX
    from eetools.sensors.dswx.masking import mask_invalid_classes

    kept = _wtr_image(ee_session, 1)  # open water
    dropped = _wtr_image(ee_session, 253)  # HLS cloud mask

    # Masking applies across all bands, so check the companion band too.
    assert (
        first_value(mask_invalid_classes(kept, DSWX_HLS_VALID_MAX), "BWTR_Binary_water")
        == 1
    )
    assert (
        first_value(
            mask_invalid_classes(dropped, DSWX_HLS_VALID_MAX), "BWTR_Binary_water"
        )
        is None
    )


def test_dswx_threshold_differs_between_products(ee_session, first_value):
    from eetools.constants import DSWX_HLS_VALID_MAX, DSWX_S1_VALID_MAX
    from eetools.sensors.dswx.masking import mask_invalid_classes

    # WTR 251 is an S1 layover/shadow mask (invalid for S1, valid_max 250) but is
    # below the HLS valid_max (252), so it must survive HLS masking and not S1.
    img = _wtr_image(ee_session, 251)
    assert (
        first_value(
            mask_invalid_classes(img, DSWX_S1_VALID_MAX), "WTR_Water_classification"
        )
        is None
    )
    assert (
        first_value(
            mask_invalid_classes(img, DSWX_HLS_VALID_MAX), "WTR_Water_classification"
        )
        == 251
    )


@pytest.mark.slow
def test_get_dswx_hls_collection_real(ee_session, small_aoi):
    from eetools.sensors.dswx.preprocessing import get_dswx_hls_collection

    col = get_dswx_hls_collection(
        small_aoi,
        ee_session.Date("2025-01-01"),
        ee_session.Date("2025-04-01"),
    )
    assert isinstance(col, ee_session.ImageCollection)
    if col.size().getInfo() == 0:
        pytest.skip("no DSWx-HLS scenes for the AOI/window")
    assert "WTR_Water_classification" in col.first().bandNames().getInfo()


@pytest.mark.slow
def test_get_dswx_s1_collection_real(ee_session, small_aoi):
    from eetools.sensors.dswx.preprocessing import get_dswx_s1_collection

    col = get_dswx_s1_collection(
        small_aoi,
        ee_session.Date("2025-01-01"),
        ee_session.Date("2025-04-01"),
    )
    assert isinstance(col, ee_session.ImageCollection)
    if col.size().getInfo() == 0:
        pytest.skip("no DSWx-S1 scenes for the AOI/window")
    assert "WTR_Water_classification" in col.first().bandNames().getInfo()
