import ee

from eetools.constants import DSWX_WTR_BAND


def mask_invalid_classes(
    image: ee.Image,
    valid_max: int,
    wtr_band: str = DSWX_WTR_BAND,
) -> ee.Image:
    """Mask DSWx pixels whose WTR class is an invalid/no-data mask code.

    DSWx WTR codes at or above a product-specific threshold are sensor masks, not water
    classes, and differ between products: HLS uses 252 (snow) / 253 (cloud) / 254
    (ocean) -> ``valid_max`` 252; S1 uses 250 (HAND) / 251 (layover-shadow) / 254
    (ocean) -> ``valid_max`` 250. (The GEE catalog's S1 sample code wrongly reuses the
    HLS 252 threshold.) Pixels with ``WTR >= valid_max`` are masked across all bands.

    Args:
        image: DSWx ee.Image carrying the WTR classification band.
        valid_max: First invalid class code; pixels with WTR >= valid_max are masked (DSWX_HLS_VALID_MAX for HLS, DSWX_S1_VALID_MAX for S1).
        wtr_band: Name of the water-classification band (default DSWX_WTR_BAND).

    Returns:
        ee.Image with invalid-class pixels masked across all bands, preserving image properties.
    """
    valid = image.select(wtr_band).lt(valid_max)
    return image.updateMask(valid)
