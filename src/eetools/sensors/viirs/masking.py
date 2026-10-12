import ee

from eetools.constants import VIIRS_NTL_BAND, VIIRS_NTL_MAX_QUALITY_FLAG


def mask_viirs_ntl_qa(
    image: ee.Image,
    max_quality_flag: int = VIIRS_NTL_MAX_QUALITY_FLAG,
    mask_snow: bool = True,
) -> ee.Image:
    """Mask the VNP46A2 BRDF-corrected NTL band to high-quality, snow-free retrievals.

    Only the DNB_BRDF_Corrected_NTL band is masked; the gap-filled band is left
    untouched because it is designed to cover poor-quality/cloudy pixels.

    Args:
        image: VNP46A2 ee.Image containing DNB_BRDF_Corrected_NTL, Mandatory_Quality_Flag, and Snow_Flag bands.
        max_quality_flag: Highest Mandatory_Quality_Flag kept: 0 high-quality persistent, 1 high-quality ephemeral (e.g. fires, flares), 2 poor quality (default VIIRS_NTL_MAX_QUALITY_FLAG = 1). Fill (255) is always dropped.
        mask_snow: If True, also masks pixels where Snow_Flag marks snow/ice (default True).

    Returns:
        ee.Image with DNB_BRDF_Corrected_NTL masked where the quality flag exceeds max_quality_flag (or snow is present); other bands are unchanged.
    """
    keep = image.select("Mandatory_Quality_Flag").lte(max_quality_flag)
    if mask_snow:
        keep = keep.And(image.select("Snow_Flag").eq(0))
    ntl = image.select(VIIRS_NTL_BAND).updateMask(keep)
    return image.addBands(ntl, overwrite=True)
