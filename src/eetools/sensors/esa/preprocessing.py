import ee

from eetools.constants import ESA_BAND, ESA_WC_COLLECTION


def get_land_cover(aoi: ee.Geometry | None = None) -> ee.Image:
    """Load the raw ESA WorldCover categorical land-cover classification, optionally
    clipped to an AOI.

    Unlike :func:`get_esa_land_mask`, this returns the full categorical 'Map' band
    (class codes 10-100, see ``ESA_CLASS_MAP``) rather than a binary land mask.

    Args:
        aoi: Optional area of interest as ee.Geometry; if provided, the image is clipped to it.

    Returns:
        ee.Image with a single 'land_cover' band (categorical class codes), clipped to aoi when provided.
    """
    land_cover = (
        ee.ImageCollection(ESA_WC_COLLECTION)
        .mosaic()
        .select([ESA_BAND], ["land_cover"])
    )
    if aoi is not None:
        land_cover = land_cover.clip(aoi)
    return land_cover


def get_esa_land_mask() -> ee.Image:
    """Create a binary land mask from the ESA WorldCover classification, excluding built-up areas and water.

    Args:
        None.

    Returns:
        ee.Image with a single 'land_mask' band where 1 = land (excludes class 50 built-up and class 80 water).
    """
    esa_wc = ee.ImageCollection(ESA_WC_COLLECTION).mosaic().select(ESA_BAND)
    land_mask = esa_wc.neq(50).And(esa_wc.neq(80)).rename("land_mask")
    return land_mask


def apply_land_mask(image: ee.Image) -> ee.Image:
    """Apply the ESA WorldCover land mask to an image, preserving all image properties.

    Args:
        image: ee.Image to mask.

    Returns:
        ee.Image with built-up and water pixels masked out and all original properties preserved.
    """
    image = ee.Image(image)
    land_mask = get_esa_land_mask()
    return image.updateMask(land_mask).copyProperties(image, image.propertyNames())
