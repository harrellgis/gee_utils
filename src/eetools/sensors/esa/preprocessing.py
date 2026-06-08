import ee

from eetools.constants import ESA_BAND, ESA_WC_COLLECTION


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
