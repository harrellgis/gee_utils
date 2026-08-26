import ee

from eetools.constants import META_CANOPY_HEIGHT_BAND, META_CANOPY_HEIGHT_COLLECTION


def get_canopy_height(aoi: ee.Geometry | None = None) -> ee.Image:
    """Load the Meta/WRI 1m global canopy height mosaic, optionally clipped to an AOI.

    The source asset is a tiled ee.ImageCollection; tiles are mosaicked to a single
    wall-to-wall image. This is a fused 2009-2020 baseline (~80% 2018-2020 imagery),
    not a dated annual layer -- do not difference it against another epoch to infer
    change.

    Args:
        aoi: Optional area of interest as ee.Geometry; if provided, the image is clipped to it.

    Returns:
        ee.Image with a single 'canopy_height' band (metres), clipped to aoi when provided.
    """
    canopy = (
        ee.ImageCollection(META_CANOPY_HEIGHT_COLLECTION)
        .mosaic()
        .select([META_CANOPY_HEIGHT_BAND], ["canopy_height"])
    )
    if aoi is not None:
        canopy = canopy.clip(aoi)
    return canopy
