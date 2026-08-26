import ee

from eetools.constants import ISDA_CARBON_COLLECTION, ISDA_CARBON_TOPSOIL_BAND


def get_soil_carbon(aoi: ee.Geometry | None = None) -> ee.Image:
    """Load the iSDA Africa topsoil (0-20cm) total soil carbon layer, optionally
    clipped to an AOI.

    The source asset is a single static ee.Image (Africa-only, 30m nominal
    resolution); this selects the topsoil mean band and renames it for downstream
    use.

    Args:
        aoi: Optional area of interest as ee.Geometry; if provided, the image is clipped to it.

    Returns:
        ee.Image with a single 'soil_carbon' band, clipped to aoi when provided.
    """
    carbon = ee.Image(ISDA_CARBON_COLLECTION).select(
        [ISDA_CARBON_TOPSOIL_BAND], ["soil_carbon"]
    )
    if aoi is not None:
        carbon = carbon.clip(aoi)
    return carbon
