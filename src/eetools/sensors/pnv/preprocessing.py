import ee

from eetools.constants import PNV_BIOME_BAND, PNV_BIOME_IMAGE


def get_pnv_biome(aoi: ee.Geometry | None = None) -> ee.Image:
    """Load the OpenLandMap potential natural vegetation (PNV) biome map (~1 km).

    PNV reflects the reference ecosystem rather than the current altered state, so it
    is the fallback national-class proxy (with the RESOLVE ``ECO_ID``) outside VECEA
    coverage. Class codes/names are in the image properties
    ``biome_type_class_values`` / ``biome_type_class_names``.

    Args:
        aoi: Optional ee.Geometry to clip to.

    Returns:
        ee.Image with a single integer 'pnv_biome' band.
    """
    image = ee.Image(PNV_BIOME_IMAGE).select([PNV_BIOME_BAND], ["pnv_biome"])
    if aoi is not None:
        image = image.clip(aoi)
    return image
