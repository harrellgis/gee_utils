import ee

from eetools.constants import (
    RESOLVE_BIOME_FIELD,
    RESOLVE_ECO_ID_FIELD,
    RESOLVE_ECOREGIONS_COLLECTION,
)


def get_ecoregion(aoi: ee.Geometry | None = None) -> ee.FeatureCollection:
    """Load RESOLVE Ecoregions 2017 polygons, optionally filtered to an AOI.

    Polygons are split, so match ecoregions on ``ECO_ID``, never on the polygon.

    Args:
        aoi: Optional ee.Geometry; only intersecting polygons are returned.

    Returns:
        ee.FeatureCollection of ecoregion polygons (ECO_ID, ECO_NAME, BIOME_NUM, ...).
    """
    fc = ee.FeatureCollection(RESOLVE_ECOREGIONS_COLLECTION)
    if aoi is not None:
        fc = fc.filterBounds(aoi)
    return fc


def get_ecoregion_image(aoi: ee.Geometry | None = None) -> ee.Image:
    """Rasterise RESOLVE ecoregion and biome codes.

    Args:
        aoi: Optional ee.Geometry; restricts the painted polygons (and clips).

    Returns:
        ee.Image with integer bands 'eco_id' and 'biome_num' (masked where no
        ecoregion polygon exists). The image has no fixed projection; reduce it at an
        explicit scale.
    """
    fc = get_ecoregion(aoi)
    eco = ee.Image().int().paint(fc, RESOLVE_ECO_ID_FIELD).rename("eco_id")
    biome = ee.Image().int().paint(fc, RESOLVE_BIOME_FIELD).rename("biome_num")
    image = eco.addBands(biome)
    if aoi is not None:
        image = image.clip(aoi)
    return image
