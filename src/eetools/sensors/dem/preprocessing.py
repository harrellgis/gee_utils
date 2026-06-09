import ee

from eetools.constants import (
    COPERNICUS_DEM_BAND,
    COPERNICUS_DEM_COLLECTION,
    ELEVATION_BAND,
)


def get_copernicus_dem(aoi: ee.Geometry | None = None) -> ee.Image:
    """Load the Copernicus DEM GLO-30 as a single elevation image in its native
    projection.

    This is the canonical elevation source for the package. The GLO30 product is a
    tiled ``ee.ImageCollection``; the tiles are mosaicked and the mosaic is pinned to
    the dataset's native projection so downstream terrain analysis runs at the native
    30 m scale and CRS (a flat mosaic otherwise defaults to a 1-degree projection,
    which would make slope/aspect meaningless). The source ``"DEM"`` band is renamed to
    ``"elevation"`` (metres above the EGM2008 geoid) so it can be passed straight to
    ``ee.Terrain``.

    Args:
        aoi: Optional area of interest as ee.Geometry; if provided, the DEM is clipped to it.

    Returns:
        ee.Image with a single 'elevation' band (metres), set to the dataset's native projection.
    """
    collection = ee.ImageCollection(COPERNICUS_DEM_COLLECTION)
    native_projection = collection.first().projection()
    dem = (
        collection.mosaic()
        .setDefaultProjection(native_projection)
        .select([COPERNICUS_DEM_BAND], [ELEVATION_BAND])
    )
    if aoi is not None:
        dem = dem.clip(aoi)
    return dem


def get_terrain(
    aoi: ee.Geometry | None = None,
    elevation: ee.Image | None = None,
    add_elevation: bool = True,
) -> ee.Image:
    """Derive terrain products (slope, aspect, hillshade) from an elevation image.

    Wraps ``ee.Terrain.products``, which adds 'slope' and 'aspect' bands (degrees) and
    an unsigned-byte 'hillshade' band, copying the source elevation band through. By
    default the elevation comes from the canonical Copernicus DEM GLO-30
    (:func:`get_copernicus_dem`); pass ``elevation`` to derive terrain from a different
    DEM. The elevation image must have a single band, or a band named 'elevation'.

    Args:
        aoi: Optional area of interest as ee.Geometry; if provided, terrain is clipped to it (and the default DEM is loaded clipped to it).
        elevation: Optional elevation ee.Image (metres) to use instead of the Copernicus DEM; clipped to aoi when both are provided.
        add_elevation: If True (default), retain the 'elevation' band alongside slope/aspect/hillshade; if False, return only slope/aspect/hillshade.

    Returns:
        ee.Image with bands 'slope', 'aspect', 'hillshade' (and 'elevation' when add_elevation is True), in the elevation image's projection.
    """
    if elevation is None:
        elevation = get_copernicus_dem(aoi=aoi)
    elif aoi is not None:
        elevation = ee.Image(elevation).clip(aoi)

    terrain = ee.Terrain.products(elevation)

    if not add_elevation:
        terrain = terrain.select(["slope", "aspect", "hillshade"])

    return terrain
