import ee

from eetools.constants import (
    COAST_DISTANCE_MAX_M,
    COAST_DISTANCE_SCALE_M,
    COPERNICUS_DEM_COLLECTION,
    COPERNICUS_WBM_BAND,
    COPERNICUS_WBM_OCEAN,
    GMW_2020_COLLECTION,
    GMW_BAND,
    MERIT_HAND_BAND,
    MERIT_HYDRO_IMAGE,
)


def get_hand(aoi: ee.Geometry | None = None) -> ee.Image:
    """Load MERIT Hydro height above nearest drainage (HAND, ~90 m).

    Args:
        aoi: Optional ee.Geometry to clip to.

    Returns:
        ee.Image with a single 'hand' band (metres).
    """
    image = ee.Image(MERIT_HYDRO_IMAGE).select([MERIT_HAND_BAND], ["hand"])
    if aoi is not None:
        image = image.clip(aoi)
    return image


def get_ocean_mask() -> ee.Image:
    """Return a 0/1 ocean mask from the Copernicus GLO-30 water-body mask (WBM == 1).

    The GLO-30 DEM is not masked over the ocean, so its WBM band is used instead of
    the DEM footprint. Lakes (2) and rivers (3) are not ocean.

    Args:
        None.

    Returns:
        ee.Image with a single 'ocean' band (1 ocean, 0 otherwise), native projection.
    """
    col = ee.ImageCollection(COPERNICUS_DEM_COLLECTION)
    wbm = col.select(COPERNICUS_WBM_BAND).mosaic()
    return (
        wbm.eq(COPERNICUS_WBM_OCEAN)
        .unmask(0)
        .rename("ocean")
        .setDefaultProjection(col.first().projection())
    )


def get_coast_distance(
    region: ee.Geometry,
    crs: str,
    scale_m: float = COAST_DISTANCE_SCALE_M,
    max_distance_m: float = COAST_DISTANCE_MAX_M,
) -> ee.Image:
    """Distance to the ocean (metres), computed on a coarse grid.

    The ocean mask is resampled to ``scale_m`` (from its pyramid) and a
    Euclidean distance is computed up to ``max_distance_m``; beyond it the value is
    ``max_distance_m``.

    Args:
        region: Area the distances are needed for; the ocean mask is clipped to it
            buffered by max_distance_m so the reprojection stays bounded.
        crs: CRS string of the output grid (use the focal UTM zone).
        scale_m: Output pixel size in metres (default 1000).
        max_distance_m: Search radius in metres (default 50 km).

    Returns:
        ee.Image with a single 'coast_dist' band (metres), reprojected to crs/scale_m.
    """
    proj = ee.Projection(crs).atScale(scale_m)
    bounds = ee.Geometry(region).buffer(max_distance_m, 1000).bounds(1000)
    # The pyramided WBM mask is accurate enough for a 1 km coastal-distance proxy and
    # avoids aggregating ~1,000 native 30 m pixels per output pixel.
    ocean = get_ocean_mask().clip(bounds).reproject(proj).gt(0)
    dist = (
        ocean.selfMask()
        .fastDistanceTransform(int(max_distance_m / scale_m) + 1)
        .sqrt()
        .multiply(scale_m)
        .unmask(max_distance_m)
        .min(max_distance_m)
        .reproject(proj)
        .clip(bounds)
    )
    return dist.rename("coast_dist").toFloat()


def get_mangrove_extent_2020() -> ee.Image:
    """Load the Global Mangrove Watch 2020 mangrove extent (tiles mosaicked).

    Args:
        None.

    Returns:
        ee.Image with a single 'mangrove' band (1 where mangrove, masked elsewhere).
    """
    col = ee.ImageCollection(GMW_2020_COLLECTION)
    return (
        col.mosaic()
        .select([GMW_BAND], ["mangrove"])
        .setDefaultProjection(col.first().projection())
    )
