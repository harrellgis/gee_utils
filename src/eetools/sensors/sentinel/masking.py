import ee

from eetools.constants import (
    BUFFER_M,
    CLD_PRB_THRESH,
    CLD_PRJ_DIST_KM,
    CLOUD_FILTER,
    DDT_SCALE_M,
    ERODE_RADIUS_M,
    MORPH_SCALE_M,
    NIR_DRK_THRESH,
    S2_CLOUD_PROB_COLLECTION,
    S2_SR_COLLECTION,
)
from eetools.sensors import masking as _masking


def mask_edges(image: ee.Image) -> ee.Image:
    """Mask noisy edge pixels using the Sentinel-2 20 m (B8A) and 60 m (B9) band masks.

    Args:
        image: Sentinel-2 ee.Image with bands B8A and B9 present.

    Returns:
        ee.Image with edge pixels masked out using the intersection of B8A and B9 masks.
    """
    return image.updateMask(image.select("B8A").mask()).updateMask(
        image.select("B9").mask()
    )


def get_s2_sr_cld_col(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> ee.ImageCollection:
    """Build a joined Sentinel-2 SR and s2cloudless collection filtered to the AOI and
    date range.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.

    Returns:
        ee.ImageCollection of Sentinel-2 SR images with the corresponding s2cloudless probability image saved as the 's2cloudless' property on each image.
    """
    s2_sr_col = (
        ee.ImageCollection(S2_SR_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CLOUD_FILTER))
        .map(mask_edges)
    )

    s2_cloudless_col = (
        ee.ImageCollection(S2_CLOUD_PROB_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    return ee.ImageCollection(
        ee.Join.saveFirst("s2cloudless").apply(
            primary=s2_sr_col,
            secondary=s2_cloudless_col,
            condition=ee.Filter.equals(
                leftField="system:index", rightField="system:index"
            ),
        )
    )


def add_cld_shdw_mask(image: ee.Image) -> ee.Image:
    """Add a 'cloudmask' band derived from s2cloudless probability, dark NIR pixels, and directional shadow projection.

    Args:
        image: Sentinel-2 ee.Image with the 's2cloudless' property set (output of get_s2_sr_cld_col).

    Returns:
        ee.Image with an additional 'cloudmask' band where 1 = cloud or shadow pixel.
    """
    cld_prb = ee.Image(image.get("s2cloudless")).select("probability")
    is_cloud = cld_prb.gt(CLD_PRB_THRESH)

    not_water = image.select("SCL").neq(6)
    dark_pixels = image.select("B8").multiply(0.0001).lt(NIR_DRK_THRESH).And(not_water)

    shadow_azimuth_deg = ee.Number(90).subtract(
        ee.Number(image.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
    )
    max_dist_px = ee.Number(CLD_PRJ_DIST_KM).multiply(1000).divide(DDT_SCALE_M).int()

    clouds_for_ddt = is_cloud.reproject(
        crs=image.select(0).projection(), scale=DDT_SCALE_M
    )
    cld_proj = (
        clouds_for_ddt.directionalDistanceTransform(shadow_azimuth_deg, max_dist_px)
        .select("distance")
        .mask()
    )

    is_shadow = cld_proj.And(dark_pixels)

    opened = (
        is_cloud.Or(is_shadow)
        .reproject(crs=image.select(0).projection(), scale=MORPH_SCALE_M)
        .focalMin(ERODE_RADIUS_M, units="meters")
        .focalMax(BUFFER_M, units="meters")
    )

    return image.addBands(opened.rename("cloudmask"))


def apply_cld_shdw_mask(image: ee.Image) -> ee.Image:
    """Apply the inverse of the 'cloudmask' band to mask cloud and shadow pixels from
    all bands.

    Args:
        image: ee.Image with a 'cloudmask' band added by add_cld_shdw_mask.

    Returns:
        ee.Image with cloud and shadow pixels masked out across all bands.
    """
    return _masking.apply_cloud_mask(image)


def build_cloudfree_s2sr_col(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> ee.ImageCollection:
    """Build a Sentinel-2 SR collection with cloud and shadow masking fully applied.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.

    Returns:
        ee.ImageCollection of cloud- and shadow-masked Sentinel-2 SR images.
    """
    return (
        get_s2_sr_cld_col(aoi, start_date, end_date)
        .map(add_cld_shdw_mask)
        .map(apply_cld_shdw_mask)
    )


def build_s2_non_water_mask(
    s2_collection: ee.ImageCollection,
    mndwi_thresh: float = 0.1,
    ndvi_thresh: float = 0.2,
    nir_thresh: float = 0.15,
) -> ee.Image:
    """Build a boolean non-water mask from a Sentinel-2 collection median composite using spectral thresholds.

    Args:
        s2_collection: ee.ImageCollection with MNDWI, NDVI, and B8 bands already computed.
        mndwi_thresh: MNDWI threshold above which a pixel is considered water (default 0.1).
        ndvi_thresh: NDVI threshold below which a pixel is considered water (default 0.2).
        nir_thresh: B8 reflectance threshold below which a pixel is considered water (default 0.15).

    Returns:
        ee.Image with a single 'non_water' band where 1 = land and 0 = water.
    """
    return _masking.build_non_water_mask(
        s2_collection,
        nir_band="B8",
        mndwi_thresh=mndwi_thresh,
        ndvi_thresh=ndvi_thresh,
        nir_thresh=nir_thresh,
    )


def apply_water_mask(image: ee.Image, non_water_mask: ee.Image) -> ee.Image:
    """Apply a precomputed non-water mask to all bands of an image.

    Args:
        image: ee.Image to mask.
        non_water_mask: Single-band ee.Image where 1 = valid land pixel (output of build_s2_non_water_mask).

    Returns:
        ee.Image with water pixels masked out across all bands.
    """
    return _masking.apply_water_mask(image, non_water_mask)
