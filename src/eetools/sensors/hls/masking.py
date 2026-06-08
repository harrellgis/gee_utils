import ee

from eetools.constants import (
    HLS_CLOUD_FILTER,
    HLS_L30_COLLECTION,
    HLS_MASK_ADJACENT,
    HLS_MASK_HIGH_AEROSOL,
    HLS_MASK_MODERATE_AEROSOL,
    HLS_MASK_SNOW,
    HLS_MASK_WATER_IN_QA,
    HLS_S30_COLLECTION,
)
from eetools.sensors import masking as _masking


def get_hls_l30_col(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> ee.ImageCollection:
    """Filter the HLSL30 collection to the AOI, date range, and cloud-coverage
    threshold.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.

    Returns:
        ee.ImageCollection of HLSL30 images passing the CLOUD_COVERAGE filter.
    """
    return (
        ee.ImageCollection(HLS_L30_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUD_COVERAGE", HLS_CLOUD_FILTER))
    )


def get_hls_s30_col(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> ee.ImageCollection:
    """Filter the HLSS30 collection to the AOI, date range, and cloud-coverage
    threshold.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.

    Returns:
        ee.ImageCollection of HLSS30 images passing the CLOUD_COVERAGE filter.
    """
    return (
        ee.ImageCollection(HLS_S30_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUD_COVERAGE", HLS_CLOUD_FILTER))
    )


def add_fmask_cloud_mask(image: ee.Image) -> ee.Image:
    """Add a 'cloudmask' band derived from HLS Fmask bits for cloud, shadow, and optionally adjacent pixels, snow, water, and aerosol.

    Args:
        image: HLS ee.Image with an 'Fmask' band. Fmask bits used: bit 1 (cloud), bit 2 (adjacent to cloud/shadow), bit 3 (cloud shadow), bit 4 (snow/ice), bit 5 (water), bits 6-7 (aerosol level).

    Returns:
        ee.Image with an additional 'cloudmask' band where 1 = pixel to exclude, controlled by module-level HLS_MASK_* constants.
    """
    fmask = image.select("Fmask")

    is_cloud = fmask.bitwiseAnd(1 << 1).neq(0)
    is_adjacent = fmask.bitwiseAnd(1 << 2).neq(0)
    is_shadow = fmask.bitwiseAnd(1 << 3).neq(0)
    is_snow = fmask.bitwiseAnd(1 << 4).neq(0)
    is_water = fmask.bitwiseAnd(1 << 5).neq(0)

    aerosol = fmask.rightShift(6).bitwiseAnd(3)
    is_moderate_aerosol = aerosol.eq(2)
    is_high_aerosol = aerosol.eq(3)

    mask = is_cloud.Or(is_shadow)

    if HLS_MASK_ADJACENT:
        mask = mask.Or(is_adjacent)
    if HLS_MASK_SNOW:
        mask = mask.Or(is_snow)
    if HLS_MASK_WATER_IN_QA:
        mask = mask.Or(is_water)
    if HLS_MASK_MODERATE_AEROSOL:
        mask = mask.Or(is_moderate_aerosol)
    if HLS_MASK_HIGH_AEROSOL:
        mask = mask.Or(is_high_aerosol)

    return image.addBands(mask.rename("cloudmask"))


def apply_cld_shdw_mask(image: ee.Image) -> ee.Image:
    """Apply the inverse of the HLS 'cloudmask' band to mask excluded pixels from all
    bands.

    Args:
        image: HLS ee.Image with a 'cloudmask' band added by add_fmask_cloud_mask.

    Returns:
        ee.Image with cloud, shadow, and any additionally flagged pixels masked out across all bands.
    """
    return _masking.apply_cloud_mask(image)


def build_cloudfree_hls_l30_col(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> ee.ImageCollection:
    """Build an HLSL30 collection with Fmask-based QA masking fully applied.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.

    Returns:
        ee.ImageCollection of cloud- and shadow-masked HLSL30 images.
    """
    return (
        get_hls_l30_col(aoi, start_date, end_date)
        .map(add_fmask_cloud_mask)
        .map(apply_cld_shdw_mask)
    )


def build_cloudfree_hls_s30_col(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> ee.ImageCollection:
    """Build an HLSS30 collection with Fmask-based QA masking fully applied.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.

    Returns:
        ee.ImageCollection of cloud- and shadow-masked HLSS30 images.
    """
    return (
        get_hls_s30_col(aoi, start_date, end_date)
        .map(add_fmask_cloud_mask)
        .map(apply_cld_shdw_mask)
    )


def build_hls_non_water_mask(
    hls_collection: ee.ImageCollection,
    mndwi_thresh: float = 0.1,
    ndvi_thresh: float = 0.2,
    nir_thresh: float = 0.15,
) -> ee.Image:
    """Build a boolean non-water mask from an HLS collection median composite using spectral thresholds.

    Args:
        hls_collection: ee.ImageCollection with MNDWI, NDVI, and NIR bands already computed.
        mndwi_thresh: MNDWI threshold above which a pixel is considered water (default 0.1).
        ndvi_thresh: NDVI threshold below which a pixel is considered water (default 0.2).
        nir_thresh: NIR reflectance threshold below which a pixel is considered water (default 0.15).

    Returns:
        ee.Image with a single 'non_water' band where 1 = land and 0 = water.
    """
    return _masking.build_non_water_mask(
        hls_collection,
        nir_band="NIR",
        mndwi_thresh=mndwi_thresh,
        ndvi_thresh=ndvi_thresh,
        nir_thresh=nir_thresh,
    )


def apply_water_mask(image: ee.Image, non_water_mask: ee.Image) -> ee.Image:
    """Apply a precomputed non-water mask to all bands of an image.

    Args:
        image: ee.Image to mask.
        non_water_mask: Single-band ee.Image where 1 = valid land pixel (output of build_hls_non_water_mask).

    Returns:
        ee.Image with water pixels masked out across all bands.
    """
    return _masking.apply_water_mask(image, non_water_mask)
