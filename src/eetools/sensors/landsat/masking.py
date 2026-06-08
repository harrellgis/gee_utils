import ee

from eetools.constants import L8_SR_COLLECTION, L8_CLOUD_FILTER


def mask_edges(image: ee.Image) -> ee.Image:
    """Return the image unchanged; included for interface compatibility with the Sentinel workflow.

    Args:
        image: Landsat 8 ee.Image.

    Returns:
        ee.Image identical to the input; Landsat 8 C2 L2 does not require the Sentinel-specific edge-mask logic.
    """
    return image


def get_l8_sr_cld_col(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> ee.ImageCollection:
    """Filter the Landsat 8 SR collection to the AOI, date range, and cloud-cover threshold.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.

    Returns:
        ee.ImageCollection of Landsat 8 SR images passing the CLOUD_COVER filter.
    """
    return (
        ee.ImageCollection(L8_SR_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUD_COVER", L8_CLOUD_FILTER))
        .map(mask_edges)
    )


def add_cld_shdw_mask(image: ee.Image) -> ee.Image:
    """Add a 'cloudmask' band derived from Landsat 8 C2 L2 QA_PIXEL and QA_RADSAT bands.

    Args:
        image: Landsat 8 C2 L2 ee.Image with QA_PIXEL and QA_RADSAT bands present.

    Returns:
        ee.Image with an additional 'cloudmask' band where 1 = cloud, shadow, cirrus, or saturated pixel (QA_PIXEL bits 0-4 and QA_RADSAT).
    """
    qa_pixel = image.select("QA_PIXEL")
    qa_mask = qa_pixel.bitwiseAnd(int("11111", 2)).neq(0)
    rad_sat_mask = image.select("QA_RADSAT").neq(0)
    cloudmask = qa_mask.Or(rad_sat_mask).rename("cloudmask")
    return image.addBands(cloudmask)


def apply_cld_shdw_mask(image: ee.Image) -> ee.Image:
    """Apply the inverse of the 'cloudmask' band to mask cloud and shadow pixels from all bands.

    Args:
        image: Landsat 8 ee.Image with a 'cloudmask' band added by add_cld_shdw_mask.

    Returns:
        ee.Image with cloud and shadow pixels masked out across all bands.
    """
    return image.updateMask(image.select("cloudmask").Not())


def build_cloudfree_l8sr_col(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> ee.ImageCollection:
    """Build a Landsat 8 SR collection with cloud and shadow masking fully applied.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.

    Returns:
        ee.ImageCollection of cloud- and shadow-masked Landsat 8 SR images.
    """
    return (
        get_l8_sr_cld_col(aoi, start_date, end_date)
        .map(add_cld_shdw_mask)
        .map(apply_cld_shdw_mask)
    )


def build_l8_non_water_mask(
    l8_collection: ee.ImageCollection,
    mndwi_thresh: float = 0.1,
    ndvi_thresh: float = 0.2,
    nir_thresh: float = 0.15,
) -> ee.Image:
    """Build a boolean non-water mask from a Landsat 8 collection median composite using spectral thresholds.

    Args:
        l8_collection: ee.ImageCollection with MNDWI, NDVI, and SR_B5 bands already computed.
        mndwi_thresh: MNDWI threshold above which a pixel is considered water (default 0.1).
        ndvi_thresh: NDVI threshold below which a pixel is considered water (default 0.2).
        nir_thresh: SR_B5 (NIR) reflectance threshold below which a pixel is considered water (default 0.15).

    Returns:
        ee.Image with a single 'non_water' band where 1 = land and 0 = water.
    """
    comp = l8_collection.median()
    water = (
        comp.select("MNDWI").gt(mndwi_thresh)
        .And(comp.select("NDVI").lt(ndvi_thresh))
        .And(comp.select("SR_B5").lt(nir_thresh))
    )
    return water.Not().rename("non_water")


def apply_water_mask(image: ee.Image, non_water_mask: ee.Image) -> ee.Image:
    """Apply a precomputed non-water mask to all bands of an image.

    Args:
        image: ee.Image to mask.
        non_water_mask: Single-band ee.Image where 1 = valid land pixel (output of build_l8_non_water_mask).

    Returns:
        ee.Image with water pixels masked out across all bands.
    """
    return image.updateMask(non_water_mask)
