"""Sentinel-2 SR cloud, shadow and water masking.

Two pixel cloud masks are available, chosen with ``cloud_mask=``:

* ``"cloud_score_plus"`` (default): Google Cloud Score+ (``cs`` / ``cs_cdf``, 0-1,
  higher = clearer). One score covers cloud, shadow, haze and thin cirrus, so no
  separate shadow step is needed. A pixel is kept where the score is >= the threshold.
* ``"s2cloudless"``: s2cloudless probability > ``CLD_PRB_THRESH`` plus a dark-NIR
  shadow projection and morphological opening (the eetools default before 0.11.0).

Both write a ``cloudmask`` band (1 = exclude) that ``apply_cld_shdw_mask`` applies.
"""

import ee

from eetools.constants import (
    BUFFER_M,
    CLD_PRB_THRESH,
    CLD_PRJ_DIST_KM,
    CLOUD_FILTER,
    CS_PLUS_BAND,
    CS_PLUS_BANDS,
    CS_PLUS_THRESH,
    DDT_SCALE_M,
    ERODE_RADIUS_M,
    MORPH_SCALE_M,
    NIR_DRK_THRESH,
    S2_CLOUD_MASK_METHODS,
    S2_CLOUD_PROB_COLLECTION,
    S2_CLOUD_SCORE_PLUS_COLLECTION,
    S2_DEFAULT_CLOUD_MASK,
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


def validate_s2_cloud_mask_args(
    cloud_mask: str,
    cs_band: str = CS_PLUS_BAND,
    cs_threshold: float = CS_PLUS_THRESH,
) -> None:
    """Check the Sentinel-2 cloud-mask options before any graph is built.

    Args:
        cloud_mask: Pixel cloud-mask method, one of ``S2_CLOUD_MASK_METHODS``
            ('cloud_score_plus' or 's2cloudless').
        cs_band: Cloud Score+ band, one of ``CS_PLUS_BANDS`` ('cs' or 'cs_cdf').
            Only checked when ``cloud_mask`` is 'cloud_score_plus'.
        cs_threshold: Cloud Score+ clear threshold in [0, 1]. Only checked when
            ``cloud_mask`` is 'cloud_score_plus'.

    Returns:
        None.

    Raises:
        ValueError: If ``cloud_mask`` or ``cs_band`` is unknown, or ``cs_threshold``
            is outside [0, 1].
    """
    if cloud_mask not in S2_CLOUD_MASK_METHODS:
        raise ValueError(
            f"Unknown cloud_mask '{cloud_mask}'. Expected one of "
            f"{list(S2_CLOUD_MASK_METHODS)}."
        )
    if cloud_mask != "cloud_score_plus":
        return
    if cs_band not in CS_PLUS_BANDS:
        raise ValueError(
            f"Unknown Cloud Score+ band '{cs_band}'. Expected one of "
            f"{list(CS_PLUS_BANDS)}."
        )
    if not 0 <= cs_threshold <= 1:
        raise ValueError(
            f"cs_threshold must be in [0, 1] (Cloud Score+ scores are 0-1), "
            f"got {cs_threshold}."
        )


def _get_s2_sr_base_col(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    cloud_filter: float | None,
) -> ee.ImageCollection:
    """S2 SR scenes over the AOI and dates, scene-filtered and edge-masked."""
    s2_sr_col = (
        ee.ImageCollection(S2_SR_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    if cloud_filter is not None:
        s2_sr_col = s2_sr_col.filter(
            ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_filter)
        )
    return s2_sr_col.map(mask_edges)


def _join_on_index(
    primary: ee.ImageCollection,
    secondary_id: str,
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    property_name: str,
) -> ee.ImageCollection:
    """Save the matching secondary image (same system:index) as a property.

    ``Join.saveFirst`` drops primary scenes with no match, so a scene whose cloud
    product is missing is left out rather than passed through unmasked.
    """
    secondary = (
        ee.ImageCollection(secondary_id)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    return ee.ImageCollection(
        ee.Join.saveFirst(property_name).apply(
            primary=primary,
            secondary=secondary,
            condition=ee.Filter.equals(
                leftField="system:index", rightField="system:index"
            ),
        )
    )


def get_s2_sr_cld_col(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    cloud_filter: float | None = CLOUD_FILTER,
) -> ee.ImageCollection:
    """Build a joined Sentinel-2 SR and s2cloudless collection filtered to the AOI and
    date range.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.
        cloud_filter: Maximum scene CLOUDY_PIXEL_PERCENTAGE (default CLOUD_FILTER, 50). None keeps every scene; use it for per-scene work on small sites, where a tile that is mostly cloudy can still be clear over the site.

    Returns:
        ee.ImageCollection of Sentinel-2 SR images with the corresponding s2cloudless probability image saved as the 's2cloudless' property on each image.
    """
    return _join_on_index(
        _get_s2_sr_base_col(aoi, start_date, end_date, cloud_filter),
        S2_CLOUD_PROB_COLLECTION,
        aoi,
        start_date,
        end_date,
        "s2cloudless",
    )


def get_s2_sr_cs_col(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    cloud_filter: float | None = CLOUD_FILTER,
) -> ee.ImageCollection:
    """Build a joined Sentinel-2 SR and Cloud Score+ collection filtered to the AOI and
    date range.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.
        cloud_filter: Maximum scene CLOUDY_PIXEL_PERCENTAGE (default CLOUD_FILTER, 50). None keeps every scene. See get_s2_sr_cld_col.

    Returns:
        ee.ImageCollection of Sentinel-2 SR images with the matching Cloud Score+ image (bands 'cs', 'cs_cdf') saved as the 'cloud_score_plus' property on each image. Scenes without a Cloud Score+ match are dropped.
    """
    return _join_on_index(
        _get_s2_sr_base_col(aoi, start_date, end_date, cloud_filter),
        S2_CLOUD_SCORE_PLUS_COLLECTION,
        aoi,
        start_date,
        end_date,
        "cloud_score_plus",
    )


def add_cs_plus_mask(
    image: ee.Image,
    band: str = CS_PLUS_BAND,
    threshold: float = CS_PLUS_THRESH,
) -> ee.Image:
    """Add a 'cloudmask' band from the joined Cloud Score+ image.

    Cloud Score+ already scores shadows, haze and thin cirrus, so no shadow
    projection or morphology is applied (Google's recommended usage). Pixels where
    the score itself is masked stay masked once the cloud mask is applied.

    Args:
        image: Sentinel-2 ee.Image with the 'cloud_score_plus' property set (output of get_s2_sr_cs_col).
        band: Cloud Score+ band to threshold, 'cs_cdf' (default, CS_PLUS_BAND) or 'cs'.
        threshold: Clear threshold in [0, 1] (default CS_PLUS_THRESH, 0.60); pixels with score < threshold are flagged.

    Returns:
        ee.Image with an additional 'cloudmask' band where 1 = cloud, shadow or haze pixel.
    """
    score = ee.Image(image.get("cloud_score_plus")).select(band)
    return image.addBands(score.lt(threshold).rename("cloudmask"))


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
        image: ee.Image with a 'cloudmask' band added by add_cld_shdw_mask or add_cs_plus_mask.

    Returns:
        ee.Image with cloud and shadow pixels masked out across all bands.
    """
    return _masking.apply_cloud_mask(image)


def build_cloudfree_s2sr_col(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    cloud_filter: float | None = CLOUD_FILTER,
    cloud_mask: str = S2_DEFAULT_CLOUD_MASK,
    cs_band: str = CS_PLUS_BAND,
    cs_threshold: float = CS_PLUS_THRESH,
) -> ee.ImageCollection:
    """Build a Sentinel-2 SR collection with cloud and shadow masking fully applied.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.
        cloud_filter: Maximum scene CLOUDY_PIXEL_PERCENTAGE (default CLOUD_FILTER); None keeps every scene. See get_s2_sr_cld_col.
        cloud_mask: Pixel cloud-mask method: 'cloud_score_plus' (default, S2_DEFAULT_CLOUD_MASK) or 's2cloudless' (the pre-0.11.0 default; pass it to reproduce older composites).
        cs_band: Cloud Score+ band, 'cs_cdf' (default) or 'cs'. Ignored for s2cloudless.
        cs_threshold: Cloud Score+ clear threshold in [0, 1] (default CS_PLUS_THRESH, 0.60). Ignored for s2cloudless.

    Returns:
        ee.ImageCollection of cloud- and shadow-masked Sentinel-2 SR images.

    Raises:
        ValueError: If ``cloud_mask``, ``cs_band`` or ``cs_threshold`` is invalid (see validate_s2_cloud_mask_args).
    """
    validate_s2_cloud_mask_args(cloud_mask, cs_band, cs_threshold)
    if cloud_mask == "s2cloudless":
        return (
            get_s2_sr_cld_col(aoi, start_date, end_date, cloud_filter=cloud_filter)
            .map(add_cld_shdw_mask)
            .map(apply_cld_shdw_mask)
        )
    return (
        get_s2_sr_cs_col(aoi, start_date, end_date, cloud_filter=cloud_filter)
        .map(lambda img: add_cs_plus_mask(img, band=cs_band, threshold=cs_threshold))
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
