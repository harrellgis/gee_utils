import ee

from eetools.constants import S1_EDGE_THRESHOLD_DB, S1_SPECKLE_RADIUS_M


def mask_edges(
    image: ee.Image,
    edge_threshold_db: float = S1_EDGE_THRESHOLD_DB,
) -> ee.Image:
    """Mask Sentinel-1 scene-edge pixels whose backscatter falls below a dB threshold.

    GRD scenes carry very low backscatter along their swath edges (an artefact of the
    terrain-correction border). Pixels below ``edge_threshold_db`` are masked per band.

    Args:
        image: Sentinel-1 GRD ee.Image with backscatter bands in dB (e.g. VV, VH).
        edge_threshold_db: Backscatter (dB) below which a pixel is treated as a noisy scene edge and masked out (default S1_EDGE_THRESHOLD_DB).

    Returns:
        ee.Image with edge pixels masked across all bands, preserving image properties.
    """
    not_edge = image.gte(edge_threshold_db)
    return image.updateMask(image.mask().And(not_edge))


def apply_speckle_filter(
    image: ee.Image,
    radius_m: float = S1_SPECKLE_RADIUS_M,
) -> ee.Image:
    """Apply a focal-median speckle filter to Sentinel-1 backscatter.

    GRD scenes are calibrated and terrain-corrected but not speckle-filtered. A focal
    median suppresses speckle while preserving edges and, because the median commutes
    with the monotonic dB log transform, it is unbiased in dB (unlike a focal mean,
    which would need to be computed in linear power space to avoid bias).

    Args:
        image: Sentinel-1 GRD ee.Image with backscatter bands in dB.
        radius_m: Focal-median neighbourhood radius in metres (default S1_SPECKLE_RADIUS_M).

    Returns:
        ee.Image of the speckle-filtered backscatter with the original band names and system:time_start preserved.
    """
    source = image
    filtered = image.focalMedian(radius=radius_m, units="meters")
    return ee.Image(filtered.copyProperties(source, ["system:time_start"]))
