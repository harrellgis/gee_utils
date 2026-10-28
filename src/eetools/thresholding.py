"""Adaptive (per-scene) image thresholding with Otsu's method.

Otsu picks the value that best splits a bimodal histogram into two classes
(maximum between-class sum of squares), so the threshold adapts to each scene
instead of being fixed. Edge-based Otsu (Donchyts et al. 2016) builds that
histogram only from pixels near Canny edges, where both classes are present in
similar amounts, which keeps the split reliable when one class dominates the
region (e.g. a little water in a mostly-land scene).
"""

import ee

from eetools.constants import (
    CANNY_SIGMA,
    CANNY_THRESHOLD,
    DEFAULT_MAX_PIXELS,
    OTSU_HISTOGRAM_BUCKETS,
)

# --------------------------------------------------------------------------- #
# Private helpers                                                              #
# --------------------------------------------------------------------------- #


def _validate_buckets(buckets: int) -> None:
    if buckets < 2:
        raise ValueError(f"buckets must be at least 2; got {buckets}")


def _region_histogram(
    image: ee.Image, region: ee.Geometry, scale: float, buckets: int
) -> ee.Dictionary:
    # Histogram of the first band; an empty dictionary when no pixels are valid
    # (the reducer then returns null for the band).
    image = ee.Image(image).select([0])
    stats = image.reduceRegion(
        reducer=ee.Reducer.histogram(buckets),
        geometry=region,
        scale=scale,
        maxPixels=int(DEFAULT_MAX_PIXELS),
        tileScale=4,
    )
    histogram = stats.get(image.bandNames().get(0))
    return ee.Dictionary(ee.Algorithms.If(histogram, histogram, {}))


def _threshold_from_histogram(
    histogram: ee.Dictionary, fallback: float, min_value: float | None
) -> ee.Number:
    threshold = ee.Number(
        ee.Algorithms.If(
            histogram.contains("bucketMeans"), otsu_threshold(histogram), fallback
        )
    )
    if min_value is not None:
        threshold = threshold.max(min_value)
    return threshold


# --------------------------------------------------------------------------- #
# Otsu                                                                         #
# --------------------------------------------------------------------------- #


def otsu_threshold(histogram: ee.Dictionary) -> ee.Number:
    """Compute the Otsu threshold of a histogram produced by ee.Reducer.histogram.

    Each bucket mean is tried as the split point; the one that maximizes the
    between-class sum of squares of the two resulting classes is returned.

    Args:
        histogram: ee.Dictionary with 'histogram' (counts) and 'bucketMeans' arrays, as output by ee.Reducer.histogram for one band.

    Returns:
        ee.Number bucket mean at which the histogram is best split into two classes.
    """
    histogram = ee.Dictionary(histogram)
    counts = ee.Array(histogram.get("histogram"))
    means = ee.Array(histogram.get("bucketMeans"))
    size = means.length().get([0])
    total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    sum_ = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
    mean = sum_.divide(total)
    indices = ee.List.sequence(1, size)

    # Between-class sum of squares when the first i buckets form class a.
    def compute_bss(i: ee.Number) -> ee.Number:
        a_counts = counts.slice(0, 0, i)
        a_count = a_counts.reduce(ee.Reducer.sum(), [0]).get([0])
        a_means = means.slice(0, 0, i)
        a_mean = (
            a_means.multiply(a_counts)
            .reduce(ee.Reducer.sum(), [0])
            .get([0])
            .divide(a_count)
        )
        b_count = total.subtract(a_count)
        b_mean = sum_.subtract(a_count.multiply(a_mean)).divide(b_count)
        return a_count.multiply(a_mean.subtract(mean).pow(2)).add(
            b_count.multiply(b_mean.subtract(mean).pow(2))
        )

    bss = indices.map(compute_bss)
    return ee.Number(means.sort(bss).get([-1]))


def otsu_threshold_image(
    image: ee.Image,
    region: ee.Geometry,
    scale: float,
    buckets: int = OTSU_HISTOGRAM_BUCKETS,
    min_value: float | None = None,
    fallback: float = 0.0,
) -> ee.Number:
    """Compute a plain Otsu threshold from all valid pixels of an image in a region.

    Use this when the region's histogram is already clearly bimodal; when one class
    dominates, prefer :func:`edge_otsu_threshold`.

    Args:
        image: ee.Image whose first band is thresholded (e.g. NDWI, or VV backscatter in dB).
        region: ee.Geometry over which the histogram is computed.
        scale: Pixel size in metres for the histogram reduction.
        buckets: Number of histogram buckets (default OTSU_HISTOGRAM_BUCKETS); must be at least 2.
        min_value: Optional lower bound; the threshold is raised to this value if it falls below it.
        fallback: Value returned when the region has no valid pixels (default 0.0); set it per index.

    Returns:
        ee.Number threshold. Raises ValueError if buckets < 2.
    """
    _validate_buckets(buckets)
    histogram = _region_histogram(image, region, scale, buckets)
    return _threshold_from_histogram(histogram, fallback, min_value)


# --------------------------------------------------------------------------- #
# Edge sampling                                                                #
# --------------------------------------------------------------------------- #


def canny_edges(
    image: ee.Image,
    scale: float,
    canny_threshold: float = CANNY_THRESHOLD,
    canny_sigma: float = CANNY_SIGMA,
) -> ee.Image:
    """Detect sharp transitions in an image with the Canny edge detector.

    Edges within 3 * scale metres of the image's masked border are dropped so the
    data boundary is not mistaken for a class boundary.

    Args:
        image: ee.Image whose first band is searched for edges.
        scale: Pixel size in metres; sets the border-erosion distance.
        canny_threshold: Gradient magnitude threshold, in the units of the image (default CANNY_THRESHOLD, tuned for normalized-difference indices).
        canny_sigma: Gaussian smoothing sigma in pixels applied before edge detection (default CANNY_SIGMA).

    Returns:
        ee.Image with one band that is non-zero on edge pixels (mask with itself to display).
    """
    image = ee.Image(image).select([0])
    interior = (
        image.mask()
        .gt(0)
        .focalMin(
            radius=ee.Number(scale).multiply(3), kernelType="circle", units="meters"
        )
    )
    edges = ee.Algorithms.CannyEdgeDetector(image, canny_threshold, canny_sigma)
    return ee.Image(edges).multiply(interior)


def canny_edge_buffer(
    image: ee.Image,
    scale: float,
    canny_threshold: float = CANNY_THRESHOLD,
    canny_sigma: float = CANNY_SIGMA,
) -> ee.Image:
    """Mask an image to the pixels within one pixel (scale metres) of Canny edges.

    These pixels straddle the class boundary, so their histogram holds both classes
    in similar proportions; this is the sample used by :func:`edge_otsu_threshold`,
    and is useful for plotting that histogram.

    Args:
        image: ee.Image whose first band is sampled.
        scale: Pixel size in metres; sets the edge buffer distance.
        canny_threshold: Gradient magnitude threshold, in the units of the image (default CANNY_THRESHOLD).
        canny_sigma: Gaussian smoothing sigma in pixels (default CANNY_SIGMA).

    Returns:
        ee.Image of the first band, masked to pixels near detected edges.
    """
    image = ee.Image(image).select([0])
    edges = canny_edges(image, scale, canny_threshold, canny_sigma)
    edge_buffer = edges.focalMax(radius=scale, kernelType="square", units="meters")
    return image.updateMask(edge_buffer)


# --------------------------------------------------------------------------- #
# Edge-based threshold                                                         #
# --------------------------------------------------------------------------- #


def edge_otsu_threshold(
    image: ee.Image,
    region: ee.Geometry,
    scale: float,
    canny_threshold: float = CANNY_THRESHOLD,
    canny_sigma: float = CANNY_SIGMA,
    buckets: int = OTSU_HISTOGRAM_BUCKETS,
    min_value: float | None = None,
    fallback: float = 0.0,
) -> ee.Number:
    """Compute an Otsu threshold from pixels near Canny edges (edge-based Otsu).

    Typical uses: surface water from NDWI/MNDWI or Sentinel-1 VV/VH (dB), burn scars
    from dNBR, and change maps from index differences.

    Args:
        image: ee.Image whose first band is thresholded.
        region: ee.Geometry over which the edge histogram is computed.
        scale: Pixel size in metres for edge buffering and the histogram reduction.
        canny_threshold: Gradient magnitude threshold, in the units of the image (default CANNY_THRESHOLD).
        canny_sigma: Gaussian smoothing sigma in pixels (default CANNY_SIGMA).
        buckets: Number of histogram buckets (default OTSU_HISTOGRAM_BUCKETS); must be at least 2.
        min_value: Optional lower bound; the threshold is raised to this value if it falls below it.
        fallback: Value returned when no edges are found in the region (default 0.0); set it per index (e.g. 0.3 for NDWI in the GEE book).

    Returns:
        ee.Number threshold. Raises ValueError if buckets < 2.
    """
    _validate_buckets(buckets)
    image_edge = canny_edge_buffer(image, scale, canny_threshold, canny_sigma)
    histogram = _region_histogram(image_edge, region, scale, buckets)
    return _threshold_from_histogram(histogram, fallback, min_value)
