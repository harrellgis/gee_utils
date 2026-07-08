import ee

from eetools.constants import SATELLITE_EMBEDDING_COLLECTION
from eetools.utils import validate_collection_date_range


def validate_satellite_embedding_date_range(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> None:
    """Validate that the requested date window overlaps available Google Satellite
    Embedding V1 imagery over the AOI.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Requested start date as ee.Date.
        end_date: Requested end date as ee.Date.

    Returns:
        None. Raises ValueError if the date range falls outside the available Satellite Embedding data extent.
    """
    validate_collection_date_range(
        collection_id=SATELLITE_EMBEDDING_COLLECTION,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor_label="Satellite Embedding imagery",
    )


def get_satellite_embedding_collection(
    aoi: ee.Geometry,
    start_date: str | ee.Date,
    end_date: str | ee.Date,
) -> ee.ImageCollection:
    """Build a Google Satellite Embedding V1 collection over the AOI and date range.

    Satellite Embedding is an annual, analysis-ready product from the AlphaEarth
    Foundations model: each 10 m pixel is a 64-dimensional unit-length embedding vector
    (bands ``A00``–``A63``) summarizing one calendar year of multi-sensor surface
    trajectories. It is a foundation-model *output*, not spectral data, so unlike the
    optical sensors this builder applies **no** scale factor, cloud/water masking, or
    spectral indexing — the full 64-band stack is returned as-is.

    Because the cadence is annual (``system:time_start`` = Jan 1), a date window of
    ``[YYYY-01-01, (YYYY+1)-01-01)`` yields that single year's image(s) over the AOI.
    Each image is generated in its own local UTM zone (``UTM_ZONE`` property); for an AOI
    spanning multiple zones the collection contains one image per intersecting tile, which
    must be reprojected to a common CRS before stacking.

    The 64 bands collectively define one coordinate and are not independently
    interpretable — feed the whole stack to classifiers/clustering, and never compute
    per-band indices or select a subset for analysis (selecting three bands is valid only
    for RGB visualization).

    Args:
        aoi: Area of interest as ee.Geometry; used to filter the collection by bounds.
        start_date: Collection start date as a string (e.g. '2023-01-01') or ee.Date.
        end_date: Collection end date (exclusive) as a string or ee.Date.

    Returns:
        ee.ImageCollection of 64-band (``A00``–``A63``) annual embedding images filtered to the AOI and date range, from the GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL dataset.

    Raises:
        ValueError: If the date range is invalid or falls outside the available Satellite Embedding data extent over the AOI.
    """
    start = ee.Date(start_date)
    end = ee.Date(end_date)

    validate_satellite_embedding_date_range(aoi, start, end)

    return (
        ee.ImageCollection(SATELLITE_EMBEDDING_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start, end)
    )


def embedding_similarity(image_a: ee.Image, image_b: ee.Image) -> ee.Image:
    """Compute the per-pixel cosine similarity between two embedding images.

    Satellite Embedding vectors are unit-length, so their dot product equals the cosine
    similarity between the two years' 64-D vectors: ``1`` means identical surface
    conditions and lower values mean greater change — the product's built-in
    change-detection measure. Both images must carry the same embedding bands (the full
    ``A00``–``A63`` stack); pass two annual images (e.g. successive years) over the same
    AOI and CRS.

    Args:
        image_a: First embedding ee.Image (e.g. the earlier year).
        image_b: Second embedding ee.Image (e.g. the later year); must share image_a's bands.

    Returns:
        Single-band ee.Image named 'similarity' holding the per-pixel dot product (cosine similarity, in [-1, 1]) of the two embedding vectors.
    """
    dot = ee.Image(image_a).multiply(ee.Image(image_b)).reduce(ee.Reducer.sum())
    return dot.rename("similarity")
