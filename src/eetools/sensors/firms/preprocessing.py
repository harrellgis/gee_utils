"""FIRMS MODIS active-fire detections (daily, 1 km) and per-region annual counts."""

import ee

from eetools.constants import (
    FIRMS_BANDS,
    FIRMS_COLLECTION,
    FIRMS_CONFIDENCE_BAND,
    FIRMS_DEFAULT_MIN_CONFIDENCE,
    FIRMS_FIRE_BAND,
)
from eetools.utils import validate_collection_date_range


def add_fire_band(
    image: ee.Image, min_confidence: float = FIRMS_DEFAULT_MIN_CONFIDENCE
) -> ee.Image:
    """Add a 0/1 ``fire`` band: 1 where a detection has confidence >= min_confidence.

    Args:
        image: FIRMS ee.Image with the ``confidence`` band.
        min_confidence: Minimum detection confidence in % (default 30, dropping MODIS "low" confidence).

    Returns:
        ee.Image with the FIRMS bands plus ``fire`` (1 = detection kept, 0 = below the threshold, masked where FIRMS has no detection), properties preserved.
    """
    image = ee.Image(image)
    fire = (
        image.select(FIRMS_CONFIDENCE_BAND)
        .gte(min_confidence)
        .rename(FIRMS_FIRE_BAND)
        .toUint8()
    )
    return ee.Image(image.addBands(fire).copyProperties(image, image.propertyNames()))


def get_firms(
    aoi: ee.Geometry,
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    min_confidence: float = FIRMS_DEFAULT_MIN_CONFIDENCE,
) -> ee.ImageCollection:
    """Build a FIRMS active-fire collection (one image per day, 1 km) with a ``fire`` flag.

    FIRMS images are masked wherever nothing was detected, so sum ``fire`` over time
    (after ``unmask(0)`` if a full grid is needed) to get fire pixel-days.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Start date (inclusive) as a string or ee.Date.
        end_date: End date (exclusive) as a string or ee.Date.
        min_confidence: Minimum detection confidence in % for ``fire`` = 1 (default 30). Must be in [0, 100].

    Returns:
        ee.ImageCollection of daily images with bands ``T21`` (K), ``confidence`` (%) and ``fire`` (0/1).

    Raises:
        ValueError: If min_confidence is outside [0, 100] or the dates fall outside the FIRMS record.
    """
    if not 0 <= min_confidence <= 100:
        raise ValueError(f"min_confidence must be in [0, 100], got {min_confidence}.")
    start, end = ee.Date(start_date), ee.Date(end_date)
    validate_collection_date_range(
        collection_id=FIRMS_COLLECTION,
        aoi=aoi,
        start_date=start,
        end_date=end,
        sensor_label="FIRMS active fire",
    )
    return (
        ee.ImageCollection(FIRMS_COLLECTION)
        .filterDate(start, end)
        .filterBounds(aoi)
        .select(FIRMS_BANDS)
        .map(lambda image: add_fire_band(image, min_confidence))
    )


def firms_counts(
    regions: ee.FeatureCollection,
    start_year: int,
    end_year: int,
    *,
    scale: float,
    id_property: str = "site_id",
    min_confidence: float = FIRMS_DEFAULT_MIN_CONFIDENCE,
) -> ee.FeatureCollection:
    """Count active-fire pixel-days per region and calendar year.

    For each year the daily ``fire`` flags are summed per pixel, then summed over each
    region with an **unweighted** sum, so a 1 km detection counts once when its pixel
    centre is in the region (a weighted sum would count fractions of edge pixels).
    Small regions (a few km) may contain no pixel centre at 1 km; pass the regions'
    buffered windows if that matters. Export the rows with ``io.export_csv_table``.

    Args:
        regions: Polygons to count over.
        start_year: First year (inclusive).
        end_year: Last year (inclusive).
        scale: Reduction scale in metres (required; FIRMS native is ``FIRMS_NATIVE_SCALE_M`` = 1000).
        id_property: Region property written as ``region_id``.
        min_confidence: Minimum detection confidence in % (default 30).

    Returns:
        ee.FeatureCollection of geometry-free rows with ``region_id``, ``year`` and ``fire_count`` (fire pixel-days; 0 when none), one per region x year.

    Raises:
        ValueError: If end_year < start_year, scale is not positive, min_confidence is outside [0, 100], or the years fall outside the FIRMS record.
    """
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) must be >= start_year ({start_year}).")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}.")
    regions = ee.FeatureCollection(regions)
    fires = get_firms(
        regions.geometry().bounds(),
        f"{start_year}-01-01",
        f"{end_year + 1}-01-01",
        min_confidence,
    ).select(FIRMS_FIRE_BAND)

    tables = []
    for year in range(start_year, end_year + 1):
        total = (
            fires.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .sum()
            .unmask(0, False)
            .rename(FIRMS_FIRE_BAND)
        )
        reduced = total.reduceRegions(
            collection=regions, reducer=ee.Reducer.sum().unweighted(), scale=scale
        )
        tables.append(
            reduced.map(
                lambda f, year=year: ee.Feature(
                    None,
                    {
                        "region_id": f.get(id_property),
                        "year": year,
                        "fire_count": f.get("sum"),
                    },
                )
            )
        )
    return ee.FeatureCollection(tables).flatten()
