"""GEDI L4A biomass and L2A height monthly rasters (25 m shots), quality-masked."""

from collections.abc import Sequence

import ee

from eetools.constants import (
    GEDI_DEGRADE_FLAG_BAND,
    GEDI_L2A_DEFAULT_BANDS,
    GEDI_L2A_MONTHLY_COLLECTION,
    GEDI_L2A_QUALITY_BAND,
    GEDI_L4A_DEFAULT_BANDS,
    GEDI_L4A_MONTHLY_COLLECTION,
    GEDI_L4A_QUALITY_BAND,
)
from eetools.utils import validate_collection_date_range


def mask_gedi_quality(
    image: ee.Image, quality_band: str, bands: Sequence[str]
) -> ee.Image:
    """Keep GEDI shots whose quality flag is 1 and ``degrade_flag`` is 0.

    Args:
        image: GEDI monthly ee.Image with ``quality_band``, ``degrade_flag`` and ``bands``.
        quality_band: ``l4_quality_flag`` (L4A) or ``quality_flag`` (L2A).
        bands: Bands to return.

    Returns:
        ee.Image with only ``bands``, masked to good shots, properties preserved.
    """
    image = ee.Image(image)
    good = (
        image.select(quality_band).eq(1).And(image.select(GEDI_DEGRADE_FLAG_BAND).eq(0))
    )
    return ee.Image(
        image.select(list(bands))
        .updateMask(good)
        .copyProperties(image, image.propertyNames())
    )


def _gedi_collection(
    collection_id: str,
    quality_band: str,
    aoi: ee.Geometry,
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    bands: Sequence[str],
    label: str,
) -> ee.ImageCollection:
    bands = list(bands)
    if not bands:
        raise ValueError("bands must name at least one band.")
    start, end = ee.Date(start_date), ee.Date(end_date)
    validate_collection_date_range(
        collection_id=collection_id,
        aoi=aoi,
        start_date=start,
        end_date=end,
        sensor_label=label,
    )

    return (
        ee.ImageCollection(collection_id)
        .filterDate(start, end)
        .filterBounds(aoi)
        .map(lambda image: mask_gedi_quality(image, quality_band, bands))
    )


def get_gedi_l4a_monthly(
    aoi: ee.Geometry,
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    bands: Sequence[str] = GEDI_L4A_DEFAULT_BANDS,
) -> ee.ImageCollection:
    """GEDI L4A footprint biomass (monthly 25 m rasters of shots), quality-masked.

    Keeps shots with ``l4_quality_flag == 1`` and ``degrade_flag == 0``. Each
    unmasked pixel is one shot. Record: 2019-03 to 2025-07 (GEDI V3 is not in EE).

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Start date (inclusive) as a string or ee.Date.
        end_date: End date (exclusive) as a string or ee.Date.
        bands: Bands to keep (default ``agbd`` and ``agbd_se``, Mg/ha).

    Returns:
        ee.ImageCollection of masked GEDI L4A images with the requested bands.

    Raises:
        ValueError: If bands is empty or the dates fall outside the record.
    """
    return _gedi_collection(
        GEDI_L4A_MONTHLY_COLLECTION,
        GEDI_L4A_QUALITY_BAND,
        aoi,
        start_date,
        end_date,
        bands,
        "GEDI L4A monthly",
    )


def get_gedi_l2a_monthly(
    aoi: ee.Geometry,
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    bands: Sequence[str] = GEDI_L2A_DEFAULT_BANDS,
) -> ee.ImageCollection:
    """GEDI L2A footprint relative heights (monthly 25 m rasters of shots), quality-masked.

    Keeps shots with ``quality_flag == 1`` and ``degrade_flag == 0``. Each unmasked
    pixel is one shot. Record: 2019-03 to 2025-07 (GEDI V3 is not in EE).

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Start date (inclusive) as a string or ee.Date.
        end_date: End date (exclusive) as a string or ee.Date.
        bands: Bands to keep (default ``rh50``, ``rh75``, ``rh95``, ``rh98``, ``rh100``, m).

    Returns:
        ee.ImageCollection of masked GEDI L2A images with the requested bands.

    Raises:
        ValueError: If bands is empty or the dates fall outside the record.
    """
    return _gedi_collection(
        GEDI_L2A_MONTHLY_COLLECTION,
        GEDI_L2A_QUALITY_BAND,
        aoi,
        start_date,
        end_date,
        bands,
        "GEDI L2A monthly",
    )


def gedi_footprint_counts(
    collection: ee.ImageCollection,
    regions: ee.FeatureCollection,
    start_year: int,
    end_year: int,
    *,
    scale: float,
    id_property: str = "site_id",
) -> ee.FeatureCollection:
    """Count valid GEDI shots per region and calendar year.

    For each year the masked collection's first band is counted per pixel (one
    shot per pixel per month) and summed over each region with an **unweighted**
    sum on the collection's own grid, so each shot counts once. Use the native
    ``GEDI_SCALE_M`` (25 m); a coarser scale would sample rather than count.
    Export the rows with ``io.export_csv_table``.

    Args:
        collection: Output of ``get_gedi_l4a_monthly`` / ``get_gedi_l2a_monthly`` (or any masked GEDI collection).
        regions: Polygons to count over.
        start_year: First year (inclusive).
        end_year: Last year (inclusive).
        scale: Reduction scale in metres (required; use 25).
        id_property: Region property written as ``region_id``.

    Returns:
        ee.FeatureCollection of geometry-free rows ``region_id``, ``year``, ``n_footprints`` (0 when none), one per region x year.

    Raises:
        ValueError: If end_year < start_year or scale is not positive.
    """
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) must be >= start_year ({start_year}).")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}.")
    collection = ee.ImageCollection(collection)
    regions = ee.FeatureCollection(regions)
    first = collection.first().select(0)
    projection = ee.Image(first).projection()

    tables = []
    for year in range(start_year, end_year + 1):
        counts = (
            collection.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .map(lambda image: ee.Image(image).select(0).rename("shots"))
            .count()
            .unmask(0, False)
            .setDefaultProjection(projection)
        )
        reduced = counts.reduceRegions(
            collection=regions, reducer=ee.Reducer.sum().unweighted(), scale=scale
        )
        tables.append(
            reduced.map(
                lambda f, year=year: ee.Feature(
                    None,
                    {
                        "region_id": f.get(id_property),
                        "year": year,
                        "n_footprints": f.get("sum"),
                    },
                )
            )
        )
    return ee.FeatureCollection(tables).flatten()
