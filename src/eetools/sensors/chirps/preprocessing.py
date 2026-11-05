from typing import Literal

import ee

from eetools.compositing import build_period_composites
from eetools.constants import (
    CHIRPS_CADENCE_COLLECTIONS,
    CHIRPS_COLLECTION,
    CHIRPS_PRECIP_BAND,
    CLIMATOLOGY_END_YEAR,
    CLIMATOLOGY_START_YEAR,
)
from eetools.io import export_table_to_drive
from eetools.utils import validate_collection_date_range
from eetools.zonal import collection_to_region_timeseries


def _chirps_collection_id(cadence: str) -> str:
    if cadence not in CHIRPS_CADENCE_COLLECTIONS:
        raise ValueError(
            f"cadence must be one of {sorted(CHIRPS_CADENCE_COLLECTIONS)}, "
            f"got '{cadence}'."
        )
    return CHIRPS_CADENCE_COLLECTIONS[cadence]


def validate_chirps_date_range(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    collection_id: str = CHIRPS_COLLECTION,
) -> None:
    """Validate that the requested date window overlaps available CHIRPS v3 imagery
    (the daily reanalysis by default, or the pentad collection) over the AOI.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Requested start date as ee.Date.
        end_date: Requested end date as ee.Date.
        collection_id: CHIRPS collection to validate against (default the daily product).

    Returns:
        None. Raises ValueError if the date range falls outside the available CHIRPS data extent.
    """
    validate_collection_date_range(
        collection_id=collection_id,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor_label="CHIRPS imagery",
    )


def process_chirps_image(image: ee.Image) -> ee.Image:
    """Select the CHIRPS precipitation band and preserve temporal properties.

    Works for both cadences: daily images carry ``day`` (mm/day), pentad images carry
    ``pentad`` (mm per pentad). Sums over a year or more agree between cadences; daily
    means do not.

    Args:
        image: Raw CHIRPS ee.Image with the precipitation band and time properties.

    Returns:
        ee.Image containing only the precipitation band (mm/day for daily images, mm per pentad for pentad images), with system:time_start, year, month, day and pentad properties preserved where present.
    """
    source = ee.Image(image)
    image = source.select([CHIRPS_PRECIP_BAND])
    return image.copyProperties(
        source, ["system:time_start", "year", "month", "day", "pentad"]
    )


def get_chirps_collection(
    aoi: ee.Geometry,
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    cadence: Literal["daily", "pentad"] = "daily",
) -> ee.ImageCollection:
    """Build a CHIRPS v3 precipitation collection for the AOI and date range.

    ``cadence="daily"`` (default) uses the Daily Reanalysis (mm/day);
    ``cadence="pentad"`` uses the pentad collection (mm per pentad, 72 per year).
    Annual or multi-year totals agree between the two to < 0.0001 mm, so request
    ``"pentad"`` for whole-year sums (~5x fewer image reads) and ``"daily"`` for short
    windows (pentad edges can shift a window by up to 5 days).

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as a string (e.g. '2020-01-01') or ee.Date.
        end_date: Collection end date as a string or ee.Date.
        cadence: 'daily' (UCSB-CHC/CHIRPS/V3/DAILY_RNL, default) or 'pentad' (UCSB-CHC/CHIRPS/V3/PENTAD).

    Returns:
        ee.ImageCollection of precipitation images (band 'precipitation').

    Raises:
        ValueError: If cadence is unknown or the date range falls outside the available data.
    """
    collection_id = _chirps_collection_id(cadence)
    start = ee.Date(start_date)
    end = ee.Date(end_date)

    validate_chirps_date_range(aoi, start, end, collection_id=collection_id)

    return (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi)
        .filterDate(start, end)
        .map(process_chirps_image)
    )


def get_chirps_climatology(
    aoi: ee.Geometry,
    start_year: int = CLIMATOLOGY_START_YEAR,
    end_year: int = CLIMATOLOGY_END_YEAR,
) -> ee.Image:
    """Mean annual CHIRPS rainfall over a fixed window, built once as one image.

    Pentads are summed over the whole window and divided by the number of years
    (identical to averaging per-year sums), at native ~5.5 km.

    Args:
        aoi: Area of interest as ee.Geometry (used for bounds and date validation).
        start_year: First year (inclusive; default 1991).
        end_year: Last year (inclusive; default 2020).

    Returns:
        ee.Image with a single 'rain_mean_annual' band (mm/yr), native projection.

    Raises:
        ValueError: If end_year < start_year or the window is outside the data.
    """
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) must be >= start_year ({start_year}).")
    col = get_chirps_collection(
        aoi, f"{start_year}-01-01", f"{end_year + 1}-01-01", cadence="pentad"
    )
    n_years = end_year - start_year + 1
    projection = ee.Image(col.first()).projection()
    return (
        col.select(CHIRPS_PRECIP_BAND)
        .sum()
        .divide(n_years)
        .rename("rain_mean_annual")
        .toFloat()
        .setDefaultProjection(projection)
    )


def get_chirps_annual_totals(
    aoi: ee.Geometry,
    start_year: int,
    end_year: int,
) -> ee.ImageCollection:
    """Annual CHIRPS rainfall totals (pentad cadence), one image per calendar year.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_year: First year (inclusive).
        end_year: Last year (inclusive).

    Returns:
        ee.ImageCollection of single-band 'precipitation' images (mm/yr), each with
        'year' and 'system:time_start' (Jan 1) properties.

    Raises:
        ValueError: If end_year < start_year or the window is outside the data.
    """
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) must be >= start_year ({start_year}).")
    col = get_chirps_collection(
        aoi, f"{start_year}-01-01", f"{end_year + 1}-01-01", cadence="pentad"
    )

    def _year_total(year: ee.Number) -> ee.Image:
        start = ee.Date.fromYMD(year, 1, 1)
        return (
            col.filterDate(start, start.advance(1, "year"))
            .select(CHIRPS_PRECIP_BAND)
            .sum()
            .rename(CHIRPS_PRECIP_BAND)
            .set({"year": year, "system:time_start": start.millis()})
        )

    return ee.ImageCollection(ee.List.sequence(start_year, end_year).map(_year_total))


def export_total_rainfall_table(
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    aoi: ee.Geometry,
    temporal_scale: str,
    export_folder: str,
    reducer: ee.Reducer | None = None,
    scale: int = 5566,
    file_prefix: str | None = None,
) -> ee.FeatureCollection:
    """Compute total CHIRPS rainfall per period over an AOI and export it as a CSV table.

    Runs the full workflow end to end: build the daily CHIRPS collection for the window,
    sum daily precipitation into monthly or annual totals (one image per period), reduce
    each period's total over the AOI (default: spatial mean depth in mm), and start a
    Google Drive table-export task with the resulting long-format timeseries.

    Args:
        start_date: Window start as a string (e.g. '2020-01-01') or ee.Date.
        end_date: Window end (exclusive) as a string or ee.Date.
        aoi: Area of interest as ee.Geometry; bounds both the collection and the reduction.
        temporal_scale: Aggregation period; either 'monthly' or 'annual'.
        export_folder: Google Drive folder name to write the CSV into.
        reducer: Spatial ee.Reducer applied per period over the AOI; defaults to ee.Reducer.mean() (AOI-mean rainfall depth in mm).
        scale: Pixel scale in metres for the reduction (default 5566, CHIRPS native resolution).
        file_prefix: Filename prefix (without extension) and task description; defaults to 'chirps_{temporal_scale}_total_rainfall'.

    Returns:
        ee.FeatureCollection of the per-period rainfall timeseries (one feature per period, carrying date/year/month metadata and the per-band reduction result). A batch table-export task to Drive is started as a side effect.

    Raises:
        ValueError: If temporal_scale is not 'monthly' or 'annual', or if the date range falls outside the available CHIRPS data extent over the AOI.
    """
    valid_scales = {"monthly", "annual"}
    if temporal_scale not in valid_scales:
        raise ValueError(f"temporal_scale must be one of {sorted(valid_scales)}")

    if file_prefix is None:
        file_prefix = f"chirps_{temporal_scale}_total_rainfall"

    daily = get_chirps_collection(aoi, start_date, end_date)

    period_totals = build_period_composites(
        daily,
        bands=[CHIRPS_PRECIP_BAND],
        start_date=start_date,
        end_date=end_date,
        temporal_scale=temporal_scale,
        composite_stat="sum",
    )

    rainfall_ts = collection_to_region_timeseries(
        period_totals,
        region=aoi,
        bands=[CHIRPS_PRECIP_BAND],
        reducer=reducer,
        scale=scale,
        # 5566 m is CHIRPS' native resolution: no finer scale exists to suggest.
        min_pixels_per_region=None,
    )

    export_table_to_drive(
        collection=rainfall_ts,
        description=file_prefix,
        folder=export_folder,
        fileNamePrefix=file_prefix,
        fileFormat="CSV",
    )

    return rainfall_ts
