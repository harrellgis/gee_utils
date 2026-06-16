import ee

from eetools.constants import CHIRPS_COLLECTION, CHIRPS_PRECIP_BAND
from eetools.io import export_table_to_drive
from eetools.utils import validate_collection_date_range
from eetools.visualization.summaries import (
    build_period_composites,
    collection_to_region_timeseries,
)


def validate_chirps_date_range(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> None:
    """Validate that the requested date window overlaps available CHIRPS v3 Daily
    Reanalysis imagery over the AOI.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Requested start date as ee.Date.
        end_date: Requested end date as ee.Date.

    Returns:
        None. Raises ValueError if the date range falls outside the available CHIRPS data extent.
    """
    validate_collection_date_range(
        collection_id=CHIRPS_COLLECTION,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor_label="CHIRPS imagery",
    )


def process_chirps_image(image: ee.Image) -> ee.Image:
    """Select the CHIRPS precipitation band and preserve temporal properties.

    Args:
        image: Raw CHIRPS ee.Image with the precipitation band and time properties.

    Returns:
        ee.Image containing only the precipitation band (mm/day), with system:time_start, year, month, and day properties preserved.
    """
    source = ee.Image(image)
    image = source.select([CHIRPS_PRECIP_BAND])
    return image.copyProperties(source, ["system:time_start", "year", "month", "day"])


def get_chirps_collection(
    aoi: ee.Geometry,
    start_date: str | ee.Date,
    end_date: str | ee.Date,
) -> ee.ImageCollection:
    """Build a CHIRPS v3 Daily Reanalysis precipitation collection for the AOI and date
    range.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as a string (e.g. '2020-01-01') or ee.Date.
        end_date: Collection end date as a string or ee.Date.

    Returns:
        ee.ImageCollection of daily precipitation images (mm/day) from the UCSB-CHC/CHIRPS/V3/DAILY_RNL dataset.
    """
    start = ee.Date(start_date)
    end = ee.Date(end_date)

    validate_chirps_date_range(aoi, start, end)

    return (
        ee.ImageCollection(CHIRPS_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start, end)
        .map(process_chirps_image)
    )


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
    )

    export_table_to_drive(
        collection=rainfall_ts,
        description=file_prefix,
        folder=export_folder,
        fileNamePrefix=file_prefix,
        fileFormat="CSV",
    )

    return rainfall_ts
