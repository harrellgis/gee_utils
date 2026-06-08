import ee

from eetools.constants import CHIRPS_COLLECTION, CHIRPS_PRECIP_BAND
from eetools.utils import validate_collection_date_range


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
