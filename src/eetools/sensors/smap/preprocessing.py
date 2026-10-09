import ee

from eetools.constants import SMAP_L4_COLLECTION, SMAP_SM_BANDS
from eetools.utils import validate_collection_date_range

# --------------------------------------------------------------------------- #
# Private helpers                                                              #
# --------------------------------------------------------------------------- #


def _daily_means(
    collection: ee.ImageCollection, start_date: ee.Date, end_date: ee.Date
) -> ee.ImageCollection:
    n_days = end_date.difference(start_date, "day").ceil()
    day_offsets = ee.List.sequence(0, n_days.subtract(1))

    def make_daily_mean(offset: ee.Number) -> ee.Image:
        day_start = start_date.advance(ee.Number(offset), "day")
        subset = collection.filterDate(day_start, day_start.advance(1, "day"))
        return ee.Image(
            subset.mean().set(
                {
                    "system:time_start": day_start.millis(),
                    "year": day_start.get("year"),
                    "month": day_start.get("month"),
                    "day": day_start.get("day"),
                    "image_count": subset.size(),
                }
            )
        )

    # Days with no source images (e.g. past the latest available timestep) yield
    # bandless means; drop them rather than return empty images.
    return ee.ImageCollection.fromImages(day_offsets.map(make_daily_mean)).filter(
        ee.Filter.gt("image_count", 0)
    )


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #


def validate_smap_date_range(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> None:
    """Validate that the requested date window overlaps available SMAP L4 soil
    moisture data over the AOI.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Requested start date as ee.Date.
        end_date: Requested end date as ee.Date.

    Returns:
        None. Raises ValueError if the date range falls outside the available SMAP L4 data extent.
    """
    validate_collection_date_range(
        collection_id=SMAP_L4_COLLECTION,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor_label="SMAP L4 soil moisture",
    )


# --------------------------------------------------------------------------- #
# Collection builder                                                           #
# --------------------------------------------------------------------------- #


def get_smap_collection(
    aoi: ee.Geometry,
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    bands: list[str] | None = None,
    daily: bool = True,
) -> ee.ImageCollection:
    """Build a NASA SMAP L4 (SPL4SMGP v008) ~9 km soil moisture collection for the AOI
    and date range.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as a string (e.g. '2020-01-01') or ee.Date.
        end_date: Collection end date as a string or ee.Date.
        bands: Bands to keep (default SMAP_SM_BANDS: surface/rootzone/profile volumetric soil moisture in m^3/m^3 and the matching 0-1 wetness bands). Any SPL4SMGP band name is accepted.
        daily: If True, averages the 3-hourly timesteps into one image per calendar day (UTC), each carrying system:time_start, year, month, day, and image_count properties; if False, returns the native 3-hourly images (default True).

    Returns:
        ee.ImageCollection of SMAP L4 soil moisture images (daily means or native 3-hourly). Raises ValueError if the date range falls outside the available SMAP L4 data.
    """
    start = ee.Date(start_date)
    end = ee.Date(end_date)

    validate_smap_date_range(aoi, start, end)

    col = (
        ee.ImageCollection(SMAP_L4_COLLECTION)
        .filterDate(start, end)
        .filterBounds(aoi)
        .select(bands or SMAP_SM_BANDS)
    )

    if daily:
        col = _daily_means(col, start, end)

    return col
