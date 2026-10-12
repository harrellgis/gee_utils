import ee

from eetools.constants import (
    SM2RAIN_CLIMATOLOGY_IMAGE,
    SM2RAIN_MONTH_BANDS,
    SM2RAIN_PRECIP_BAND,
)


def get_sm2rain_climatology(months: list[int] | None = None) -> ee.ImageCollection:
    """Build the OpenLandMap SM2RAIN 1 km long-term (2007-2018) monthly precipitation
    climatology as one image per calendar month.

    The source is a single 12-band ee.Image (jan..dec), not a time series, so there is
    no date filtering or validation; images carry a 'month' property instead of
    system:time_start.

    Args:
        months: Calendar months to include as integers 1-12 (default all twelve, in the order given).

    Returns:
        ee.ImageCollection with one image per requested month, each holding a single 'precipitation' band (mean mm/month) and 'month' (1-12) and 'month_name' (e.g. 'jan') properties. Raises ValueError if any month is outside 1-12.
    """
    months = list(range(1, 13)) if months is None else months
    invalid = [m for m in months if not 1 <= m <= 12]
    if invalid:
        raise ValueError(f"months must be integers 1-12; got invalid values {invalid}.")

    climatology = ee.Image(SM2RAIN_CLIMATOLOGY_IMAGE)
    images = []
    for month in months:
        month_band = SM2RAIN_MONTH_BANDS[month - 1]
        images.append(
            climatology.select([month_band])
            .rename(SM2RAIN_PRECIP_BAND)
            .set({"month": month, "month_name": month_band})
        )
    return ee.ImageCollection(images)
