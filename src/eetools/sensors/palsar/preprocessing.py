"""ALOS-2 PALSAR-2 yearly L-band mosaics as gamma-naught dB."""

import ee

from eetools.constants import (
    PALSAR2_CALIBRATION_DB,
    PALSAR2_POL_BANDS,
    PALSAR2_QA_BAND,
    PALSAR2_QA_LAND,
    PALSAR2_YEARLY_COLLECTION,
)
from eetools.utils import validate_collection_year_range


def palsar2_dn_to_db(image: ee.Image) -> ee.Image:
    """Convert PALSAR-2 HH/HV digital numbers to gamma-naught dB, land pixels only.

    gamma0 dB = 10 * log10(DN^2) - 83.0 (JAXA calibration), computed as
    20 * log10(DN) - 83.0. DN 0 (no data) and every pixel whose ``qa`` is not land
    (water, layover, radar shadow) are masked.

    Args:
        image: PALSAR-2 yearly ee.Image with bands HH, HV and qa.

    Returns:
        ee.Image with float bands ``HH`` and ``HV`` in dB, properties preserved.
    """
    image = ee.Image(image)
    dn = image.select(PALSAR2_POL_BANDS)
    db = (
        dn.toFloat()
        .log10()
        .multiply(20)
        .add(PALSAR2_CALIBRATION_DB)
        .updateMask(dn.gt(0))
        .updateMask(image.select(PALSAR2_QA_BAND).eq(PALSAR2_QA_LAND))
        .rename(PALSAR2_POL_BANDS)
    )
    return ee.Image(db.copyProperties(image, image.propertyNames()))


def get_palsar2_yearly(
    aoi: ee.Geometry, start_year: int, end_year: int
) -> ee.ImageCollection:
    """PALSAR-2 yearly 25 m mosaics (HH, HV gamma-naught dB), one image per year.

    Each year's tiles over the AOI are converted with ``palsar2_dn_to_db`` and
    mosaicked, with the default projection pinned to the source tiles (a bare
    ``mosaic()`` has none, which breaks ``reduceResolution``). Average backscatter in
    linear power, not dB, when aggregating. Record: 2015-2024.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_year: First year (inclusive).
        end_year: Last year (inclusive).

    Returns:
        ee.ImageCollection of float HH/HV dB images with ``year`` and ``system:time_start`` (1 January).

    Raises:
        ValueError: If end_year < start_year or the years fall outside the record over the AOI.
    """
    validate_collection_year_range(
        PALSAR2_YEARLY_COLLECTION, aoi, start_year, end_year, "PALSAR-2 yearly mosaic"
    )
    source = ee.ImageCollection(PALSAR2_YEARLY_COLLECTION).filterBounds(aoi)
    images = []
    for year in range(start_year, end_year + 1):
        tiles = source.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        projection = ee.Image(tiles.first()).select(PALSAR2_POL_BANDS[0]).projection()
        images.append(
            tiles.map(palsar2_dn_to_db)
            .mosaic()
            .setDefaultProjection(projection)
            .set(
                {
                    "year": year,
                    "system:time_start": ee.Date.fromYMD(year, 1, 1).millis(),
                }
            )
        )
    return ee.ImageCollection(images)
