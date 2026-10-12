import ee

from eetools.constants import (
    VIIRS_NTL_BANDS,
    VIIRS_NTL_COLLECTION,
    VIIRS_NTL_MAX_QUALITY_FLAG,
)
from eetools.sensors.viirs.masking import mask_viirs_ntl_qa
from eetools.utils import validate_collection_date_range


def validate_viirs_ntl_date_range(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> None:
    """Validate that the requested date window overlaps available VIIRS VNP46A2
    nighttime-lights imagery over the AOI.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Requested start date as ee.Date.
        end_date: Requested end date as ee.Date.

    Returns:
        None. Raises ValueError if the date range falls outside the available VNP46A2 data extent.
    """
    validate_collection_date_range(
        collection_id=VIIRS_NTL_COLLECTION,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor_label="VIIRS VNP46A2 nighttime lights",
    )


def get_viirs_ntl_collection(
    aoi: ee.Geometry,
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    apply_qa_mask: bool = True,
    max_quality_flag: int = VIIRS_NTL_MAX_QUALITY_FLAG,
    mask_snow: bool = True,
) -> ee.ImageCollection:
    """Build a NASA Black Marble VIIRS VNP46A2 v002 daily ~500 m nighttime-lights
    collection for the AOI and date range.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as a string (e.g. '2020-01-01') or ee.Date.
        end_date: Collection end date as a string or ee.Date.
        apply_qa_mask: If True, masks DNB_BRDF_Corrected_NTL to high-quality, snow-free retrievals; the gap-filled band is never masked (default True).
        max_quality_flag: Highest Mandatory_Quality_Flag kept when masking: 0 persistent, 1 ephemeral (default 1).
        mask_snow: If True and apply_qa_mask is True, also masks snow/ice pixels (default True).

    Returns:
        ee.ImageCollection of daily VNP46A2 images with the VIIRS_NTL_BANDS: DNB_BRDF_Corrected_NTL and Gap_Filled_DNB_BRDF_Corrected_NTL radiance (nW/cm^2/sr), Latest_High_Quality_Retrieval (age in days of the gap-fill source), and the quality/snow/cloud flags. Raises ValueError if the date range falls outside the available VNP46A2 data.
    """
    start = ee.Date(start_date)
    end = ee.Date(end_date)

    validate_viirs_ntl_date_range(aoi, start, end)

    col = (
        ee.ImageCollection(VIIRS_NTL_COLLECTION)
        .filterDate(start, end)
        .filterBounds(aoi)
        .select(VIIRS_NTL_BANDS)
    )

    if apply_qa_mask:
        col = col.map(lambda img: mask_viirs_ntl_qa(img, max_quality_flag, mask_snow))

    return col
