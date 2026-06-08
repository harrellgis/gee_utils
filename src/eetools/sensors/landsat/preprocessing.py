import ee

from eetools.constants import (
    L8_SR_COLLECTION,
    L8_SCALE_FACTOR,
    L8_ADD_OFFSET,
    L8_BAND_MAP,
)
from eetools.utils import validate_collection_date_range
from eetools.sensors.landsat.masking import (
    build_cloudfree_l8sr_col,
    build_l8_non_water_mask,
    apply_water_mask,
)
from eetools.sensors.indices import calc_indices, select_base_bands


def validate_l8_sr_date_range(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> None:
    """Validate that the requested date window overlaps available Landsat 8 SR imagery over the AOI.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Requested start date as ee.Date.
        end_date: Requested end date as ee.Date.

    Returns:
        None. Raises ValueError if the date range falls outside the available Landsat 8 SR data extent.
    """
    validate_collection_date_range(
        collection_id=L8_SR_COLLECTION,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor_label="Landsat 8 SR imagery",
    )


def process_l8_image(image: ee.Image) -> ee.Image:
    """Select core Landsat 8 SR bands, apply the scale factor and additive offset, and add spectral indices.

    Args:
        image: Raw Landsat 8 C2 L2 ee.Image with original band names (SR_B2 through SR_B7).

    Returns:
        ee.Image with calibrated reflectance bands and appended index bands, preserving system:time_start.
    """
    source = image
    image = select_base_bands(
        source,
        input_bands=["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
    )
    image = image.multiply(L8_SCALE_FACTOR).add(L8_ADD_OFFSET)
    image = calc_indices(image=image, band_map=L8_BAND_MAP, include_ndre=False)
    return image.copyProperties(source, ["system:time_start"])


def get_l8_sr_collection(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    apply_water_masking: bool = True,
) -> ee.ImageCollection:
    """Build a cloud-masked Landsat 8 SR collection with spectral indices.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.
        apply_water_masking: If True, applies a spectral water mask derived from the collection median (default True).

    Returns:
        ee.ImageCollection with calibrated SR bands and index bands appended.
    """
    validate_l8_sr_date_range(aoi, start_date, end_date)

    l8_cloudfree = build_cloudfree_l8sr_col(aoi, start_date, end_date)
    l8_processed = l8_cloudfree.map(process_l8_image)

    if not apply_water_masking:
        return l8_processed

    non_water_mask = build_l8_non_water_mask(l8_processed)
    return l8_processed.map(lambda img: apply_water_mask(img, non_water_mask))
