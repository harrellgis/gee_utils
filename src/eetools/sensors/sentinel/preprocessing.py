import ee

from eetools.constants import S2_BAND_MAP, S2_SCALE_FACTOR, S2_SR_COLLECTION
from eetools.sensors.indices import calc_indices, select_base_bands
from eetools.sensors.masking import validate_water_mask_selection
from eetools.sensors.sentinel.masking import (
    apply_water_mask,
    build_cloudfree_s2sr_col,
    build_s2_non_water_mask,
)
from eetools.utils import validate_collection_date_range


def validate_s2_sr_date_range(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> None:
    """Validate that the requested date window overlaps available Sentinel-2 SR imagery
    over the AOI.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Requested start date as ee.Date.
        end_date: Requested end date as ee.Date.

    Returns:
        None. Raises ValueError if the date range falls outside the available Sentinel-2 SR data extent.
    """
    validate_collection_date_range(
        collection_id=S2_SR_COLLECTION,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor_label="Sentinel-2 SR imagery",
    )


def process_s2_image(
    image: ee.Image,
    indices: list[str] | tuple[str, ...] | None = None,
    domains: list[str] | tuple[str, ...] | None = None,
) -> ee.Image:
    """Select core Sentinel-2 SR bands, apply the reflectance scale factor, and add
    spectral indices.

    Args:
        image: Raw Sentinel-2 SR ee.Image with original band names (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, SCL).
        indices: Explicit index names to append, or None for the default harmonized core.
        domains: Index domains to append (e.g. ['vegetation', 'burn']), or None. See eetools.sensors.indices.calc_indices.

    Returns:
        ee.Image with scaled reflectance bands and appended index bands, preserving system:time_start.
    """
    source = image
    image = select_base_bands(
        source,
        input_bands=[
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B11",
            "B12",
            "SCL",
        ],
    ).multiply(S2_SCALE_FACTOR)

    image = calc_indices(
        image=image, band_map=S2_BAND_MAP, indices=indices, domains=domains
    )
    return image.copyProperties(source, ["system:time_start"])


def get_s2_sr_collection(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    apply_water_masking: bool = True,
    indices: list[str] | tuple[str, ...] | None = None,
    domains: list[str] | tuple[str, ...] | None = None,
) -> ee.ImageCollection:
    """Build a cloud- and water-masked Sentinel-2 SR collection with spectral indices.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.
        apply_water_masking: If True, applies a spectral water mask derived from the collection median (default True).
        indices: Explicit index names each image should contain, or None for the default harmonized core.
        domains: Index domains to include (e.g. ['vegetation', 'soil']), or None. See eetools.sensors.indices.calc_indices.

    Returns:
        ee.ImageCollection with SR bands scaled to [0, 1] and index bands appended.

    Raises:
        ValueError: if apply_water_masking is True but the index selection omits NDVI or MNDWI
            (both required by the spectral water mask).
    """
    validate_s2_sr_date_range(aoi, start_date, end_date)
    if apply_water_masking:
        validate_water_mask_selection(S2_BAND_MAP, indices, domains)

    s2_cloudfree = build_cloudfree_s2sr_col(aoi, start_date, end_date)
    s2_processed = s2_cloudfree.map(
        lambda img: process_s2_image(img, indices=indices, domains=domains)
    )

    if not apply_water_masking:
        return s2_processed

    non_water_mask = build_s2_non_water_mask(s2_processed)
    return s2_processed.map(lambda img: apply_water_mask(img, non_water_mask))
