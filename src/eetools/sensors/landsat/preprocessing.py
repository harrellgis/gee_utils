import ee

from eetools.constants import (
    L5_SR_COLLECTION,
    L7_SR_COLLECTION,
    L8_ADD_OFFSET,
    L8_BAND_MAP,
    L8_SCALE_FACTOR,
    L8_SR_COLLECTION,
    L9_BAND_MAP,
    L9_BANDS,
    L9_SR_COLLECTION,
    LANDSAT_C2_ADD_OFFSET,
    LANDSAT_C2_SCALE_FACTOR,
    TM_BAND_MAP,
    TM_BANDS,
)
from eetools.sensors.indices import calc_indices, select_base_bands
from eetools.sensors.landsat.masking import (
    apply_water_mask,
    build_cloudfree_l8sr_col,
    build_cloudfree_landsat_col,
    build_l8_non_water_mask,
    build_landsat_non_water_mask,
)
from eetools.utils import validate_collection_date_range


def validate_l8_sr_date_range(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> None:
    """Validate that the requested date window overlaps available Landsat 8 SR imagery
    over the AOI.

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
    """Select core Landsat 8 SR bands, apply the scale factor and additive offset, and
    add spectral indices.

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


# --------------------------------------------------------------------------- #
# Generic Landsat C2 L2 core (shared by Landsat 5 / 7 / 9)
# --------------------------------------------------------------------------- #
def process_landsat_image(
    image: ee.Image, source_bands: list[str], band_map: dict
) -> ee.Image:
    """Select reflective bands, apply C2 L2 scale/offset, and append spectral indices.

    Sensor-agnostic over the Landsat C2 L2 family — pass the sensor's reflective band
    list and band map (OLI and TM/ETM+ use different band numbering).

    Args:
        image: Raw Landsat C2 L2 ee.Image.
        source_bands: Reflective bands to select (L8_BANDS/L9_BANDS for OLI, TM_BANDS for TM/ETM+).
        band_map: Logical-key -> band-name map that drives the index functions for this sensor.

    Returns:
        ee.Image with calibrated reflectance bands and appended index bands, preserving system:time_start.
    """
    source = image
    image = select_base_bands(source, input_bands=source_bands)
    image = image.multiply(LANDSAT_C2_SCALE_FACTOR).add(LANDSAT_C2_ADD_OFFSET)
    image = calc_indices(image=image, band_map=band_map, include_ndre=False)
    return image.copyProperties(source, ["system:time_start"])


def get_landsat_sr_collection(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    collection_id: str,
    source_bands: list[str],
    band_map: dict,
    sensor_label: str,
    apply_water_masking: bool = True,
) -> ee.ImageCollection:
    """Build a cloud-masked Landsat C2 L2 SR collection with spectral indices (generic).

    Validates the date range, builds the cloud/shadow-masked collection, applies the
    scale factor/offset and index recipe, and optionally applies a spectral water mask.
    The per-sensor get_l5/l7/l9 builders are thin wrappers over this; the sensors differ
    only in collection ID, reflective band list, and band map.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.
        collection_id: Landsat C2 L2 collection asset ID.
        source_bands: Reflective bands to select for this sensor.
        band_map: Logical-key -> band-name map for this sensor.
        sensor_label: Human-readable sensor name used in date-range error messages.
        apply_water_masking: If True, apply a spectral water mask derived from the collection median (default True).

    Returns:
        ee.ImageCollection with calibrated SR bands and index bands appended.
    """
    validate_collection_date_range(
        collection_id=collection_id,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor_label=sensor_label,
    )

    cloudfree = build_cloudfree_landsat_col(aoi, start_date, end_date, collection_id)
    processed = cloudfree.map(
        lambda img: process_landsat_image(img, source_bands, band_map)
    )

    if not apply_water_masking:
        return processed

    non_water_mask = build_landsat_non_water_mask(processed, band_map["nir"])
    return processed.map(lambda img: apply_water_mask(img, non_water_mask))


# --------------------------------------------------------------------------- #
# Landsat 9 (OLI-2) — band-for-band identical to Landsat 8
# --------------------------------------------------------------------------- #
def validate_l9_sr_date_range(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> None:
    """Validate that the requested window overlaps available Landsat 9 SR imagery.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Requested start date as ee.Date.
        end_date: Requested end date as ee.Date.

    Returns:
        None. Raises ValueError if the date range falls outside the Landsat 9 data extent.
    """
    validate_collection_date_range(
        collection_id=L9_SR_COLLECTION,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor_label="Landsat 9 SR imagery",
    )


def process_l9_image(image: ee.Image) -> ee.Image:
    """Select Landsat 9 (OLI) bands, apply the scale/offset, and add spectral indices.

    Args:
        image: Raw Landsat 9 C2 L2 ee.Image (OLI band layout, SR_B2-SR_B7).

    Returns:
        ee.Image with calibrated reflectance bands and appended index bands, preserving system:time_start.
    """
    return process_landsat_image(image, L9_BANDS, L9_BAND_MAP)


def get_l9_sr_collection(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    apply_water_masking: bool = True,
) -> ee.ImageCollection:
    """Build a cloud- and (optionally) water-masked Landsat 9 SR collection with indices.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.
        apply_water_masking: If True, applies a spectral water mask derived from the collection median (default True).

    Returns:
        ee.ImageCollection with calibrated SR bands and index bands appended.
    """
    return get_landsat_sr_collection(
        aoi,
        start_date,
        end_date,
        collection_id=L9_SR_COLLECTION,
        source_bands=L9_BANDS,
        band_map=L9_BAND_MAP,
        sensor_label="Landsat 9 SR imagery",
        apply_water_masking=apply_water_masking,
    )


# --------------------------------------------------------------------------- #
# Landsat 7 (ETM+) — TM/ETM+ band layout (shifted vs OLI)
# --------------------------------------------------------------------------- #
def validate_l7_sr_date_range(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> None:
    """Validate that the requested window overlaps available Landsat 7 SR imagery.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Requested start date as ee.Date.
        end_date: Requested end date as ee.Date.

    Returns:
        None. Raises ValueError if the date range falls outside the Landsat 7 data extent.
    """
    validate_collection_date_range(
        collection_id=L7_SR_COLLECTION,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor_label="Landsat 7 SR imagery",
    )


def process_l7_image(image: ee.Image) -> ee.Image:
    """Select Landsat 7 (ETM+) bands, apply the scale/offset, and add spectral indices.

    Note: ETM+ band numbering differs from OLI (NIR=SR_B4, Red=SR_B3, SWIR1=SR_B5); the
    TM band map wires the index functions to the correct bands.

    Args:
        image: Raw Landsat 7 C2 L2 ee.Image (TM/ETM+ band layout, SR_B1-SR_B5, SR_B7).

    Returns:
        ee.Image with calibrated reflectance bands and appended index bands, preserving system:time_start.
    """
    return process_landsat_image(image, TM_BANDS, TM_BAND_MAP)


def get_l7_sr_collection(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    apply_water_masking: bool = True,
) -> ee.ImageCollection:
    """Build a cloud- and (optionally) water-masked Landsat 7 SR collection with indices.

    Note: ETM+ scenes acquired after 2003-05-31 have SLC-off wedge gaps (~22% missing);
    composite multiple dates for gap-free coverage.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.
        apply_water_masking: If True, applies a spectral water mask derived from the collection median (default True).

    Returns:
        ee.ImageCollection with calibrated SR bands and index bands appended.
    """
    return get_landsat_sr_collection(
        aoi,
        start_date,
        end_date,
        collection_id=L7_SR_COLLECTION,
        source_bands=TM_BANDS,
        band_map=TM_BAND_MAP,
        sensor_label="Landsat 7 SR imagery",
        apply_water_masking=apply_water_masking,
    )


# --------------------------------------------------------------------------- #
# Landsat 5 (TM) — TM/ETM+ band layout (shifted vs OLI)
# --------------------------------------------------------------------------- #
def validate_l5_sr_date_range(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> None:
    """Validate that the requested window overlaps available Landsat 5 SR imagery.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Requested start date as ee.Date.
        end_date: Requested end date as ee.Date.

    Returns:
        None. Raises ValueError if the date range falls outside the Landsat 5 data extent (archive ends 2012).
    """
    validate_collection_date_range(
        collection_id=L5_SR_COLLECTION,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor_label="Landsat 5 SR imagery",
    )


def process_l5_image(image: ee.Image) -> ee.Image:
    """Select Landsat 5 (TM) bands, apply the scale/offset, and add spectral indices.

    Note: TM band numbering differs from OLI (NIR=SR_B4, Red=SR_B3, SWIR1=SR_B5); the TM
    band map wires the index functions to the correct bands.

    Args:
        image: Raw Landsat 5 C2 L2 ee.Image (TM band layout, SR_B1-SR_B5, SR_B7).

    Returns:
        ee.Image with calibrated reflectance bands and appended index bands, preserving system:time_start.
    """
    return process_landsat_image(image, TM_BANDS, TM_BAND_MAP)


def get_l5_sr_collection(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    apply_water_masking: bool = True,
) -> ee.ImageCollection:
    """Build a cloud- and (optionally) water-masked Landsat 5 SR collection with indices.

    The Landsat 5 TM archive runs 1984-2012; choose a date window within that range.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.
        apply_water_masking: If True, applies a spectral water mask derived from the collection median (default True).

    Returns:
        ee.ImageCollection with calibrated SR bands and index bands appended.
    """
    return get_landsat_sr_collection(
        aoi,
        start_date,
        end_date,
        collection_id=L5_SR_COLLECTION,
        source_bands=TM_BANDS,
        band_map=TM_BAND_MAP,
        sensor_label="Landsat 5 SR imagery",
        apply_water_masking=apply_water_masking,
    )
