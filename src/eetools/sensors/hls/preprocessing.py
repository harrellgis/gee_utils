import ee

from eetools.constants import (
    HLS_BAND_MAP,
    HLS_L30_COLLECTION,
    HLS_S30_COLLECTION,
)
from eetools.sensors.hls.masking import (
    apply_water_mask,
    build_cloudfree_hls_l30_col,
    build_cloudfree_hls_s30_col,
    build_hls_non_water_mask,
)
from eetools.sensors.indices import calc_indices, select_base_bands
from eetools.sensors.masking import validate_water_mask_selection
from eetools.utils import validate_collection_date_range


def validate_hls_date_range(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> None:
    """Validate that the requested date window overlaps available HLS (L30 and S30)
    imagery over the AOI.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Requested start date as ee.Date.
        end_date: Requested end date as ee.Date.

    Returns:
        None. Raises ValueError if the date range falls outside the available HLS data extent.
    """
    validate_collection_date_range(
        collection_id=[HLS_L30_COLLECTION, HLS_S30_COLLECTION],
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor_label="HLS imagery",
    )


def add_native_crs(image: ee.Image, reference_band: str = "RED") -> ee.Image:
    """Add the native CRS of a reference band as a 'native_crs' image property.

    Args:
        image: ee.Image to annotate.
        reference_band: Name of the band whose projection CRS is stored as metadata (default 'RED').

    Returns:
        ee.Image with 'native_crs' property set to the CRS string of the reference band's projection.
    """
    image = ee.Image(image)
    native_crs = image.select(reference_band).projection().crs()
    return ee.Image(image.set("native_crs", native_crs))


def filter_hls_by_native_crs(
    collection: ee.ImageCollection,
    target_crs: str,
    reference_band: str = "RED",
) -> ee.ImageCollection:
    """Keep only HLS images whose native projection matches the expected CRS string.

    Args:
        collection: ee.ImageCollection of HLS images to filter.
        target_crs: Expected CRS string (e.g. 'EPSG:32736') used to retain only on-tile images.
        reference_band: Name of the band used to read each image's native CRS (default 'RED').

    Returns:
        ee.ImageCollection containing only images whose reference band projection matches target_crs.
    """
    return (
        ee.ImageCollection(collection)
        .map(lambda img: add_native_crs(img, reference_band=reference_band))
        .filter(ee.Filter.eq("native_crs", target_crs))
    )


def harmonize_hls_l30_bands(image: ee.Image) -> ee.Image:
    """Rename HLSL30 source bands to the common merged HLS band schema (BLUE, GREEN,
    RED, NIR, SWIR1, SWIR2).

    Args:
        image: HLSL30 ee.Image with original band names B2, B3, B4, B5, B6, B7.

    Returns:
        ee.Image with bands renamed to BLUE, GREEN, RED, NIR, SWIR1, SWIR2, preserving system:time_start and system:index.
    """
    source = image
    image = select_base_bands(
        source,
        input_bands=["B2", "B3", "B4", "B5", "B6", "B7"],
        output_bands=["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
    )
    return ee.Image(image.copyProperties(source, ["system:time_start", "system:index"]))


def harmonize_hls_s30_bands(image: ee.Image) -> ee.Image:
    """Rename HLSS30 source bands to the common merged HLS band schema (BLUE, GREEN,
    RED, NIR, SWIR1, SWIR2).

    Args:
        image: HLSS30 ee.Image with original band names B2, B3, B4, B8A, B11, B12.

    Returns:
        ee.Image with bands renamed to BLUE, GREEN, RED, NIR, SWIR1, SWIR2, preserving system:time_start and system:index.
    """
    source = image
    image = select_base_bands(
        source,
        input_bands=["B2", "B3", "B4", "B8A", "B11", "B12"],
        output_bands=["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
    )
    return ee.Image(image.copyProperties(source, ["system:time_start", "system:index"]))


def process_hls_l30_image(
    image: ee.Image,
    indices: list[str] | tuple[str, ...] | None = None,
    domains: list[str] | tuple[str, ...] | None = None,
) -> ee.Image:
    """Rename HLSL30 bands to the common schema and add spectral indices.

    Args:
        image: HLSL30 ee.Image with original band names as supplied by the HLS collection.
        indices: Explicit index names to append, or None for the default harmonized core.
        domains: Index domains to append, or None. See eetools.sensors.indices.calc_indices.

    Returns:
        ee.Image with harmonized band names and appended index bands, preserving system:time_start and system:index.
    """
    source = image
    image = harmonize_hls_l30_bands(source)
    image = calc_indices(
        image=image, band_map=HLS_BAND_MAP, indices=indices, domains=domains
    )
    return image.copyProperties(source, ["system:time_start", "system:index"])


def process_hls_s30_image(
    image: ee.Image,
    indices: list[str] | tuple[str, ...] | None = None,
    domains: list[str] | tuple[str, ...] | None = None,
) -> ee.Image:
    """Rename HLSS30 bands to the common schema and add spectral indices.

    Args:
        image: HLSS30 ee.Image with original band names as supplied by the HLS collection.
        indices: Explicit index names to append, or None for the default harmonized core.
        domains: Index domains to append, or None. See eetools.sensors.indices.calc_indices.

    Returns:
        ee.Image with harmonized band names and appended index bands, preserving system:time_start and system:index.
    """
    source = image
    image = harmonize_hls_s30_bands(source)
    image = calc_indices(
        image=image, band_map=HLS_BAND_MAP, indices=indices, domains=domains
    )
    return image.copyProperties(source, ["system:time_start", "system:index"])


def get_hls_l30_collection(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    indices: list[str] | tuple[str, ...] | None = None,
    domains: list[str] | tuple[str, ...] | None = None,
) -> ee.ImageCollection:
    """Build a processed HLSL30 collection with cloud masking and spectral indices
    applied.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.
        indices: Explicit index names each image should contain, or None for the default harmonized core.
        domains: Index domains to include, or None. See eetools.sensors.indices.calc_indices.

    Returns:
        ee.ImageCollection of cloud-masked HLSL30 images with harmonized bands and index bands appended.
    """
    return build_cloudfree_hls_l30_col(aoi, start_date, end_date).map(
        lambda img: process_hls_l30_image(img, indices=indices, domains=domains)
    )


def get_hls_s30_collection(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    indices: list[str] | tuple[str, ...] | None = None,
    domains: list[str] | tuple[str, ...] | None = None,
) -> ee.ImageCollection:
    """Build a processed HLSS30 collection with cloud masking and spectral indices
    applied.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.
        indices: Explicit index names each image should contain, or None for the default harmonized core.
        domains: Index domains to include, or None. See eetools.sensors.indices.calc_indices.

    Returns:
        ee.ImageCollection of cloud-masked HLSS30 images with harmonized bands and index bands appended.
    """
    return build_cloudfree_hls_s30_col(aoi, start_date, end_date).map(
        lambda img: process_hls_s30_image(img, indices=indices, domains=domains)
    )


def get_hls_merged_collection(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    apply_water_masking: bool = True,
    target_crs: str | None = None,
    indices: list[str] | tuple[str, ...] | None = None,
    domains: list[str] | tuple[str, ...] | None = None,
) -> ee.ImageCollection:
    """Build a merged processed HLS collection (L30 and S30) with shared bands, indices,
    and optional water masking.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.
        apply_water_masking: If True, applies a spectral water mask derived from the merged collection median (default True).
        target_crs: Optional CRS string (e.g. 'EPSG:32736') to filter out off-tile HLS images with incorrect footprints; no CRS filter applied if None.
        indices: Explicit index names each image should contain, or None for the default harmonized core.
        domains: Index domains to include, or None. See eetools.sensors.indices.calc_indices.

    Returns:
        ee.ImageCollection of merged L30 and S30 images sorted by time, with harmonized bands and index bands appended.

    Raises:
        ValueError: if apply_water_masking is True but the index selection omits NDVI or MNDWI.
    """
    validate_hls_date_range(aoi, start_date, end_date)
    if apply_water_masking:
        validate_water_mask_selection(HLS_BAND_MAP, indices, domains)

    hls_l30 = get_hls_l30_collection(
        aoi, start_date, end_date, indices=indices, domains=domains
    )
    hls_s30 = get_hls_s30_collection(
        aoi, start_date, end_date, indices=indices, domains=domains
    )
    merged = hls_l30.merge(hls_s30).sort("system:time_start")

    if target_crs is not None:
        merged = filter_hls_by_native_crs(
            collection=merged, target_crs=target_crs, reference_band="RED"
        )

    if not apply_water_masking:
        return merged

    non_water_mask = build_hls_non_water_mask(merged)
    return merged.map(lambda img: ee.Image(apply_water_mask(img, non_water_mask)))
