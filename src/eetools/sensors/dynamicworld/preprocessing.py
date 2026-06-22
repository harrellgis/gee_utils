import ee

from eetools.constants import DW_CLASSES, DW_COLLECTION
from eetools.sensors.dynamicworld.masking import mask_to_cover_types
from eetools.utils import validate_collection_date_range


def validate_dynamic_world_date_range(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> None:
    """Validate that the requested date window overlaps available Dynamic World imagery
    over the AOI.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Requested start date as ee.Date.
        end_date: Requested end date as ee.Date.

    Returns:
        None. Raises ValueError if the date range falls outside the available Dynamic World data extent.
    """
    validate_collection_date_range(
        collection_id=DW_COLLECTION,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor_label="Dynamic World imagery",
    )


def get_dynamic_world_collection(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    cover_types: list[int] | None = None,
) -> ee.ImageCollection:
    """Build a Dynamic World LULC collection over the AOI, optionally limited to cover types.

    Dynamic World is a per-Sentinel-2-acquisition product (10 m, cloud masking built in),
    carrying nine class-probability bands plus an integer ``label`` argmax band. No extra
    cloud masking or indexing is applied. Note the ``label`` band is unstable per
    acquisition — composite the probability bands (e.g. ``.mean()``) for stable LULC or
    landscape metrics rather than relying on a single ``label`` image.

    When ``cover_types`` is provided, each image is masked to only those ``label`` classes
    (across all bands), so the returned collection contains only the requested land-cover
    types; when None, all classes are kept.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.
        cover_types: Optional non-empty list of Dynamic World class integers (0-8) to keep; pixels of any other class are masked. None keeps all classes. See constants.DW_CLASSES for the code-to-name mapping.

    Returns:
        ee.ImageCollection of Dynamic World images filtered to the AOI and date range, masked to ``cover_types`` when provided.

    Raises:
        ValueError: If cover_types is an empty list or contains codes outside 0-8, or if the date range falls outside the available data extent.
    """
    if cover_types is not None:
        if len(cover_types) == 0:
            raise ValueError(
                "cover_types must be a non-empty list of class integers (0-8), or "
                "None to keep all classes."
            )
        invalid = sorted({c for c in cover_types if c not in DW_CLASSES})
        if invalid:
            raise ValueError(
                f"cover_types contains invalid Dynamic World class codes {invalid}; "
                f"valid codes are {sorted(DW_CLASSES)}."
            )

    validate_dynamic_world_date_range(aoi, start_date, end_date)

    col = (
        ee.ImageCollection(DW_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    if cover_types is not None:
        col = col.map(lambda img: mask_to_cover_types(img, cover_types))
    return col
