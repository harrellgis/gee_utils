import ee

from eetools.constants import (
    DSWX_HLS_COLLECTION,
    DSWX_HLS_VALID_MAX,
    DSWX_S1_COLLECTION,
    DSWX_S1_VALID_MAX,
)
from eetools.sensors.dswx.masking import mask_invalid_classes
from eetools.utils import validate_collection_date_range


def validate_dswx_hls_date_range(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> None:
    """Validate that the requested date window overlaps available OPERA DSWx-HLS imagery
    over the AOI.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Requested start date as ee.Date.
        end_date: Requested end date as ee.Date.

    Returns:
        None. Raises ValueError if the date range falls outside the available DSWx-HLS data extent (validated observations from April 2023).
    """
    validate_collection_date_range(
        collection_id=DSWX_HLS_COLLECTION,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor_label="OPERA DSWx-HLS imagery",
    )


def validate_dswx_s1_date_range(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> None:
    """Validate that the requested date window overlaps available OPERA DSWx-S1 imagery
    over the AOI.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Requested start date as ee.Date.
        end_date: Requested end date as ee.Date.

    Returns:
        None. Raises ValueError if the date range falls outside the available DSWx-S1 data extent (forward production from September 2024).
    """
    validate_collection_date_range(
        collection_id=DSWX_S1_COLLECTION,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor_label="OPERA DSWx-S1 imagery",
    )


def get_dswx_hls_collection(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    mask_invalid: bool = True,
) -> ee.ImageCollection:
    """Build an OPERA DSWx-HLS (optical) surface-water collection over the AOI.

    DSWx is a pre-classified product (no spectral indexing). When ``mask_invalid`` is
    True, snow (252), cloud (253), and ocean (254) mask classes are masked out across
    all bands so the collection can be composited directly (the catalog example uses
    ``ee.Reducer.mode()``). The primary band is WTR_Water_classification; BWTR_Binary_water
    gives a ready binary water layer. Being optical, this product is cloud/snow-limited —
    use get_dswx_s1_collection for all-weather coverage.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.
        mask_invalid: If True, mask WTR snow/cloud/ocean classes (>= DSWX_HLS_VALID_MAX) across all bands (default True).

    Returns:
        ee.ImageCollection of DSWx-HLS images filtered to the AOI and date range, with invalid mask classes removed when mask_invalid is True.
    """
    validate_dswx_hls_date_range(aoi, start_date, end_date)

    col = (
        ee.ImageCollection(DSWX_HLS_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    if mask_invalid:
        col = col.map(lambda img: mask_invalid_classes(img, DSWX_HLS_VALID_MAX))
    return col


def get_dswx_s1_collection(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    mask_invalid: bool = True,
) -> ee.ImageCollection:
    """Build an OPERA DSWx-S1 (radar) surface-water collection over the AOI.

    DSWx is a pre-classified product (no spectral indexing). Because it is radar-derived,
    it maps water irrespective of cloud or daylight — the advantage over DSWx-HLS for
    flood/storm mapping. When ``mask_invalid`` is True, HAND (250), layover/shadow (251),
    and ocean (254) mask classes are masked out across all bands so the collection can be
    composited directly (the catalog example uses ``ee.Reducer.max()``). Note the mask
    threshold (250) differs from DSWx-HLS (252). Inland open water is mapped only above
    ~3 ha and ~200 m width; smaller/narrower water is missed.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.
        mask_invalid: If True, mask WTR HAND/layover-shadow/ocean classes (>= DSWX_S1_VALID_MAX) across all bands (default True).

    Returns:
        ee.ImageCollection of DSWx-S1 images filtered to the AOI and date range, with invalid mask classes removed when mask_invalid is True.
    """
    validate_dswx_s1_date_range(aoi, start_date, end_date)

    col = (
        ee.ImageCollection(DSWX_S1_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    if mask_invalid:
        col = col.map(lambda img: mask_invalid_classes(img, DSWX_S1_VALID_MAX))
    return col
