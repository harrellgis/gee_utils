import ee

from eetools.constants import (
    S1_DEFAULT_INSTRUMENT_MODE,
    S1_DEFAULT_ORBIT_PASS,
    S1_DEFAULT_POLARIZATIONS,
    S1_GRD_COLLECTION,
)
from eetools.sensors.sentinel1.masking import apply_speckle_filter, mask_edges
from eetools.utils import validate_collection_date_range


def validate_s1_grd_date_range(
    aoi: ee.Geometry, start_date: ee.Date, end_date: ee.Date
) -> None:
    """Validate that the requested date window overlaps available Sentinel-1 GRD imagery
    over the AOI.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Requested start date as ee.Date.
        end_date: Requested end date as ee.Date.

    Returns:
        None. Raises ValueError if the date range falls outside the available Sentinel-1 GRD data extent.
    """
    validate_collection_date_range(
        collection_id=S1_GRD_COLLECTION,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor_label="Sentinel-1 GRD imagery",
    )


def get_s1_grd_collection(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    polarizations: list[str] | None = None,
    instrument_mode: str = S1_DEFAULT_INSTRUMENT_MODE,
    orbit_pass: str | None = S1_DEFAULT_ORBIT_PASS,
    apply_edge_masking: bool = True,
    apply_speckle_filtering: bool = True,
) -> ee.ImageCollection:
    """Build a homogeneous, edge-masked, speckle-filtered Sentinel-1 GRD collection.

    COPERNICUS/S1_GRD is a heterogeneous collection — scenes vary in polarization,
    instrument mode, resolution, and orbit pass. This filters to a single consistent
    set (requested polarizations present, one instrument mode, one orbit pass) before
    selecting the backscatter bands, because mixing them produces band-presence errors
    and nonsensical composites. ASCENDING and DESCENDING passes have different look
    geometry and must never be mixed in a time series, so a single ``orbit_pass`` is
    pinned by default.

    No spectral indices are added (SAR backscatter is not reflectance). Backscatter is
    in dB; average in linear power space (or use COPERNICUS/S1_GRD_FLOAT) rather than
    averaging dB directly.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as ee.Date.
        end_date: Collection end date as ee.Date.
        polarizations: Polarization bands that must be present and are selected (default ['VV', 'VH']).
        instrument_mode: Instrument mode to keep — 'IW', 'EW', or 'SM' (default 'IW').
        orbit_pass: Orbit pass to keep — 'ASCENDING' or 'DESCENDING'; pass None to disable the filter (advanced use only, since mixing passes corrupts a time series). Default 'DESCENDING'.
        apply_edge_masking: If True, mask low-backscatter scene edges (default True).
        apply_speckle_filtering: If True, apply a focal-median speckle filter (default True).

    Returns:
        ee.ImageCollection of Sentinel-1 GRD images carrying only the requested polarization bands (dB), filtered to one mode and orbit pass, with edge masking and speckle filtering applied per the flags.
    """
    pols = (
        list(polarizations)
        if polarizations is not None
        else list(S1_DEFAULT_POLARIZATIONS)
    )

    validate_s1_grd_date_range(aoi, start_date, end_date)

    col = (
        ee.ImageCollection(S1_GRD_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    for pol in pols:
        col = col.filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol))
    col = col.filter(ee.Filter.eq("instrumentMode", instrument_mode))
    if orbit_pass is not None:
        col = col.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))

    col = col.select(pols)

    if apply_edge_masking:
        col = col.map(mask_edges)
    if apply_speckle_filtering:
        col = col.map(apply_speckle_filter)

    return col
