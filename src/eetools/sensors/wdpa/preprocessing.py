import ee

from eetools.constants import (
    WDPA_COUNTRY_FIELD,
    WDPA_ID_FIELD,
    WDPA_POLYGONS_COLLECTION,
)
from eetools.io import export_table_to_drive


def get_wdpa_collection(
    country: str | None = None,
    identifier: int | float | None = None,
) -> ee.FeatureCollection:
    """Load the WDPA protected-area polygons, optionally filtered by country and/or PA id.

    WDPA is a vector FeatureCollection (UNEP-WCMC / IUCN). When ``country`` is given it is
    matched against the ISO3 country field; when ``identifier`` is given it is matched
    against the site id (SITE_ID in the live asset — the value historically called
    WDPAID). Both filters are applied when both are provided; if neither is given the full
    global collection is returned (lazy — it is not materialized until used).

    NOTE (license): WDPA is No-Commercial-Use without prior written UNEP-WCMC permission,
    and attribution is mandatory — flag before any paid/commercial deliverable.

    Args:
        country: ISO 3166-3 alpha-3 country code to filter on (e.g. 'BWA', 'KEN'); no country filter if None.
        identifier: Site id (SITE_ID, the value historically called WDPAID) of a specific protected area to filter on; no id filter if None.

    Returns:
        ee.FeatureCollection of WDPA polygons filtered by the provided criteria.
    """
    fc = ee.FeatureCollection(WDPA_POLYGONS_COLLECTION)
    if country is not None:
        fc = fc.filter(ee.Filter.eq(WDPA_COUNTRY_FIELD, country))
    if identifier is not None:
        fc = fc.filter(ee.Filter.eq(WDPA_ID_FIELD, identifier))
    return fc


def get_wdpa_in_aoi(aoi: ee.Geometry) -> ee.FeatureCollection:
    """Load the WDPA protected-area polygons intersecting an AOI.

    Args:
        aoi: Area of interest as ee.Geometry; features intersecting it are returned.

    Returns:
        ee.FeatureCollection of WDPA polygons whose geometry intersects the AOI.
    """
    return ee.FeatureCollection(WDPA_POLYGONS_COLLECTION).filterBounds(aoi)


def export_wdpa_in_aoi_to_drive(
    aoi: ee.Geometry,
    folder: str,
    file_prefix: str,
    file_format: str = "SHP",
) -> ee.FeatureCollection:
    """Load WDPA polygons within an AOI and export them to Google Drive as a vector file.

    Convenience wrapper over get_wdpa_in_aoi + eetools.io.export_table_to_drive: it builds
    the AOI-filtered protected-area collection, starts a Drive table-export task, and
    returns the collection.

    Args:
        aoi: Area of interest as ee.Geometry.
        folder: Google Drive folder name to write the file into.
        file_prefix: Filename prefix (without extension) and task description.
        file_format: Vector output format, e.g. 'SHP', 'GeoJSON', 'CSV' (default 'SHP').

    Returns:
        ee.FeatureCollection of the AOI-filtered WDPA polygons (a batch export task to Drive is started as a side effect).
    """
    pas = get_wdpa_in_aoi(aoi)
    export_table_to_drive(
        collection=pas,
        description=file_prefix,
        folder=folder,
        fileNamePrefix=file_prefix,
        fileFormat=file_format,
    )
    return pas


def export_wdpa_to_drive(
    folder: str,
    file_prefix: str,
    country: str | None = None,
    identifier: int | float | None = None,
    file_format: str = "SHP",
) -> ee.FeatureCollection:
    """Export WDPA protected areas for a country and/or specific PA id to Google Drive.

    Convenience wrapper over get_wdpa_collection + eetools.io.export_table_to_drive. At
    least one of ``country`` or ``identifier`` must be given — exporting the full global
    WDPA (>200k features) is refused to avoid an accidental enormous export.

    Args:
        folder: Google Drive folder name to write the file into.
        file_prefix: Filename prefix (without extension) and task description.
        country: ISO 3166-3 alpha-3 country code to filter on (e.g. 'BWA').
        identifier: Site id (SITE_ID, the value historically called WDPAID) of a specific protected area to filter on.
        file_format: Vector output format, e.g. 'SHP', 'GeoJSON', 'CSV' (default 'SHP').

    Returns:
        ee.FeatureCollection of the filtered WDPA polygons (a batch export task to Drive is started as a side effect).

    Raises:
        ValueError: If both country and identifier are None.
    """
    if country is None and identifier is None:
        raise ValueError(
            "Provide at least one of country (ISO3) or identifier (WDPAID) to bound the "
            "export; refusing to export the full global WDPA collection."
        )

    pas = get_wdpa_collection(country=country, identifier=identifier)
    export_table_to_drive(
        collection=pas,
        description=file_prefix,
        folder=folder,
        fileNamePrefix=file_prefix,
        fileFormat=file_format,
    )
    return pas
