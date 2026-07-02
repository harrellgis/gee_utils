from collections.abc import Callable

import ee

from eetools.compositing import (
    build_period_composites,
    build_seasonal_composites,
)
from eetools.io import export_table_to_drive
from eetools.vectors import get_sites_geometry
from eetools.zonal import image_collection_to_region_stats_fc


def run_site_timeseries(
    sites_fc: ee.FeatureCollection,
    collection_builder: Callable[
        [ee.Geometry, str | ee.Date, str | ee.Date], ee.ImageCollection
    ],
    bands: list[str],
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    export_folder: str,
    file_prefix: str,
    temporal_scale: str = "annual",
    composite_stat: str = "median",
    reducers: ee.Reducer | None = None,
    scale: int = 5566,
    image_properties: list[str] | None = None,
    tile_scale: int = 4,
) -> ee.FeatureCollection:
    """Run the full site spectral timeseries workflow for annual or monthly compositing.

    Derives the AOI from the sites FeatureCollection, builds the sensor collection via
    the caller-supplied builder, composites to the requested temporal window, reduces
    per region, exports the result table to Drive, and returns the stats FeatureCollection.

    The ``collection_builder`` should be a partially-applied sensor function whose
    signature matches ``(aoi, start_date, end_date) -> ee.ImageCollection``:

    .. code-block:: python

        from functools import partial
        from eetools.sensors.sentinel import get_s2_sr_collection
        from eetools.workflows import run_site_timeseries

        builder = partial(
            get_s2_sr_collection,
            apply_water_masking=True,
            indices=["NDVI", "EVI", "NBR"],
        )
        stats = run_site_timeseries(
            sites_fc=my_sites,
            collection_builder=builder,
            bands=["NDVI", "EVI", "NBR"],
            start_date="2020-01-01",
            end_date="2023-01-01",
            export_folder="my_drive_folder",
            file_prefix="s2_annual_ndvi",
        )

    Args:
        sites_fc: ee.FeatureCollection of site polygon features. Each feature must
            carry a ``site_name`` property (or the property named by the
            ``reduceRegions`` defaults in ``image_collection_to_region_stats_fc``).
            The combined geometry is used as the AOI for the collection builder.
        collection_builder: Callable with signature
            ``(aoi: ee.Geometry, start_date: str | ee.Date, end_date: str | ee.Date)
            -> ee.ImageCollection``. Typically a partially-applied ``get_*_collection``
            function with sensor-specific kwargs already bound (via
            ``functools.partial``). Must return a collection whose images carry the
            bands named in ``bands``.
        bands: List of band names to select for compositing and zonal reduction.
            These must be present in the collection the builder produces (e.g. index
            band names computed by the builder's ``indices=`` kwarg).
        start_date: Collection start date as a string ('YYYY-MM-DD') or ee.Date.
            Passed to both the builder and ``build_period_composites``.
        end_date: Collection end date as a string or ee.Date.
        export_folder: Google Drive folder name to write the output CSV into.
        file_prefix: Filename prefix (without extension) and EE task description.
        temporal_scale: Temporal aggregation window; 'annual' or 'monthly'
            (default 'annual').
        composite_stat: Statistic used to combine images within each window; one of
            'mean', 'median', or 'sum' (default 'median').
        reducers: ee.Reducer applied per region per composite image; defaults to
            mean combined with stdDev and minMax (as defined in
            ``image_collection_to_region_stats_fc``).
        scale: Pixel scale in metres for the ``reduceRegions`` call (default 5566).
        image_properties: List of image property names to copy onto each output
            feature; when None (default) all scalar image properties are copied.
        tile_scale: EE tileScale parameter to avoid memory limits (default 4).

    Returns:
        ee.FeatureCollection of per-region statistics, one feature per site per
        composite period, with composite metadata properties and per-band reduction
        results attached. A batch table-export task to Drive is started as a side
        effect.

    Raises:
        ValueError: If ``composite_stat`` is not 'mean', 'median', or 'sum', or if
            ``temporal_scale`` is not 'annual' or 'monthly'.
    """
    aoi = get_sites_geometry(sites_fc)
    collection = collection_builder(aoi, start_date, end_date)
    composites = build_period_composites(
        collection, bands, start_date, end_date, temporal_scale, composite_stat
    )
    stats_fc = image_collection_to_region_stats_fc(
        composites,
        sites_fc,
        bands,
        scale,
        reducers,
        image_properties,
        tile_scale,
    )
    export_table_to_drive(
        collection=stats_fc,
        description=file_prefix,
        folder=export_folder,
        fileNamePrefix=file_prefix,
    )
    return stats_fc


def run_seasonal_site_timeseries(
    sites_fc: ee.FeatureCollection,
    collection_builder: Callable[
        [ee.Geometry, str | ee.Date, str | ee.Date], ee.ImageCollection
    ],
    bands: list[str],
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    start_year: int,
    end_year: int,
    season_months: tuple[int, int],
    season_name: str,
    export_folder: str,
    file_prefix: str,
    composite_stat: str = "median",
    reducers: ee.Reducer | None = None,
    scale: int = 5566,
    image_properties: list[str] | None = None,
    tile_scale: int = 4,
) -> ee.FeatureCollection:
    """Run the full site spectral timeseries workflow for seasonal compositing.

    Sibling of :func:`run_site_timeseries` for the seasonal case. The raw
    collection is filtered to ``[start_date, end_date)`` first (to narrow the
    EE scan), then :func:`~eetools.compositing.build_seasonal_composites` slices
    it into per-year windows covering ``season_months`` within each year in
    ``[start_year, end_year]``.

    ``start_date``/``end_date`` should be at least as wide as the
    ``start_year``/``end_year`` + ``season_months`` window. Setting them wider
    (e.g. pulling the full archive) is safe and carries no penalty — the seasonal
    windowing ignores images outside the requested months.

    Args:
        sites_fc: ee.FeatureCollection of site polygon features.
        collection_builder: Callable with signature
            ``(aoi, start_date, end_date) -> ee.ImageCollection``.
        bands: List of band names to select for compositing and zonal reduction.
        start_date: Raw collection start date; passed directly to the builder.
        end_date: Raw collection end date; passed directly to the builder.
        start_year: First year of the seasonal composite series, inclusive.
        end_year: Last year of the seasonal composite series, inclusive.
        season_months: ``(start_month, end_month)`` as 1-based integers, both
            inclusive and in the same calendar year (e.g. ``(3, 5)`` for
            March–May). Cross-year seasons are not supported.
        season_name: Label attached as the ``season`` property on each composite
            (e.g. 'wet', 'dry').
        export_folder: Google Drive folder name.
        file_prefix: Filename prefix (without extension) and EE task description.
        composite_stat: Statistic to combine images within each season window;
            one of 'mean', 'median', or 'sum' (default 'median').
        reducers: ee.Reducer applied per region; defaults to mean + stdDev + minMax.
        scale: Pixel scale in metres (default 5566).
        image_properties: Image properties to copy onto each output feature; None
            copies all scalar properties.
        tile_scale: EE tileScale (default 4).

    Returns:
        ee.FeatureCollection of per-region per-year seasonal statistics, each
        carrying ``year``, ``season``, ``season_months``, ``composite_stat``, and
        per-band reduction results. A batch table-export task is started as a side
        effect.

    Raises:
        ValueError: If ``composite_stat`` is not 'mean', 'median', or 'sum', or if
            ``season_months`` does not satisfy ``1 ≤ start ≤ end ≤ 12``.
    """
    aoi = get_sites_geometry(sites_fc)
    collection = collection_builder(aoi, start_date, end_date)
    composites = build_seasonal_composites(
        collection,
        bands,
        start_year,
        end_year,
        season_months,
        season_name,
        composite_stat,
    )
    stats_fc = image_collection_to_region_stats_fc(
        composites,
        sites_fc,
        bands,
        scale,
        reducers,
        image_properties,
        tile_scale,
    )
    export_table_to_drive(
        collection=stats_fc,
        description=file_prefix,
        folder=export_folder,
        fileNamePrefix=file_prefix,
    )
    return stats_fc
