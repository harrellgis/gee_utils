from collections.abc import Callable

import ee

from eetools.compositing import (
    build_period_composites,
    build_seasonal_composites,
)
from eetools.constants import (
    BASELINE_CONTINUOUS_LAYERS,
    BASELINE_LAYER_NAMES,
    DEFAULT_BASELINE_EXPORT_SCALE,
    DEFAULT_CRS,
    ESA_CLASS_MAP,
)
from eetools.io import export_image_list_to_drive, export_table_to_drive
from eetools.sensors.bii.preprocessing import get_bii
from eetools.sensors.canopy_height.preprocessing import get_canopy_height
from eetools.sensors.dem.preprocessing import get_terrain
from eetools.sensors.esa.preprocessing import get_land_cover
from eetools.sensors.hansen.preprocessing import get_forest_2000, get_forest_loss_image
from eetools.sensors.isda.preprocessing import get_soil_carbon
from eetools.vectors import get_sites_geometry
from eetools.zonal import image_collection_to_region_stats_fc, summarize_class_areas


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


# --------------------------------------------------------------------------- #
# Baseline assessment                                                         #
# --------------------------------------------------------------------------- #


def build_baseline_layers(aoi: ee.Geometry) -> dict[str, ee.Image]:
    """Build the static baseline layer stack for an AOI: terrain, canopy height,
    land cover, soil carbon, biodiversity intactness, and Hansen forest-change.

    Mirrors the TGBS_Kwale baseline notebook's ``build_baseline_layers``: a fixed
    set of single-purpose static datasets, each clipped to the AOI and returned
    under a stable key so callers can summarize or export any subset. See
    :func:`run_baseline_assessment` for the full build -> summarize -> export
    workflow.

    Args:
        aoi: Area of interest as ee.Geometry; every layer is clipped to it.

    Returns:
        dict mapping BASELINE_LAYER_NAMES keys ('dem', 'slope', 'hillshade',
        'canopy_height', 'land_cover', 'soil_carbon', 'bii_all', 'forest_2000',
        'forest_loss') to single-band ee.Images.
    """
    terrain = get_terrain(aoi=aoi)
    return {
        "dem": terrain.select("elevation"),
        "slope": terrain.select("slope"),
        "hillshade": terrain.select("hillshade"),
        "canopy_height": get_canopy_height(aoi=aoi),
        "land_cover": get_land_cover(aoi=aoi),
        "soil_carbon": get_soil_carbon(aoi=aoi),
        "bii_all": get_bii(aoi, "BII All", "1km"),
        "forest_2000": get_forest_2000(aoi),
        "forest_loss": get_forest_loss_image(aoi),
    }


def summarize_baseline_layers(
    sites_fc: ee.FeatureCollection,
    layers: dict[str, ee.Image],
    scale_continuous: int = 30,
    scale_landcover: int = 10,
    crs: str = DEFAULT_CRS,
    tile_scale: int = 4,
    include_landcover_areas: bool = True,
) -> ee.FeatureCollection:
    """Summarize baseline layers per site: continuous-layer stats plus optional
    land-cover class areas.

    Reduces the continuous layers (BASELINE_CONTINUOUS_LAYERS) over each site
    polygon with mean/median/min/max/stdDev, then — when requested — chains a
    second reduction that sums per-class land-cover area (m^2) onto the same
    features. No property-based join is needed: reduceRegions already extends
    each input feature's existing properties, so passing its own output back in
    as the next reduceRegions call's collection accumulates properties across
    both passes.

    Args:
        sites_fc: ee.FeatureCollection of site polygons to summarize over.
        layers: Layer dict as returned by :func:`build_baseline_layers`; must
            contain every key in BASELINE_CONTINUOUS_LAYERS, plus 'land_cover'
            when include_landcover_areas is True.
        scale_continuous: Pixel scale in metres for the continuous-layer
            reduction (default 30).
        scale_landcover: Pixel scale in metres for the land-cover class-area
            reduction (default 10, matching ESA WorldCover's native resolution).
        crs: Coordinate reference system for both reductions (default
            DEFAULT_CRS).
        tile_scale: EE tileScale parameter to avoid memory limits (default 4).
        include_landcover_areas: If True (default), chain a
            summarize_class_areas call (ESA_CLASS_MAP) onto the continuous
            stats.

    Returns:
        ee.FeatureCollection with one feature per site, carrying its original
        properties plus one mean/median/min/max/stdDev property per continuous
        layer and, when requested, one '<class>_area_m2' property per
        ESA_CLASS_MAP entry.
    """
    continuous_stack = ee.Image.cat(
        [layers[name].rename(name) for name in BASELINE_CONTINUOUS_LAYERS]
    )
    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.median(), sharedInputs=True)
        .combine(ee.Reducer.minMax(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
    )
    stats_fc = continuous_stack.reduceRegions(
        collection=sites_fc,
        reducer=reducer,
        scale=scale_continuous,
        crs=crs,
        tileScale=tile_scale,
    )

    if include_landcover_areas:
        stats_fc = summarize_class_areas(
            regions_fc=stats_fc,
            classified_image=layers["land_cover"],
            class_map=ESA_CLASS_MAP,
            scale=scale_landcover,
            crs=crs,
            tile_scale=tile_scale,
        )

    return stats_fc


def export_baseline_layers(
    layers: dict[str, ee.Image],
    aoi: ee.Geometry,
    export_folder: str,
    layer_names: list[str] | None = None,
    scale_dict: dict[str, int] | None = None,
    crs: str = DEFAULT_CRS,
) -> dict:
    """Export a selected subset of baseline layers to Drive as individual GeoTIFFs.

    Args:
        layers: Layer dict as returned by :func:`build_baseline_layers`.
        aoi: Export region as ee.Geometry, shared by every layer.
        export_folder: Google Drive folder name to write all files into.
        layer_names: Layer keys to export; defaults to every key in
            BASELINE_LAYER_NAMES (all baseline layers).
        scale_dict: Per-layer export scale in metres, keyed by layer name;
            defaults to DEFAULT_BASELINE_EXPORT_SCALE (uniform 30m).
        crs: Coordinate reference system applied to every export (default
            DEFAULT_CRS).

    Returns:
        dict mapping each exported layer name to its started ee.batch.Task.

    Raises:
        ValueError: If layer_names or scale_dict names a layer not present in
            layers.
    """
    layer_names = layer_names if layer_names is not None else list(BASELINE_LAYER_NAMES)
    scale_dict = scale_dict if scale_dict is not None else DEFAULT_BASELINE_EXPORT_SCALE

    missing_layers = [name for name in layer_names if name not in layers]
    if missing_layers:
        raise ValueError(f"Layer(s) not found in layers dict: {missing_layers}")
    missing_scales = [name for name in layer_names if name not in scale_dict]
    if missing_scales:
        raise ValueError(f"Layer(s) not found in scale_dict: {missing_scales}")

    images = [(layers[name], name, scale_dict[name]) for name in layer_names]
    tasks = export_image_list_to_drive(
        images=images, aoi=aoi, folder=export_folder, crs=crs
    )
    return dict(zip(layer_names, tasks))


def run_baseline_assessment(
    sites_fc: ee.FeatureCollection,
    export_folder: str,
    layer_names: list[str] | None = None,
    scale_dict: dict[str, int] | None = None,
    crs: str = DEFAULT_CRS,
    scale_continuous: int = 30,
    scale_landcover: int = 10,
    tile_scale: int = 4,
    include_landcover_areas: bool = True,
    export_summary: bool = True,
    summary_file_prefix: str = "baseline_site_summaries",
) -> tuple[dict[str, ee.Image], ee.FeatureCollection]:
    """Run the full static baseline assessment workflow for a site FeatureCollection.

    Reproduces the TGBS_Kwale baseline notebook end-to-end: derives the AOI from
    ``sites_fc``, builds the static baseline layer stack (terrain, canopy
    height, land cover, soil carbon, BII, Hansen forest-change — see
    :func:`build_baseline_layers`), summarizes it per site (continuous stats
    plus optional land-cover class areas — see :func:`summarize_baseline_layers`),
    exports the requested layers to Drive as individual GeoTIFFs (see
    :func:`export_baseline_layers`), and — by default — exports the summary
    table to the same Drive folder as a CSV.

    Args:
        sites_fc: ee.FeatureCollection of site polygon features; the combined
            geometry is used as the AOI for every layer.
        export_folder: Google Drive folder name to write all rasters (and the
            summary CSV, when exported) into.
        layer_names: Layer keys to export as rasters; defaults to every key in
            BASELINE_LAYER_NAMES (all baseline layers). The returned layers dict
            and summary table always cover the full baseline stack regardless
            of this filter.
        scale_dict: Per-layer raster export scale in metres, keyed by layer
            name; defaults to DEFAULT_BASELINE_EXPORT_SCALE (uniform 30m,
            including canopy_height downsampled from its native 1m).
        crs: Coordinate reference system for both the raster and summary
            exports (default DEFAULT_CRS).
        scale_continuous: Pixel scale in metres for the per-site
            continuous-layer summary (default 30).
        scale_landcover: Pixel scale in metres for the per-site land-cover
            class-area summary (default 10).
        tile_scale: EE tileScale parameter for both summary reductions
            (default 4).
        include_landcover_areas: If True (default), the summary table includes
            one '<class>_area_m2' property per ESA_CLASS_MAP entry.
        export_summary: If True (default), export the per-site summary table
            to export_folder as a CSV.
        summary_file_prefix: Filename prefix (without extension) and EE task
            description for the summary CSV export (default
            'baseline_site_summaries').

    Returns:
        tuple of (layers, summary_fc): the full baseline layers dict (as built
        by build_baseline_layers) and the per-site summary
        ee.FeatureCollection (as built by summarize_baseline_layers). Raster
        export tasks (and the summary table export task, when requested) are
        started as a side effect.
    """
    aoi = get_sites_geometry(sites_fc)
    layers = build_baseline_layers(aoi)

    summary_fc = summarize_baseline_layers(
        sites_fc=sites_fc,
        layers=layers,
        scale_continuous=scale_continuous,
        scale_landcover=scale_landcover,
        crs=crs,
        tile_scale=tile_scale,
        include_landcover_areas=include_landcover_areas,
    )

    export_baseline_layers(
        layers=layers,
        aoi=aoi,
        export_folder=export_folder,
        layer_names=layer_names,
        scale_dict=scale_dict,
        crs=crs,
    )

    if export_summary:
        export_table_to_drive(
            collection=summary_fc,
            description=summary_file_prefix,
            folder=export_folder,
            fileNamePrefix=summary_file_prefix,
        )

    return layers, summary_fc


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
