import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import ee

from eetools.compositing import (
    build_period_composites,
    build_seasonal_composites,
)
from eetools.constants import (
    BASELINE_CONTINUOUS_LAYERS,
    BASELINE_EXTRA_LAYER_NAMES,
    BASELINE_LAYER_METADATA,
    BASELINE_LAYER_NAMES,
    BASELINE_RESAMPLE_MAX_PIXELS_CAP,
    DEFAULT_CRS,
    DEFAULT_MIN_PIXELS_PER_REGION,
    ESA_CLASS_MAP,
)
from eetools.io import (
    export_image_list_to_drive,
    export_image_to_gcs,
    export_table_to_drive,
    snapped_transform,
)
from eetools.sensors.bii.preprocessing import get_bii_image
from eetools.sensors.canopy_height.preprocessing import get_canopy_height
from eetools.sensors.dem.preprocessing import get_terrain
from eetools.sensors.esa.preprocessing import get_land_cover
from eetools.sensors.hansen.preprocessing import (
    get_forest_2000,
    get_forest_loss_image,
    get_forest_loss_year_image,
)
from eetools.sensors.human_modification.preprocessing import get_human_modification
from eetools.sensors.isda.preprocessing import get_soil_carbon
from eetools.sensors.soilgrids.preprocessing import get_soilgrids_wrb
from eetools.site_selection.workflows import (  # noqa: F401 (re-exported entry points)
    identify_degraded_sites,
    identify_reference_sites,
    result_to_feature_collection,
)
from eetools.vectors import get_sites_geometry
from eetools.zonal import (
    check_region_pixel_coverage,
    image_collection_to_region_stats_fc,
    summarize_class_areas,
)


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
    *,
    scale: int,
    image_properties: list[str] | None = None,
    tile_scale: int = 4,
    min_pixels_per_region: float | None = DEFAULT_MIN_PIXELS_PER_REGION,
) -> ee.FeatureCollection:
    """Run the full site spectral timeseries workflow for annual or monthly compositing.

    Derives the AOI from the sites FeatureCollection, builds the sensor collection via
    the caller-supplied builder, composites to the requested temporal window, reduces
    per region, exports the result table to Drive, and returns the stats FeatureCollection.

    The ``collection_builder`` should be a partially-applied sensor function whose
    signature matches ``(aoi, start_date, end_date) -> ee.ImageCollection``:

    .. code-block:: python

        from functools import partial
        from eetools.sensors.sentinel.preprocessing import get_s2_sr_collection
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
            scale=10,  # Sentinel-2 native; required, no default
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
        scale: Required keyword. Pixel scale in metres for the ``reduceRegions``
            call; there is no default because it is an accuracy/EECU trade-off that
            depends on site size. Use the sensor's native scale for site-scale work
            (10 m Sentinel-2, 30 m HLS/Landsat, 250-500 m MODIS), or one step
            coarser (e.g. 60 m for HLS, ~4x fewer pixels) to save EECU while
            keeping dozens of pixels per site; 100-250 m suits large-region
            summaries.
        image_properties: List of image property names to copy onto each output
            feature; when None (default) all scalar image properties are copied.
        tile_scale: EE tileScale parameter to avoid memory limits (default 4).
        min_pixels_per_region: Warn (``UserWarning``) if any site covers fewer
            than this many ``scale``-sized pixels (default 4); the message names
            the smallest site and suggests a scale. The check costs one
            ``getInfo()`` round trip; pass None to skip it.

    Returns:
        ee.FeatureCollection of per-region statistics, one feature per site per
        composite period, with composite metadata properties and per-band reduction
        results attached. A batch table-export task to Drive is started as a side
        effect.

    Raises:
        ValueError: If ``composite_stat`` is not 'mean', 'median', or 'sum', or if
            ``temporal_scale`` is not 'annual' or 'monthly', or if ``scale`` or
            ``min_pixels_per_region`` is not positive.
    """
    aoi = get_sites_geometry(sites_fc)
    collection = collection_builder(aoi, start_date, end_date)
    composites = build_period_composites(
        collection, bands, start_date, end_date, temporal_scale, composite_stat
    )
    if min_pixels_per_region is not None:
        check_region_pixel_coverage(
            sites_fc, scale, min_pixels_per_region, stacklevel=3
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


@dataclass(frozen=True)
class BaselineLayerSpec:
    """One static baseline layer: how to build it, how to export it, and its
    dataset metadata for a data register.

    Attributes:
        name: Registry key (also the export file name).
        builder: Callable(region) -> unclipped single-band ee.Image. The region is passed for builders that need it; most ignore it.
        export_scale_m: Default export pixel size in metres.
        native_scale_m: Source pixel size; exports coarser than this are aggregated with ``resample``.
        categorical: Class codes (exported as int16) rather than continuous values (float32).
        resample: Aggregation when coarsening: ``"mean"`` (continuous), ``"mode"`` (categorical) or ``"nearest"`` (no aggregation).
        dataset_id: Source EE asset / collection ID.
        version: Dataset version or epoch.
        licence: Licence / terms of use.
        unit: Unit of the band values.
        band: Output band name.
    """

    name: str
    builder: Callable[[ee.Geometry], ee.Image]
    export_scale_m: float
    native_scale_m: float
    categorical: bool
    resample: Literal["mean", "mode", "nearest"]
    dataset_id: str
    version: str
    licence: str
    unit: str
    band: str


def _terrain_band(band: str) -> Callable[[ee.Geometry], ee.Image]:
    def _build(region: ee.Geometry) -> ee.Image:
        return get_terrain().select(band)

    return _build


_BASELINE_BUILDERS: dict[str, Callable[[ee.Geometry], ee.Image]] = {
    "dem": _terrain_band("elevation"),
    "slope": _terrain_band("slope"),
    "hillshade": _terrain_band("hillshade"),
    "canopy_height": lambda region: get_canopy_height(pin_projection=True),
    "land_cover": lambda region: get_land_cover(pin_projection=True),
    "soil_carbon": lambda region: get_soil_carbon(),
    "bii_all": lambda region: get_bii_image("1km").select("BII All"),
    "forest_2000": lambda region: get_forest_2000(),
    "forest_loss": lambda region: get_forest_loss_image(),
    "forest_lossyear": lambda region: get_forest_loss_year_image(),
    "soilgrids_wrb": lambda region: get_soilgrids_wrb(),
    "human_modification": lambda region: get_human_modification(),
}


def _build_registry() -> dict[str, BaselineLayerSpec]:
    registry: dict[str, BaselineLayerSpec] = {}
    for name, meta in BASELINE_LAYER_METADATA.items():
        fields = cast(dict[str, Any], meta)
        registry[name] = BaselineLayerSpec(
            name=name, builder=_BASELINE_BUILDERS[name], **fields
        )
    expected = list(BASELINE_LAYER_NAMES) + list(BASELINE_EXTRA_LAYER_NAMES)
    if list(registry) != expected:  # constants out of sync: fail at import
        raise RuntimeError(
            f"BASELINE_LAYER_METADATA keys {list(registry)} != {expected}."
        )
    return registry


# Every static baseline layer, keyed by name, in canonical order: the nine default
# layers (BASELINE_LAYER_NAMES) then the opt-in extras (BASELINE_EXTRA_LAYER_NAMES).
BASELINE_LAYER_REGISTRY: dict[str, BaselineLayerSpec] = _build_registry()


def _resolve_layer_names(layers: Sequence[str] | None) -> list[str]:
    names = list(BASELINE_LAYER_NAMES) if layers is None else list(layers)
    unknown = [name for name in names if name not in BASELINE_LAYER_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown baseline layer(s) {unknown}; "
            f"choose from {list(BASELINE_LAYER_REGISTRY)}."
        )
    return names


def baseline_layer_register(layers: Sequence[str] | None = None) -> list[dict]:
    """Dataset metadata of baseline layers, one row per layer, for a data register.

    Args:
        layers: Registry keys; defaults to every layer in ``BASELINE_LAYER_REGISTRY``.

    Returns:
        list[dict]. Per layer: name, dataset_id, version, licence, unit, band, categorical, native_scale_m, export_scale_m, resample.

    Raises:
        ValueError: If a name is not in the registry.
    """
    names = list(BASELINE_LAYER_REGISTRY) if layers is None else layers
    rows = []
    for name in _resolve_layer_names(names):
        spec = BASELINE_LAYER_REGISTRY[name]
        rows.append(
            {
                "name": spec.name,
                "dataset_id": spec.dataset_id,
                "version": spec.version,
                "licence": spec.licence,
                "unit": spec.unit,
                "band": spec.band,
                "categorical": spec.categorical,
                "native_scale_m": spec.native_scale_m,
                "export_scale_m": spec.export_scale_m,
                "resample": spec.resample,
            }
        )
    return rows


def build_baseline_layers(
    region: ee.Geometry | None = None,
    layers: Sequence[str] | None = None,
    clip: bool = False,
    *,
    aoi: ee.Geometry | None = None,
) -> dict[str, ee.Image]:
    """Build static baseline layers from the registry over a region.

    Each layer comes from its ``BASELINE_LAYER_REGISTRY`` builder: terrain, canopy
    height, land cover, soil carbon, biodiversity intactness and Hansen
    forest-change by default, plus the opt-in ``forest_lossyear``,
    ``soilgrids_wrb`` and ``human_modification``. Layers are unclipped unless
    ``clip=True``; pass the study extent (e.g. bbox of all sites + buffer) as the
    export region instead. See :func:`run_baseline_assessment` for the full
    build -> summarize -> export workflow.

    Args:
        region: Region the layers are built for (and clipped to when ``clip=True``).
        layers: Registry keys to build; defaults to the nine ``BASELINE_LAYER_NAMES`` (dem, slope, hillshade, canopy_height, land_cover, soil_carbon, bii_all, forest_2000, forest_loss).
        clip: Clip every layer to ``region`` (default False; before 0.7.1 layers were always clipped).
        aoi: Deprecated alias for ``region``.

    Returns:
        dict mapping each requested layer name to a single-band ee.Image.

    Raises:
        ValueError: If a layer name is unknown, both ``region`` and ``aoi`` are given, or neither is given while ``clip=True``.
    """
    if aoi is not None:
        if region is not None:
            raise ValueError("Pass region, not both region and aoi.")
        warnings.warn(
            "build_baseline_layers(aoi=...) is deprecated; use region=.",
            DeprecationWarning,
            stacklevel=2,
        )
        region = aoi
    if clip and region is None:
        raise ValueError("clip=True needs a region.")
    out: dict[str, ee.Image] = {}
    for name in _resolve_layer_names(layers):
        image = BASELINE_LAYER_REGISTRY[name].builder(cast(ee.Geometry, region))
        out[name] = image.clip(region) if clip else image
    return out


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


def _prepare_baseline_export(image: ee.Image, name: str, scale: float) -> ee.Image:
    # Aggregate (mean / mode) when exporting coarser than the source; at or below
    # native scale EE's default nearest-neighbour resampling is used. Layers not in
    # the registry are exported as they are.
    spec = BASELINE_LAYER_REGISTRY.get(name)
    if spec is None or spec.resample == "nearest":
        return image
    ratio = scale / spec.native_scale_m
    if ratio <= 1.001:
        return image
    reducer = ee.Reducer.mean() if spec.resample == "mean" else ee.Reducer.mode()
    max_pixels = min(math.ceil(ratio + 1) ** 2, BASELINE_RESAMPLE_MAX_PIXELS_CAP)
    return image.reduceResolution(reducer=reducer, maxPixels=max_pixels)


def export_baseline_layers(
    layers: dict[str, ee.Image],
    region: ee.Geometry | None = None,
    export_folder: str | None = None,
    layer_names: list[str] | None = None,
    scale_dict: dict[str, float] | None = None,
    crs: str = DEFAULT_CRS,
    *,
    dest: Literal["drive", "gcs"] = "drive",
    bucket: str | None = None,
    prefix: str | None = None,
    skip_existing: bool = True,
    start: bool = True,
    aoi: ee.Geometry | None = None,
) -> dict[str, Any]:
    """Export baseline layers, one GeoTIFF per layer, to Drive or Cloud Storage.

    Each layer is exported at its registry ``export_scale_m`` (override per layer
    with ``scale_dict``). When that is coarser than the layer's native scale it is
    aggregated first with ``reduceResolution``: mean for continuous layers, mode for
    categorical ones (e.g. canopy height 1 m -> 10 m mean by default).

    ``dest="gcs"`` writes cloud-optimised GeoTIFFs to
    ``gs://<bucket>/<prefix>/<layer>.tif`` on ``snapped_transform(region, crs,
    scale)`` (categorical layers as int16 with no-data -32768, continuous as float32
    with -9999), skipping layers that already exist when ``skip_existing``. ``crs``
    must then be a projected CRS (e.g. the project UTM zone), because the scale is
    in metres.

    Args:
        layers: Layer dict as returned by :func:`build_baseline_layers`.
        region: Export region shared by every layer (e.g. bbox of all sites + buffer).
        export_folder: Google Drive folder (required for ``dest="drive"``).
        layer_names: Layer keys to export; defaults to every key of ``layers``.
        scale_dict: Per-layer export scale in metres; layers it omits use the registry ``export_scale_m``.
        crs: Output CRS (default DEFAULT_CRS; pass a projected CRS for ``dest="gcs"``).
        dest: ``"drive"`` (default) or ``"gcs"``.
        bucket: Cloud Storage bucket (required for ``dest="gcs"``).
        prefix: Object prefix (required for ``dest="gcs"``).
        skip_existing: ``dest="gcs"`` only: skip layers whose tif already exists.
        start: ``dest="gcs"`` only: start the tasks (Drive tasks always start).
        aoi: Deprecated alias for ``region``.

    Returns:
        dict mapping each exported layer name to its ee.batch.Task (``None`` for a skipped GCS layer).

    Raises:
        ValueError: If a layer is missing from ``layers``, has no scale (not in the registry or ``scale_dict``), ``dest`` is unknown, a destination argument is missing, or ``dest="gcs"`` is used with EPSG:4326.
    """
    if aoi is not None:
        if region is not None:
            raise ValueError("Pass region, not both region and aoi.")
        warnings.warn(
            "export_baseline_layers(aoi=...) is deprecated; use region=.",
            DeprecationWarning,
            stacklevel=2,
        )
        region = aoi
    if region is None:
        raise ValueError("region is required.")
    names = list(layers) if layer_names is None else list(layer_names)
    missing_layers = [name for name in names if name not in layers]
    if missing_layers:
        raise ValueError(f"Layer(s) not found in layers dict: {missing_layers}")
    scales = {
        name: spec.export_scale_m for name, spec in BASELINE_LAYER_REGISTRY.items()
    }
    if scale_dict is not None:
        scales.update(scale_dict)
    missing_scales = [name for name in names if name not in scales]
    if missing_scales:
        raise ValueError(f"Layer(s) not found in scale_dict: {missing_scales}")

    prepared = [
        (name, _prepare_baseline_export(layers[name], name, scales[name]), scales[name])
        for name in names
    ]

    if dest == "drive":
        if export_folder is None:
            raise ValueError('export_folder is required for dest="drive".')
        tasks = export_image_list_to_drive(
            images=[(image, name, scale) for name, image, scale in prepared],
            aoi=region,
            folder=export_folder,
            crs=crs,
        )
        return dict(zip(names, tasks))

    if dest != "gcs":
        raise ValueError(f'dest must be "drive" or "gcs", got {dest!r}.')
    if bucket is None or prefix is None:
        raise ValueError('bucket and prefix are required for dest="gcs".')
    if crs.upper() == "EPSG:4326":
        raise ValueError(
            'dest="gcs" needs a projected crs (e.g. the project UTM zone): '
            "scales are in metres and the grid is snapped in crs units."
        )
    transforms: dict[float, list[float]] = {}
    result: dict[str, Any] = {}
    for name, image, scale in prepared:
        if scale not in transforms:  # one small getInfo per distinct scale
            transforms[scale] = snapped_transform(region, crs, scale)
        spec = BASELINE_LAYER_REGISTRY.get(name)
        categorical = spec.categorical if spec is not None else False
        result[name] = export_image_to_gcs(
            image,
            region,
            bucket=bucket,
            file_name_prefix=f"{prefix}/{name}",
            crs=crs,
            crs_transform=transforms[scale],
            dtype="int16" if categorical else "float32",
            skip_existing=skip_existing,
            start=start,
        )
    return result


def run_baseline_assessment(
    sites_fc: ee.FeatureCollection,
    export_folder: str,
    layer_names: list[str] | None = None,
    scale_dict: dict[str, float] | None = None,
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
            name; layers it omits use the registry ``export_scale_m`` (e.g.
            canopy_height 10 m mean, land_cover 10 m, bii_all 1 km). Pass
            ``DEFAULT_BASELINE_EXPORT_SCALE`` for the pre-0.7.1 uniform 30 m.
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
    layers = build_baseline_layers(aoi, clip=True)

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
        region=aoi,
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
    *,
    scale: int,
    image_properties: list[str] | None = None,
    tile_scale: int = 4,
    min_pixels_per_region: float | None = DEFAULT_MIN_PIXELS_PER_REGION,
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
        scale: Required keyword. Pixel scale in metres for the ``reduceRegions``
            call; see :func:`run_site_timeseries` for guidance.
        image_properties: Image properties to copy onto each output feature; None
            copies all scalar properties.
        tile_scale: EE tileScale (default 4).
        min_pixels_per_region: Warn if any site covers fewer than this many
            ``scale``-sized pixels (default 4); None skips the check.

    Returns:
        ee.FeatureCollection of per-region per-year seasonal statistics, each
        carrying ``year``, ``season``, ``season_months``, ``composite_stat``, and
        per-band reduction results. A batch table-export task is started as a side
        effect.

    Raises:
        ValueError: If ``composite_stat`` is not 'mean', 'median', or 'sum', or if
            ``season_months`` does not satisfy ``1 ≤ start ≤ end ≤ 12``, or if
            ``scale`` or ``min_pixels_per_region`` is not positive.
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
    if min_pixels_per_region is not None:
        check_region_pixel_coverage(
            sites_fc, scale, min_pixels_per_region, stacklevel=3
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
