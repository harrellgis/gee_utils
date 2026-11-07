"""Earth Engine side of site selection: focal profile, predictor/condition stacks,
and the batched reducer that fills ScreeningTables.

Every image here is built lazily; nothing is computed until :func:`reduce_cells` (or
:func:`profile_focal`) pulls a table. Each layer family is reduced at its native scale
in the focal UTM grid, and focal-plus-buffer pixels are masked out of every candidate
reduction before it runs.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import ee
import numpy as np
import pandas as pd
import shapely
import shapely.geometry

from eetools.constants import (
    CANOPY_HEIGHT_REDUCE_SCALE_M,
    CHIRPS_NATIVE_SCALE_M,
    CORE_FRAC_RADIUS_M,
    DW_COLLECTION,
    DYNAMIC_WORLD_CONTEXT_BANDS,
    DYNAMIC_WORLD_CONTEXT_SCALE_M,
    EMBEDDING_REDUCE_SCALE_M,
    ESA_BAND,
    ESA_WC_COLLECTION,
    FRACTION_SCALE_M,
    HAND_RIPARIAN_MAX_M,
    HANSEN_LOSS_BAND,
    HANSEN_LOSSYEAR_BAND,
    HANSEN_LOSSYEAR_EPOCH,
    HANSEN_LOSSYEAR_MAX,
    HANSEN_TREE_COVER_THRESHOLD,
    HANSEN_TREECOVER_BAND,
    MODIS_NBAR_SCALE_M,
    MODIS_VI_SCALE_M,
    NATURAL_WORLDCOVER_CLASSES,
    PREDICTOR_NATIVE_SCALE_M,
    REDUCE_BATCH_SIZE,
    REDUCE_RESOLUTION_MAX_PIXELS,
    SATELLITE_EMBEDDING_BANDS,
    SATELLITE_EMBEDDING_COLLECTION,
    SITE_SELECTION_INDICATOR_SCALE_M,
    TERRACLIMATE_NATIVE_SCALE_M,
    TPI_RADIUS_M,
    VIIRS_NTL_GAP_FILLED_BAND,
    VIIRS_REDUCE_SCALE_M,
    WDPA_DESIGNATED,
    WDPA_I_IV_CATEGORIES,
    WDPA_IUCN_FIELD,
    WDPA_IUCN_PROTECTION_WEIGHTS,
    WDPA_POLYGONS_COLLECTION,
    WDPA_RASTER_SCALE_M,
    WDPA_STATUS_FIELD,
    WDPA_STRICT_CATEGORIES,
    WORLDCLIM_DEFAULT_BANDS,
    WORLDCOVER_BARE,
    WORLDCOVER_BUILT,
    WORLDCOVER_CROPLAND,
    WORLDCOVER_WATER,
)
from eetools.sensors.bii.preprocessing import get_bii_image
from eetools.sensors.canopy_height.preprocessing import get_canopy_height
from eetools.sensors.chirps.preprocessing import (
    get_chirps_annual_totals,
    get_chirps_climatology,
)
from eetools.sensors.dem.preprocessing import get_copernicus_dem, get_tpi
from eetools.sensors.hansen.preprocessing import get_forest_change_image
from eetools.sensors.human_modification.preprocessing import get_human_modification
from eetools.sensors.hydro.preprocessing import (
    get_coast_distance,
    get_hand,
    get_sbtn_natural,
)
from eetools.sensors.isda.preprocessing import get_soil_texture
from eetools.sensors.modis.preprocessing import (
    annual_sens_slope,
    get_modis_annual_nbar_indices,
    get_modis_annual_ndvi,
)
from eetools.sensors.pnv.preprocessing import get_pnv_biome
from eetools.sensors.resolve.preprocessing import get_ecoregion_image
from eetools.sensors.soilgrids.preprocessing import get_soilgrids_wrb
from eetools.sensors.terraclimate.preprocessing import get_terraclimate_climatology
from eetools.sensors.vecea.preprocessing import get_vecea_class
from eetools.sensors.viirs.preprocessing import get_viirs_ntl_collection
from eetools.sensors.worldclim.preprocessing import get_worldclim_bio
from eetools.site_selection.types import ScreeningTables

MULTI_STATS = ["mean", "p10", "p50", "p90"]
COND_STATS = ["mean", "p10", "p90"]


def mean_percentile_reducer(percentiles: Sequence[int]) -> ee.Reducer:
    """``mean()`` combined with ``percentile(percentiles)`` (shared inputs).

    Args:
        percentiles: Percentiles to add (e.g. [10, 50, 90]).

    Returns:
        ee.Reducer emitting 'mean' and 'p<NN>' outputs.
    """
    return ee.Reducer.mean().combine(
        ee.Reducer.percentile(list(percentiles)), sharedInputs=True
    )


# --------------------------------------------------------------------------- #
# Run context                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class RunContext:
    """Everything the image builders need for one focal site.

    Attributes are set by the workflow from the focal profile; see
    :func:`profile_focal`.
    """

    focal_ee: ee.Geometry
    exclusion_ee: ee.Geometry
    search_ee: ee.Geometry
    crs: str
    end_year: int
    trend_start_year: int
    state_start_year: int
    asset_ids: dict[str, str]
    analysis_scale_m: int = 100
    tile_scale: int = 4
    natural_mask: str = "worldcover"
    target_cover_classes: tuple[int, ...] | None = None
    coastal: bool = False
    class_set: list[int] = field(default_factory=list)
    group_members: list[int] = field(default_factory=list)
    class_fallback: bool = False
    pnv_code: int | None = None
    eco_id: int | None = None
    wrb_code: int | None = None
    wrb_related: list[int] = field(default_factory=list)
    substrate_fallback: bool = False
    texture_code: int | None = None
    hand_riparian: bool = False
    baseline_start_year: int | None = None  # degraded anchor window (None: not used)
    baseline_end_year: int | None = None

    def outside_mask(self) -> ee.Image:
        """1 outside the focal site + buffer (and user exclusions), 0 inside."""
        return (
            ee.Image.constant(1)
            .paint(ee.FeatureCollection([ee.Feature(self.exclusion_ee)]), 0)
            .rename("outside")
        )

    def proj(self, scale: float) -> ee.Projection:
        """Focal UTM projection at a given scale (origin at the projection origin)."""
        return ee.Projection(self.crs).atScale(scale)


def _worldcover() -> ee.Image:
    col = ee.ImageCollection(ESA_WC_COLLECTION)
    return col.mosaic().select(ESA_BAND).setDefaultProjection(col.first().projection())


def _remap_in(image: ee.Image, values: Sequence[int], name: str) -> ee.Image:
    """0/1 indicator of ``image`` being in ``values`` (0 where masked)."""
    values = [int(v) for v in values]
    if not values:
        return ee.Image.constant(0).rename(name)
    return image.remap(values, [1] * len(values), 0).unmask(0).rename(name).toFloat()


# --------------------------------------------------------------------------- #
# Stacks                                                                       #
# --------------------------------------------------------------------------- #
def build_environment_stack(
    ctx: RunContext | None = None,
    coastal: bool = False,
    crs: str | None = None,
    region: ee.Geometry | None = None,
) -> ee.Image:
    """Condition-independent MESS predictors as one image (not focal-masked).

    Args:
        ctx: Optional RunContext (supplies coastal flag and CRS).
        coastal: Add distance to coast (ignored when ctx is given).
        crs: CRS for the coast-distance grid (ignored when ctx is given).
        region: Region bounding the coast-distance computation (ignored with ctx).

    Returns:
        ee.Image with bands elevation, slope, tpi, sand, clay, hand, bio01, bio04,
        bio12, bio15, cwd, aridity (and coast_dist when coastal).
    """
    if ctx is not None:
        coastal, crs, region = ctx.coastal, ctx.crs, ctx.search_ee
    dem = get_copernicus_dem()
    slope = ee.Terrain.slope(dem).rename("slope")
    tpi = get_tpi(dem, radius_m=TPI_RADIUS_M)
    soil = get_soil_texture().select(["sand", "clay"])
    bands = [
        dem.toFloat(),
        slope.toFloat(),
        tpi.toFloat(),
        soil,
        get_hand().toFloat(),
        get_worldclim_bio(WORLDCLIM_DEFAULT_BANDS),
        get_terraclimate_climatology().select(["cwd", "aridity"]),
    ]
    if coastal and crs:
        if region is None:
            raise ValueError("A region is required for the coast-distance predictor.")
        bands.append(get_coast_distance(region, crs))
    return ee.Image.cat(bands)


def build_gate_indicators(ctx: RunContext) -> ee.Image:
    """Stage 1 90 m stack: 0/1 gate indicators, elevation and WorldClim (spec §5.1).

    Indicators exist for every relaxation level (class set, group set, WRB same,
    WRB related, HAND class), so relaxing a gate re-filters in pandas. Fallback
    indicators (PNV biome + ECO_ID, iSDA texture) are built only when the primary
    layer is missing at the focal site. Focal-plus-buffer pixels are masked.

    Args:
        ctx: RunContext with the focal sets.

    Returns:
        ee.Image with bands land, in_class_set, in_group_set, pnv_eco_same,
        wrb_same, wrb_related, texture_same, hand_focal, elevation, bio01, bio04,
        bio12, bio15.
    """
    land = _worldcover().neq(WORLDCOVER_WATER).unmask(0).rename("land").toFloat()
    vecea = get_vecea_class(asset_id=ctx.asset_ids.get("vecea"))
    wrb = get_soilgrids_wrb(asset_id=ctx.asset_ids.get("soilgrids_wrb"))
    bands = [
        land,
        _remap_in(vecea, ctx.class_set, "in_class_set"),
        _remap_in(vecea, ctx.group_members, "in_group_set"),
    ]
    if ctx.class_fallback and ctx.pnv_code is not None and ctx.eco_id is not None:
        same = (
            get_pnv_biome()
            .eq(ctx.pnv_code)
            .And(get_ecoregion_image(ctx.search_ee).select("eco_id").eq(ctx.eco_id))
            .unmask(0)
        )
        bands.append(same.rename("pnv_eco_same").toFloat())
    else:
        bands.append(ee.Image.constant(0).toFloat().rename("pnv_eco_same"))
    same_wrb = [ctx.wrb_code] if ctx.wrb_code is not None else []
    bands += [
        _remap_in(wrb, same_wrb, "wrb_same"),
        _remap_in(wrb, ctx.wrb_related, "wrb_related"),
    ]
    if ctx.substrate_fallback and ctx.texture_code is not None:
        tex = get_soil_texture().select("texture_class")
        bands.append(_remap_in(tex, [ctx.texture_code], "texture_same"))
    else:
        bands.append(ee.Image.constant(0).toFloat().rename("texture_same"))
    hand = get_hand()
    riparian = hand.lt(HAND_RIPARIAN_MAX_M)
    focal_class = riparian if ctx.hand_riparian else riparian.Not()
    bands.append(focal_class.unmask(0).rename("hand_focal").toFloat())
    bands.append(get_copernicus_dem().toFloat())
    bands.append(get_worldclim_bio(WORLDCLIM_DEFAULT_BANDS))
    return ee.Image.cat(bands).updateMask(ctx.outside_mask())


GATE_BANDS = [
    "land",
    "in_class_set",
    "in_group_set",
    "pnv_eco_same",
    "wrb_same",
    "wrb_related",
    "texture_same",
    "hand_focal",
    "elevation",
    "bio01",
    "bio04",
    "bio12",
    "bio15",
]


def build_climate_stack(ctx: RunContext) -> ee.Image:
    """CHIRPS + TerraClimate climatologies (+ coast distance) for the climate call.

    Not focal-masked: climate pixels (~5 km) are larger than the exclusion zone, and a
    masked climate pixel would leave small cells without a value.

    Args:
        ctx: RunContext.

    Returns:
        ee.Image with bands rain_mean_annual, cwd, aridity (and coast_dist).
    """
    bands = [
        get_chirps_climatology(ctx.search_ee),
        get_terraclimate_climatology().select(["cwd", "aridity"]),
    ]
    if ctx.coastal:
        bands.append(get_coast_distance(ctx.search_ee, ctx.crs))
    return ee.Image.cat(bands)


def climate_bands(ctx: RunContext) -> list[str]:
    """Band names of :func:`build_climate_stack`."""
    return ["rain_mean_annual", "cwd", "aridity"] + (
        ["coast_dist"] if ctx.coastal else []
    )


TERRAIN_SOIL_BANDS = ["elevation", "slope", "tpi", "sand", "clay", "hand"]


def build_terrain_soil_stack(ctx: RunContext) -> ee.Image:
    """Fine-varying MESS predictors (terrain, soil texture, HAND), focal-masked.

    Args:
        ctx: RunContext.

    Returns:
        ee.Image with bands elevation, slope, tpi, sand, clay, hand.
    """
    return (
        build_environment_stack(ctx)
        .select(TERRAIN_SOIL_BANDS)
        .updateMask(ctx.outside_mask())
    )


def _state_and_trend(
    annual: ee.ImageCollection, band: str, state_start: int, out_prefix: str
) -> ee.Image:
    state = (
        annual.filter(ee.Filter.gte("year", state_start))
        .select(band)
        .median()
        .rename(f"{out_prefix}_state")
    )
    trend = annual_sens_slope(annual, band).rename(f"{out_prefix}_trend")
    return state.addBands(trend)


def build_ndvi_metrics(ctx: RunContext) -> ee.Image:
    """MOD13Q1 productivity state and trend (250 m), focal-masked.

    ``ndvi_state`` = median of annual mean NDVI over the state window; ``ndvi_trend``
    = Sen's slope of annual mean NDVI over [trend_start_year, end_year].

    Args:
        ctx: RunContext.

    Returns:
        ee.Image with bands ndvi_state, ndvi_trend.
    """
    annual = get_modis_annual_ndvi(ctx.search_ee, ctx.trend_start_year, ctx.end_year)
    return _state_and_trend(annual, "NDVI", ctx.state_start_year, "ndvi").updateMask(
        ctx.outside_mask()
    )


def _ndmi_state(ctx: RunContext, annual: ee.ImageCollection) -> ee.Image:
    return (
        annual.filter(ee.Filter.gte("year", ctx.state_start_year))
        .select("NDMI")
        .median()
        .rename("ndmi_state")
    )


def build_nbar_metrics(ctx: RunContext) -> ee.Image:
    """MCD43A4 NDMI state and NBR trend (500 m), focal-masked.

    Args:
        ctx: RunContext.

    Returns:
        ee.Image with bands ndmi_state, nbr_trend.
    """
    annual = get_modis_annual_nbar_indices(
        ctx.search_ee, ctx.trend_start_year, ctx.end_year
    )
    nbr = annual_sens_slope(annual, "NBR").rename("nbr_trend")
    return _ndmi_state(ctx, annual).addBands(nbr).updateMask(ctx.outside_mask())


def build_baseline_state(ctx: RunContext, index: str, masked: bool = True) -> ee.Image:
    """Baseline-window state of one anchor index (degraded anchor constraint, D1).

    Median of annual means over [baseline_start_year, baseline_end_year], the same
    statistic as the current ``*_state`` window, so the two windows compare
    like-for-like.

    Args:
        ctx: RunContext with baseline_start_year / baseline_end_year set.
        index: 'ndvi' (MOD13Q1, 250 m) or 'ndmi' (MCD43A4, 500 m).
        masked: Mask out the focal site + buffer (candidate reductions).

    Returns:
        ee.Image with the single band ``<index>_base``.

    Raises:
        ValueError: If the baseline window is unset or the index is unknown.
    """
    start, end = ctx.baseline_start_year, ctx.baseline_end_year
    if start is None or end is None:
        raise ValueError("RunContext has no baseline window.")
    if index == "ndvi":
        annual = get_modis_annual_ndvi(ctx.search_ee, start, end).select("NDVI")
    elif index == "ndmi":
        annual = get_modis_annual_nbar_indices(ctx.search_ee, start, end).select("NDMI")
    else:
        raise ValueError(f"Unknown anchor index {index!r}.")
    img = annual.median().rename(f"{index}_base")
    return img.updateMask(ctx.outside_mask()) if masked else img


def reduce_focal_anchor_state(ctx: RunContext) -> dict[str, float]:
    """Focal-site means of the anchor indices in the baseline and current windows.

    One round trip. The images are unmasked (the focal site is what is measured), and
    each index is reduced at its native MODIS scale. reduceRegion weights partially
    covered pixels, so focal sites smaller than one pixel still get a value.

    Args:
        ctx: RunContext (baseline window optional).

    Returns:
        {'ndvi_state', 'ndmi_state', 'ndvi_base', 'ndmi_base'} -> float (NaN where
        unavailable; the *_base keys are NaN when the context has no baseline window).
    """

    def mean(img: ee.Image, scale: float) -> ee.Dictionary:
        return img.reduceRegion(
            ee.Reducer.mean(),
            ctx.focal_ee,
            scale=scale,
            crs=ctx.crs,
            tileScale=ctx.tile_scale,
            maxPixels=int(1e10),
        )

    ndvi_annual = get_modis_annual_ndvi(
        ctx.focal_ee, ctx.state_start_year, ctx.end_year
    )
    nbar_annual = get_modis_annual_nbar_indices(
        ctx.focal_ee, ctx.state_start_year, ctx.end_year
    )
    ndvi = ndvi_annual.select("NDVI").median().rename("ndvi_state")
    ndmi = _ndmi_state(ctx, nbar_annual)
    if ctx.baseline_start_year is not None:
        ndvi = ndvi.addBands(build_baseline_state(ctx, "ndvi", masked=False))
        ndmi = ndmi.addBands(build_baseline_state(ctx, "ndmi", masked=False))
    vals = cast(
        dict,
        mean(ndvi, MODIS_VI_SCALE_M).combine(mean(ndmi, MODIS_NBAR_SCALE_M)).getInfo()
        or {},
    )
    keys = ("ndvi_state", "ndmi_state", "ndvi_base", "ndmi_base")
    return {k: float("nan") if vals.get(k) is None else float(vals[k]) for k in keys}


def build_rue_trend(ctx: RunContext) -> ee.Image:
    """Rain-use-efficiency trend: Sen's slope of annual NDVI / CHIRPS annual rainfall.

    RESTREND-style; removes rainfall-driven greening and browning in drylands.

    Args:
        ctx: RunContext.

    Returns:
        ee.Image with a single 'rue_trend' band (NDVI per 1000 mm, per year).
    """
    ndvi = get_modis_annual_ndvi(ctx.search_ee, ctx.trend_start_year, ctx.end_year)
    rain = get_chirps_annual_totals(ctx.search_ee, ctx.trend_start_year, ctx.end_year)
    joined = ee.Join.inner().apply(
        ndvi, rain, ee.Filter.equals(leftField="year", rightField="year")
    )

    def _ratio(pair: ee.Feature) -> ee.Image:
        pair = ee.Feature(pair)
        n = ee.Image(pair.get("primary"))
        r = ee.Image(pair.get("secondary"))
        return (
            n.divide(r.divide(1000)).rename("rue").toFloat().set("year", n.get("year"))
        )

    rue = ee.ImageCollection(joined.map(_ratio))
    return (
        annual_sens_slope(rue, "rue").rename("rue_trend").updateMask(ctx.outside_mask())
    )


def build_viirs(ctx: RunContext) -> ee.Image:
    """Mean gap-filled VIIRS night-light radiance over the state window, focal-masked.

    Args:
        ctx: RunContext.

    Returns:
        ee.Image with a single 'viirs' band (nW/cm^2/sr).
    """
    col = get_viirs_ntl_collection(
        ctx.search_ee,
        f"{ctx.state_start_year}-01-01",
        f"{ctx.end_year + 1}-01-01",
        apply_qa_mask=False,
    )
    return (
        col.select(VIIRS_NTL_GAP_FILLED_BAND)
        .mean()
        .rename("viirs")
        .updateMask(ctx.outside_mask())
    )


def build_fraction_image(ctx: RunContext) -> ee.Image:
    """The single 250 m static condition image (spec §5.3, C12).

    Native 0/1 masks are focal-masked *before* aggregation and averaged to the 250 m
    focal UTM grid with ``reduceResolution(mean)``; every cell is a multiple of 250 m,
    so cell means are exact area fractions. ``core_frac`` is a 1 km focal mean of the
    unmasked natural fraction (so a degraded focal site still counts as fragmentation
    next to it), masked afterwards. gHM (90 m) is aggregated the same way; BII (1 km)
    is resampled. WDPA shares are painted at 30 m and aggregated.

    Args:
        ctx: RunContext.

    Returns:
        ee.Image (250 m, focal UTM) with bands natural_frac, bare_frac, crop_frac,
        built_frac, water_frac, loss_since_2000_frac, recent_loss_frac,
        forest_loss_frac, core_frac, ghm, bii, pa_share, pa_i_iv_share,
        pa_strict_share, pa_overlap.
    """
    outside = ctx.outside_mask()
    p250 = ctx.proj(FRACTION_SCALE_M)

    def agg(img: ee.Image, masked: bool = True) -> ee.Image:
        img = img.toFloat()
        if masked:
            img = img.updateMask(outside)
        return img.reduceResolution(
            ee.Reducer.mean(), maxPixels=REDUCE_RESOLUTION_MAX_PIXELS
        ).reproject(p250)

    wc = _worldcover()
    if ctx.natural_mask == "sbtn":
        natural = get_sbtn_natural().select("natural").eq(1)
    else:
        classes = list(ctx.target_cover_classes or NATURAL_WORLDCOVER_CLASSES)
        natural = wc.remap(classes, [1] * len(classes), 0)
    natural = natural.rename("natural_frac")
    lc = ee.Image.cat(
        [
            natural,
            wc.eq(WORLDCOVER_BARE).rename("bare_frac"),
            wc.eq(WORLDCOVER_CROPLAND).rename("crop_frac"),
            wc.eq(WORLDCOVER_BUILT).rename("built_frac"),
            wc.eq(WORLDCOVER_WATER).rename("water_frac"),
        ]
    )
    natural_unmasked = agg(natural, masked=False)
    core = (
        natural_unmasked.focalMean(
            radius=CORE_FRAC_RADIUS_M, kernelType="circle", units="meters"
        )
        .reproject(p250)
        .rename("core_frac")
        .updateMask(outside)
    )

    gfc = get_forest_change_image()
    forest2000 = gfc.select(HANSEN_TREECOVER_BAND).gte(HANSEN_TREE_COVER_THRESHOLD)
    # lossyear is masked where no loss occurred: unmask so "no loss" counts as 0.
    lossyear = gfc.select(HANSEN_LOSSYEAR_BAND).unmask(0)
    hansen_end = min(ctx.end_year, HANSEN_LOSSYEAR_EPOCH + HANSEN_LOSSYEAR_MAX)

    def loss_between(y0: int, y1: int) -> ee.Image:
        a, b = y0 - HANSEN_LOSSYEAR_EPOCH, y1 - HANSEN_LOSSYEAR_EPOCH
        return forest2000.And(lossyear.gte(a)).And(lossyear.lte(b))

    hansen = ee.Image.cat(
        [
            forest2000.And(gfc.select(HANSEN_LOSS_BAND)).rename("loss_since_2000_frac"),
            loss_between(hansen_end - 4, hansen_end).rename("recent_loss_frac"),
            loss_between(max(ctx.trend_start_year, 2001), hansen_end).rename(
                "forest_loss_frac"
            ),
        ]
    ).setDefaultProjection(gfc.projection())

    ghm = get_human_modification()
    bii = (
        get_bii_image()
        .select(["BII All"], ["bii"])
        .reproject(p250)
        .updateMask(outside)
        .toFloat()
    )

    pas = (
        ee.FeatureCollection(WDPA_POLYGONS_COLLECTION)
        .filterBounds(ctx.search_ee)
        .filter(ee.Filter.eq(WDPA_STATUS_FIELD, WDPA_DESIGNATED))
    )
    weights = ee.Dictionary(WDPA_IUCN_PROTECTION_WEIGHTS)
    pas_w = pas.map(
        lambda f: f.set("pa_w", weights.get(f.get(WDPA_IUCN_FIELD), 0.6))
    ).sort("pa_w")
    p30 = ctx.proj(WDPA_RASTER_SCALE_M)
    pa = (
        ee.Image.cat(
            [
                ee.Image.constant(0).paint(pas, 1).rename("pa_share"),
                ee.Image.constant(0)
                .paint(
                    pas.filter(
                        ee.Filter.inList(WDPA_IUCN_FIELD, list(WDPA_I_IV_CATEGORIES))
                    ),
                    1,
                )
                .rename("pa_i_iv_share"),
                ee.Image.constant(0)
                .paint(
                    pas.filter(
                        ee.Filter.inList(WDPA_IUCN_FIELD, list(WDPA_STRICT_CATEGORIES))
                    ),
                    1,
                )
                .rename("pa_strict_share"),
                ee.Image.constant(0)
                .toFloat()
                .paint(pas_w, "pa_w")
                .rename("pa_overlap"),
            ]
        )
        .toFloat()
        .reproject(p30)
    )
    return ee.Image.cat(
        [
            agg(lc.setDefaultProjection(wc.projection())),
            agg(hansen),
            core,
            agg(ghm),
            bii,
            agg(pa),
        ]
    )


FRACTION_BANDS = [
    "natural_frac",
    "bare_frac",
    "crop_frac",
    "built_frac",
    "water_frac",
    "loss_since_2000_frac",
    "recent_loss_frac",
    "forest_loss_frac",
    "core_frac",
    "ghm",
    "bii",
    "pa_share",
    "pa_i_iv_share",
    "pa_strict_share",
    "pa_overlap",
]


def build_canopy(ctx: RunContext) -> ee.Image:
    """Meta 1 m canopy height (reduced at >= 30 m), focal-masked.

    Args:
        ctx: RunContext.

    Returns:
        ee.Image with a single 'canopy_height' band (m).
    """
    return get_canopy_height().toFloat().updateMask(ctx.outside_mask())


def build_label_image(ctx: RunContext) -> ee.Image:
    """Categorical label image for majority classes (VECEA or PNV, and WRB).

    Args:
        ctx: RunContext.

    Returns:
        ee.Image with bands 'vecea_class' (or 'pnv_biome' under the class fallback)
        and 'wrb_group', focal-masked.
    """
    if ctx.class_fallback:
        cls = get_pnv_biome().rename("pnv_biome")
    else:
        cls = get_vecea_class(asset_id=ctx.asset_ids.get("vecea"))
    wrb = get_soilgrids_wrb(asset_id=ctx.asset_ids.get("soilgrids_wrb"))
    return cls.addBands(wrb).updateMask(ctx.outside_mask())


def build_embedding_image(ctx: RunContext, year: int, masked: bool = True) -> ee.Image:
    """Satellite Embedding V1 mosaic for one year (64 bands).

    Args:
        ctx: RunContext.
        year: Calendar year.
        masked: Apply the focal-plus-buffer mask (False for the focal query itself).

    Returns:
        ee.Image with bands A00..A63.
    """
    img = (
        ee.ImageCollection(SATELLITE_EMBEDDING_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(ctx.search_ee)
        .mosaic()
        .select(SATELLITE_EMBEDDING_BANDS)
    )
    return img.updateMask(ctx.outside_mask()) if masked else img


def build_dw_context(ctx: RunContext) -> ee.Image:
    """Dynamic World mean bare / crops / built probability over the state window.

    Ranked-site context only (never reduced over the pool, never scored).

    Args:
        ctx: RunContext.

    Returns:
        ee.Image with bands dw_bare, dw_crops, dw_built.
    """
    col = (
        ee.ImageCollection(DW_COLLECTION)
        .filterBounds(ctx.search_ee)
        .filterDate(f"{ctx.state_start_year}-01-01", f"{ctx.end_year + 1}-01-01")
        .select(DYNAMIC_WORLD_CONTEXT_BANDS)
    )
    return col.mean().rename([f"dw_{b}" for b in DYNAMIC_WORLD_CONTEXT_BANDS])


# --------------------------------------------------------------------------- #
# Reduction families                                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Family:
    """A memoised reduction family: one image, one reducer, one scale."""

    name: str
    bands: tuple[str, ...]
    stats: tuple[str, ...] | None  # None -> single-output reducer (column = band)
    scale: float
    reducer: str  # 'multi' (mean + p10/50/90), 'cond' (mean + p10/90), 'mean', 'median', 'hist'


def family_reducer(kind: str) -> ee.Reducer:
    """ee.Reducer for a Family.reducer kind."""
    if kind == "multi":
        return mean_percentile_reducer([10, 50, 90])
    if kind == "cond":
        return mean_percentile_reducer([10, 90])
    if kind == "mean":
        return ee.Reducer.mean()
    if kind == "median":
        return ee.Reducer.median()
    if kind == "hist":
        return ee.Reducer.frequencyHistogram()
    raise ValueError(f"Unknown reducer kind {kind!r}.")


def output_columns(bands: Sequence[str], stats: Sequence[str] | None) -> list[str]:
    """Canonical column names a family produces.

    Args:
        bands: Image band names.
        stats: Reducer output names, or None for single-output reducers.

    Returns:
        ``<band>_<stat>`` for multi-output reducers, else ``<band>``.
    """
    if stats is None:
        return list(bands)
    return [f"{b}_{s}" for b in bands for s in stats]


def _rename_reduced(
    df: pd.DataFrame,
    bands: Sequence[str],
    stats: Sequence[str] | None,
    single_name: str,
) -> pd.DataFrame:
    """Normalise EE output names (single-band images drop the band prefix)."""
    if len(bands) == 1:
        b = bands[0]
        if stats is None:
            df = df.rename(columns={single_name: b})
        else:
            df = df.rename(columns={s: f"{b}_{s}" for s in stats})
    return df


def cells_feature_collection(cells: pd.DataFrame, crs: str) -> ee.FeatureCollection:
    """Server-side cell polygons from (cell_id, ix, iy, size) rows.

    Only integer indices travel in the request; rectangles are built server-side in
    the grid CRS (planar, geodesic=False).

    Args:
        cells: DataFrame indexed by cell_id with ix, iy and size columns.
        crs: Grid CRS string.

    Returns:
        ee.FeatureCollection with a 'cell_id' property per feature.
    """
    rows = [
        [cid, int(ix), int(iy), float(size)]
        for cid, ix, iy, size in zip(
            cells.index, cells["ix"], cells["iy"], cells["size"], strict=True
        )
    ]

    def _make(row: Any) -> ee.Feature:
        row = ee.List(row)
        size = ee.Number(row.get(3))
        x0 = ee.Number(row.get(1)).multiply(size)
        y0 = ee.Number(row.get(2)).multiply(size)
        rect = ee.Geometry.Rectangle(
            ee.List([x0, y0, x0.add(size), y0.add(size)]), crs, False
        )
        return ee.Feature(rect, {"cell_id": row.get(0)})

    return ee.FeatureCollection(ee.List(rows).map(_make))


def _pull(fc: ee.FeatureCollection) -> pd.DataFrame:
    fc = fc.select([".*"], None, False)
    try:
        df = ee.data.computeFeatures(
            {"expression": fc, "fileFormat": "PANDAS_DATAFRAME"}
        )
    except Exception:  # noqa: BLE001 - fall back to getInfo on older APIs
        info = cast(dict, fc.getInfo())
        df = pd.DataFrame([f["properties"] for f in info["features"]])
    return df.drop(columns=["geo"], errors="ignore")


def reduce_cells(
    image: ee.Image,
    cells: pd.DataFrame,
    *,
    bands: Sequence[str],
    stats: Sequence[str] | None,
    reducer: ee.Reducer,
    scale: float,
    crs: str,
    tile_scale: int = 4,
    single_name: str = "mean",
    batch_size: int = REDUCE_BATCH_SIZE,
) -> pd.DataFrame:
    """Reduce an image over grid cells in batches and pull a pandas table.

    One ``reduceRegions`` per batch with an explicit scale, CRS and tileScale (never
    bestEffort); the table is pulled once per batch with ``computeFeatures``.

    Args:
        image: Image to reduce.
        cells: Cells (index cell_id; ix, iy, size).
        bands: Band names of ``image`` (for column naming).
        stats: Reducer output names for multi-output reducers, else None.
        reducer: ee.Reducer.
        scale: Reduction scale (m).
        crs: Grid CRS.
        tile_scale: reduceRegions tileScale.
        single_name: Output name of a single-output reducer on a one-band image.
        batch_size: Cells per request.

    Returns:
        DataFrame indexed by cell_id with ``output_columns(bands, stats)`` columns
        (NaN where a reduction was null).
    """
    want = output_columns(bands, stats)
    if len(cells) == 0:
        return pd.DataFrame(columns=want, index=pd.Index([], name="cell_id"))
    frames = []
    for start in range(0, len(cells), batch_size):
        chunk = cells.iloc[start : start + batch_size]
        fc = image.reduceRegions(
            collection=cells_feature_collection(chunk, crs),
            reducer=reducer,
            scale=scale,
            crs=crs,
            tileScale=tile_scale,
        )
        frames.append(_pull(fc))
    df = pd.concat(frames, ignore_index=True)
    df = _rename_reduced(df, bands, stats, single_name)
    df = df.set_index("cell_id")
    df.index.name = "cell_id"
    for c in want:
        if c not in df.columns:
            df[c] = np.nan
    return df.reindex(cells.index)[want]


# --------------------------------------------------------------------------- #
# Table store (mutable working copy of ScreeningTables)                        #
# --------------------------------------------------------------------------- #
class TableStore:
    """Mutable working copy of ScreeningTables used during one run.

    Every family is memoised by cell_id via a ``_done_<family>`` column; only missing
    cells are reduced.
    """

    def __init__(self, tables: ScreeningTables) -> None:
        self.coarse = tables.coarse.copy()
        self.fine = tables.fine.copy()
        self.embeddings = tables.embeddings.copy()
        self.labels = tables.labels.copy()
        self.key = tables.key
        self.created = tables.created
        self.reused_cells = {"coarse": len(self.coarse), "fine": len(self.fine)}
        self.reductions: list[dict[str, Any]] = []

    def table(self, level: str) -> pd.DataFrame:
        """Return the coarse or fine table."""
        return {"coarse": self.coarse, "fine": self.fine, "labels": self.labels}[level]

    def _set(self, level: str, df: pd.DataFrame) -> None:
        if level == "coarse":
            self.coarse = df
        elif level == "fine":
            self.fine = df
        elif level == "labels":
            self.labels = df
        else:
            raise ValueError(f"Unknown table level {level!r}.")

    def missing(self, level: str, family: str, ids: Sequence[str]) -> list[str]:
        """Cell ids in ``ids`` not yet reduced for ``family``."""
        tab = self.table(level)
        flag = f"_done_{family}"
        if flag not in tab.columns:
            return list(ids)
        done = tab.index[tab[flag].fillna(False).astype(bool)]
        done_set = set(done)
        return [i for i in ids if i not in done_set]

    def update(self, level: str, family: str, values: pd.DataFrame) -> None:
        """Insert reduced values and mark the family done for those cells."""
        tab = self.table(level)
        values = values.copy()
        values[f"_done_{family}"] = True
        new_ids = values.index.difference(tab.index)
        if len(new_ids):
            tab = pd.concat(
                [tab, pd.DataFrame(index=pd.Index(new_ids, name="cell_id"))]
            )
        new_cols = [c for c in values.columns if c not in tab.columns]
        if new_cols:
            # New columns keep the incoming dtype (floats, JSON strings, bools).
            tab = pd.concat([tab, values[new_cols].reindex(tab.index)], axis=1)
        for col in values.columns:
            if col in new_cols:
                continue
            numeric_in = pd.api.types.is_numeric_dtype(values[col])
            if pd.api.types.is_numeric_dtype(tab[col]) != numeric_in:
                tab[col] = tab[col].astype(object)
            tab.loc[values.index, col] = values[col]
        tab.index.name = "cell_id"
        self._set(level, tab)

    def to_tables(self) -> ScreeningTables:
        """Freeze into ScreeningTables (rows sorted by cell_id for determinism)."""

        def _clean(df: pd.DataFrame) -> pd.DataFrame:
            df = df.sort_index()
            for c in df.columns:
                if c.startswith("_done_"):
                    df[c] = df[c].fillna(False).astype(bool)
            return df

        emb = self.embeddings
        if len(emb):
            emb = emb.sort_values(["cell_id", "year"]).reset_index(drop=True)
        return ScreeningTables(
            coarse=_clean(self.coarse.copy()),
            fine=_clean(self.fine.copy()),
            embeddings=emb,
            labels=self.labels.sort_index(),
            key=self.key,
            created=self.created,
        )


def ensure_family(
    store: TableStore,
    level: str,
    family: Family,
    cells: pd.DataFrame,
    image_fn: Any,
    ctx: RunContext,
) -> None:
    """Reduce ``family`` over the cells that don't have it yet (memoised).

    Args:
        store: TableStore.
        level: 'coarse' or 'fine'.
        family: Family spec.
        cells: Cells needing the family (index cell_id; ix, iy, size).
        image_fn: Zero-argument callable building the family's image (called only
            if something is missing).
        ctx: RunContext (CRS, tileScale).

    Returns:
        None. The store is updated in place.
    """
    todo = store.missing(level, family.name, list(cells.index))
    if not todo:
        return
    single = {"mean": "mean", "median": "median", "hist": "histogram"}.get(
        family.reducer, "mean"
    )
    df = reduce_cells(
        image_fn(),
        cells.loc[todo],
        bands=family.bands,
        stats=family.stats,
        reducer=family_reducer(family.reducer),
        scale=family.scale,
        crs=ctx.crs,
        tile_scale=ctx.tile_scale,
        single_name=single,
    )
    if family.reducer == "hist":
        for c in df.columns:
            df[c] = [
                json.dumps(v, sort_keys=True) if isinstance(v, dict) else None
                for v in df[c]
            ]
    store.reductions.append(
        {
            "level": level,
            "family": family.name,
            "cells": len(todo),
            "scale": family.scale,
        }
    )
    store.update(level, family.name, df)


def families(ctx: RunContext) -> dict[str, Family]:
    """The reduction families of a run, keyed by name."""
    return {
        "stage1": Family(
            "stage1",
            tuple(GATE_BANDS),
            tuple(MULTI_STATS),
            SITE_SELECTION_INDICATOR_SCALE_M,
            "multi",
        ),
        "climate": Family(
            "climate",
            tuple(climate_bands(ctx)),
            None,
            TERRACLIMATE_NATIVE_SCALE_M,
            "mean",
        ),
        "env": Family(
            "env", tuple(TERRAIN_SOIL_BANDS), None, ctx.analysis_scale_m, "median"
        ),
        "ndvi": Family(
            "ndvi",
            ("ndvi_state", "ndvi_trend"),
            tuple(COND_STATS),
            MODIS_VI_SCALE_M,
            "cond",
        ),
        "nbar": Family(
            "nbar",
            ("ndmi_state", "nbr_trend"),
            tuple(COND_STATS),
            MODIS_NBAR_SCALE_M,
            "cond",
        ),
        "fraction": Family(
            "fraction",
            tuple(FRACTION_BANDS),
            tuple(COND_STATS),
            FRACTION_SCALE_M,
            "cond",
        ),
        # Degraded anchor (D1): baseline-window state, fine cells only. Separate
        # families so reference-run tables are completed rather than invalidated.
        "ndvi_base": Family(
            "ndvi_base", ("ndvi_base",), None, MODIS_VI_SCALE_M, "mean"
        ),
        "nbar_base": Family(
            "nbar_base", ("ndmi_base",), None, MODIS_NBAR_SCALE_M, "mean"
        ),
        "rue": Family("rue", ("rue_trend",), None, MODIS_VI_SCALE_M, "mean"),
        "viirs": Family("viirs", ("viirs",), None, VIIRS_REDUCE_SCALE_M, "mean"),
        "canopy": Family(
            "canopy", ("canopy_height",), None, CANOPY_HEIGHT_REDUCE_SCALE_M, "median"
        ),
        "labels": Family(
            "labels",
            ("pnv_biome" if ctx.class_fallback else "vecea_class", "wrb_group"),
            None,
            SITE_SELECTION_INDICATOR_SCALE_M,
            "hist",
        ),
    }


def family_image_fn(name: str, ctx: RunContext) -> Any:
    """Zero-argument image builder for a family name."""
    return {
        "stage1": lambda: build_gate_indicators(ctx),
        "climate": lambda: build_climate_stack(ctx),
        "env": lambda: build_terrain_soil_stack(ctx),
        "ndvi": lambda: build_ndvi_metrics(ctx),
        "nbar": lambda: build_nbar_metrics(ctx),
        "ndvi_base": lambda: build_baseline_state(ctx, "ndvi"),
        "nbar_base": lambda: build_baseline_state(ctx, "ndmi"),
        "fraction": lambda: build_fraction_image(ctx),
        "rue": lambda: build_rue_trend(ctx),
        "viirs": lambda: build_viirs(ctx),
        "canopy": lambda: build_canopy(ctx),
        "labels": lambda: build_label_image(ctx),
    }[name]


def ensure_embeddings(
    store: TableStore, cells: pd.DataFrame, year: int, ctx: RunContext
) -> pd.DataFrame:
    """Mean 64-D embedding per cell for one year (memoised by (cell_id, year)).

    Args:
        store: TableStore.
        cells: Cells (index cell_id; ix, iy, size).
        year: Embedding year.
        ctx: RunContext.

    Returns:
        DataFrame indexed by cell_id with A00..A63 for the requested cells.
    """
    emb = store.embeddings
    have = set()
    if len(emb):
        have = set(emb.loc[emb["year"] == year, "cell_id"])
    todo = [c for c in cells.index if c not in have]
    if todo:
        df = reduce_cells(
            build_embedding_image(ctx, year),
            cells.loc[todo],
            bands=SATELLITE_EMBEDDING_BANDS,
            stats=None,
            reducer=ee.Reducer.mean(),
            scale=EMBEDDING_REDUCE_SCALE_M,
            crs=ctx.crs,
            tile_scale=ctx.tile_scale,
        )
        df = df.reset_index()
        df.insert(1, "year", int(year))
        store.embeddings = pd.concat([emb, df], ignore_index=True) if len(emb) else df
        store.reductions.append(
            {
                "level": "fine",
                "family": f"embedding_{year}",
                "cells": len(todo),
                "scale": EMBEDDING_REDUCE_SCALE_M,
            }
        )
    emb = store.embeddings
    sub = emb.loc[(emb["year"] == year) & emb["cell_id"].isin(list(cells.index))]
    return sub.drop(columns="year").set_index("cell_id").reindex(cells.index)


def region_mean_embedding(
    ctx: RunContext, region: ee.Geometry, year: int
) -> np.ndarray:
    """Mean embedding over an arbitrary region (focal site or seeds), unmasked.

    Args:
        ctx: RunContext.
        region: ee.Geometry.
        year: Embedding year.

    Returns:
        (64,) numpy vector (NaN where unavailable).
    """
    vals = (
        build_embedding_image(ctx, year, masked=False)
        .reduceRegion(
            ee.Reducer.mean(),
            region,
            scale=EMBEDDING_REDUCE_SCALE_M,
            crs=ctx.crs,
            tileScale=ctx.tile_scale,
            maxPixels=int(1e10),
        )
        .getInfo()
    )
    vals = cast(dict, vals or {})
    return np.array(
        [np.nan if vals.get(b) is None else vals[b] for b in SATELLITE_EMBEDDING_BANDS],
        float,
    )


def reduce_dw_context(ctx: RunContext, cells: pd.DataFrame) -> pd.DataFrame:
    """Dynamic World bare/crops/built context for ranked cells and the focal site.

    Args:
        ctx: RunContext.
        cells: Ranked cells (index cell_id; ix, iy, size).

    Returns:
        DataFrame indexed by cell_id (plus a '__focal__' row) with dw_bare, dw_crops,
        dw_built.
    """
    img = build_dw_context(ctx)
    bands = [f"dw_{b}" for b in DYNAMIC_WORLD_CONTEXT_BANDS]
    fc = cells_feature_collection(cells, ctx.crs).merge(
        ee.FeatureCollection([ee.Feature(ctx.focal_ee, {"cell_id": "__focal__"})])
    )
    out = img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=DYNAMIC_WORLD_CONTEXT_SCALE_M,
        crs=ctx.crs,
        tileScale=ctx.tile_scale,
    )
    df = _pull(out).set_index("cell_id")
    for b in bands:
        if b not in df.columns:
            df[b] = np.nan
    return df[bands]


# --------------------------------------------------------------------------- #
# Focal profile                                                                #
# --------------------------------------------------------------------------- #
CATEGORICAL_NATIVE_SCALE = {
    "vecea_class": 92,
    "wrb_group": 230,
    "pnv_biome": 1000,
    "eco_id": 1000,
    "texture_class": 30,
    "hand_riparian": 90,
}


def _hist_shares(hist: dict | None) -> dict[int, float]:
    if not hist:
        return {}
    total = float(sum(hist.values()))
    if total <= 0:
        return {}
    return {int(float(k)): v / total for k, v in hist.items()}


@dataclass
class FocalProfile:
    """Raw focal statistics pulled in one request (see :func:`profile_focal`)."""

    native_counts: dict[str, float]
    shares: dict[str, dict[int, float]]
    buffered: dict[str, bool]
    elevation: dict[str, float]
    rainfall: dict[str, float]
    coast_dist: float | None
    mess_reference: pd.DataFrame
    span_samples: pd.DataFrame


def profile_focal(
    focal_ee: ee.Geometry,
    crs: str,
    *,
    asset_ids: dict[str, str],
    min_focal_class_pixels: int,
    reference_buffer_m: float,
    n_samples: int,
    analysis_scale_m: int,
    coastal_check: bool,
    mess_region: ee.Geometry | None = None,
) -> FocalProfile:
    """Pull every focal statistic the run needs in a single request.

    Categorical shares are computed on the same 90 m UTM label images as the
    candidates (focal site, and focal + reference_buffer_m); native pixel counts
    decide which to use (fewer than ``min_focal_class_pixels`` -> buffered). The MESS
    reference sample takes each predictor at its native scale over a buffer large
    enough to hold real pixels: ``max(reference_buffer_m, 1.5 * native_px * 3)``.

    Args:
        focal_ee: Focal geometry.
        crs: Focal UTM CRS.
        asset_ids: Resolved uploaded-asset IDs.
        min_focal_class_pixels: Native-pixel threshold for buffered shares.
        reference_buffer_m: Profile buffer (m).
        n_samples: Pixels sampled per predictor group.
        analysis_scale_m: Scale of the joint span sample.
        coastal_check: Compute the focal distance to the coast.
        mess_region: Region for the MESS reference (seed sites); default the focal site.

    Returns:
        FocalProfile.
    """
    from eetools.constants import MESS_MIN_NATIVE_PX

    vecea = get_vecea_class(asset_id=asset_ids.get("vecea"))
    wrb = get_soilgrids_wrb(asset_id=asset_ids.get("soilgrids_wrb"))
    hand = get_hand()
    cats = {
        "vecea_class": vecea,
        "wrb_group": wrb,
        "pnv_biome": get_pnv_biome(),
        "eco_id": get_ecoregion_image(focal_ee.buffer(50_000)).select("eco_id"),
        "texture_class": get_soil_texture().select("texture_class"),
        "hand_riparian": hand.lt(HAND_RIPARIAN_MAX_M).rename("hand_riparian"),
    }
    buffered_geom = focal_ee.buffer(reference_buffer_m)
    req: dict[str, Any] = {}
    for name, img in cats.items():
        native = CATEGORICAL_NATIVE_SCALE[name]
        req[f"count_{name}"] = (
            img.reduceRegion(
                ee.Reducer.count(), focal_ee, scale=native, crs=crs, maxPixels=int(1e10)
            )
            .values()
            .get(0)
        )
        for tag, geom in (("focal", focal_ee), ("buffer", buffered_geom)):
            req[f"hist_{name}_{tag}"] = (
                img.reduceRegion(
                    ee.Reducer.frequencyHistogram(),
                    geom,
                    scale=SITE_SELECTION_INDICATOR_SCALE_M,
                    crs=crs,
                    maxPixels=int(1e10),
                )
                .values()
                .get(0)
            )
    dem = get_copernicus_dem()
    req["elevation"] = dem.reduceRegion(
        mean_percentile_reducer([5, 50, 95]),
        focal_ee,
        scale=30,
        crs=crs,
        maxPixels=int(1e10),
    )
    rain = get_chirps_climatology(focal_ee.buffer(20_000))
    req["rain_mean"] = rain.reduceRegion(
        ee.Reducer.mean(), focal_ee, scale=CHIRPS_NATIVE_SCALE_M / 4, crs=crs
    ).get("rain_mean_annual")
    req["rain_pct"] = rain.reduceRegion(
        ee.Reducer.percentile([5, 95]),
        focal_ee.buffer(reference_buffer_m),
        scale=1000,
        crs=crs,
    )
    if coastal_check:
        req["coast_dist"] = (
            get_coast_distance(focal_ee, crs)
            .reduceRegion(ee.Reducer.mean(), focal_ee, scale=1000, crs=crs)
            .get("coast_dist")
        )

    env = build_environment_stack(
        coastal=coastal_check, crs=crs, region=focal_ee.buffer(30_000)
    )
    region = mess_region if mess_region is not None else focal_ee
    groups: dict[float, list[str]] = {}
    preds = list(PREDICTOR_NATIVE_SCALE_M)
    if not coastal_check:
        preds.remove("coast_dist")
    for p in preds:
        groups.setdefault(float(PREDICTOR_NATIVE_SCALE_M[p]), []).append(p)
    for scale, names in groups.items():
        buf = max(reference_buffer_m, 1.5 * scale * MESS_MIN_NATIVE_PX**0.5)
        samp = env.select(names).sample(
            region=region.buffer(buf),
            scale=scale,
            projection=crs,
            numPixels=n_samples,
            seed=11,
            geometries=False,
        )
        for n in names:
            req[f"mess_{n}"] = samp.aggregate_array(n)
    span = env.select(preds).sample(
        region=focal_ee,
        scale=analysis_scale_m,
        projection=crs,
        numPixels=n_samples,
        seed=13,
        geometries=False,
    )
    for n in preds:
        req[f"span_{n}"] = span.aggregate_array(n)

    info = cast(dict, ee.Dictionary(req).getInfo())

    counts = {k[6:]: float(v or 0) for k, v in info.items() if k.startswith("count_")}
    shares: dict[str, dict[int, float]] = {}
    buffered: dict[str, bool] = {}
    for name in cats:
        use_buffer = counts.get(name, 0) < min_focal_class_pixels
        buffered[name] = use_buffer
        shares[name] = _hist_shares(
            info.get(f"hist_{name}_{'buffer' if use_buffer else 'focal'}")
        )
    elev = info.get("elevation") or {}
    elevation = {
        "median": elev.get("elevation_p50", elev.get("elevation_mean")),
        "p5": elev.get("elevation_p5"),
        "p95": elev.get("elevation_p95"),
        "mean": elev.get("elevation_mean"),
    }
    rp = info.get("rain_pct") or {}
    rainfall = {
        "mean": info.get("rain_mean"),
        "p5": rp.get("rain_mean_annual_p5", rp.get("p5")),
        "p95": rp.get("rain_mean_annual_p95", rp.get("p95")),
    }
    mess_rows = []
    for n in preds:
        for v in info.get(f"mess_{n}") or []:
            mess_rows.append({"predictor": n, "value": v})
    mess_long = pd.DataFrame(mess_rows, columns=["predictor", "value"])
    span_df = pd.DataFrame(
        {n: pd.Series(info.get(f"span_{n}") or [], dtype=float) for n in preds}
    )
    if span_df.empty:
        warnings.warn(
            "Focal span sample is empty; the focal site may be smaller than one pixel.",
            stacklevel=2,
        )
    return FocalProfile(
        native_counts=counts,
        shares=shares,
        buffered=buffered,
        elevation={
            k: (float(v) if v is not None else np.nan) for k, v in elevation.items()
        },
        rainfall={
            k: (float(v) if v is not None else np.nan) for k, v in rainfall.items()
        },
        coast_dist=(
            None if info.get("coast_dist") is None else float(info["coast_dist"])
        ),
        mess_reference=mess_long,
        span_samples=span_df,
    )


def shapely_to_ee(geom: shapely.Geometry) -> ee.Geometry:
    """Convert a lon/lat shapely geometry to an ee.Geometry (planar=False)."""
    return ee.Geometry(shapely.geometry.mapping(geom), "EPSG:4326", False)
