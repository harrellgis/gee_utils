"""User-facing workflows: identify_reference_sites and identify_degraded_sites.

Reduce once, score on features, rank in Python: each Earth Engine reduction is pulled
into ScreeningTables once, and every gate, MESS score, percentile, rank, thinning and
spanning step then runs in pandas over those tables (zero EECU). See the 2026-11-05
simplified spec (vault ``raw/notes/gee_utils``) for the design rationale.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from eetools.constants import (
    ANCHOR_INDICES,
    ANCHOR_REVIEW_PCT,
    CHIRPS_PENTAD_COLLECTION,
    COARSE_MIN_SHARE_FLOOR,
    COAST_PREDICTOR,
    COASTAL_MIN_LIMIT_M,
    COASTAL_TRIGGER_M,
    DEFAULT_EXCLUDE_BUFFER_M,
    DEFAULT_FOCAL_SAMPLES,
    DEFAULT_REFERENCE_BUFFER_M,
    DEGRADATION_TYPE_PROFILES,
    DEGRADED_MAX_BUILT_FRAC,
    DEGRADED_MAX_DW_BUILT,
    DW_BUILT_SHORTLIST_FACTOR,
    DW_COLLECTION,
    DYNAMIC_WORLD_FIRST_YEAR,
    EMBEDDING_FIRST_YEAR,
    EMBEDDING_LATEST_YEAR,
    ENV_PREDICTORS,
    ESA_WC_COLLECTION,
    HANSEN_GFC_COLLECTION,
    HUMAN_MODIFICATION_COLLECTION,
    ISDA_SAND_IMAGE,
    ISDA_TEXTURE_IMAGE,
    MERIT_HYDRO_IMAGE,
    MESS_RANGE_FLOORS,
    MODIS_NBAR_COLLECTION,
    MODIS_NBAR_SCALE_M,
    MODIS_VI_COLLECTION,
    PA_EXCLUDE_MIN_SHARE,
    PATHWAY_GATE_PCT,
    PNV_BIOME_IMAGE,
    RECENT_LOSS_YEARS,
    RESOLVE_ECOREGIONS_COLLECTION,
    SATELLITE_EMBEDDING_COLLECTION,
    SBTN_NATURAL_LANDS_IMAGE,
    STATE_WINDOW_YEARS,
    TERRACLIMATE_COLLECTION,
    TREND_WINDOW_YEARS,
    VECEA_DOI,
    VECEA_EXCLUDED_CLASSES,
    VECEA_MOSAIC_CLASSES,
    VECEA_VERSION,
    VIIRS_NTL_COLLECTION,
    WDPA_POLYGONS_COLLECTION,
    WDPA_STRICT_MAX_SEED_CELLS,
    WDPA_STRICT_MIN_CELLS,
    WDPA_STRICT_MIN_SHARE,
    WORLDCLIM_BIO_IMAGE,
    WORLDCOVER_YEAR,
)
from eetools.sensors.assets import check_required_assets
from eetools.sensors.soilgrids.preprocessing import wrb_name, wrb_related_codes
from eetools.sensors.vecea.preprocessing import (
    vecea_class_code,
    vecea_group,
    vecea_group_members,
)
from eetools.site_selection import predictors as P
from eetools.site_selection.domain import (
    Rung,
    add_focal_distance,
    apply_exclusions,
    build_candidate_grid,
    cell_geodataframe,
    derive_cell_size,
    geodesic_area_m2,
    geojson_of,
    geometry_hash,
    load_focal_geometry,
    refine_cells,
    search_rungs,
    to_lonlat,
    to_utm,
    utm_projection_for,
)
from eetools.site_selection.scoring import (
    MatrixContext,
    anchor_check,
    anchor_constraint,
    baseline_window,
    build_layer_register,
    build_selection_matrix,
    degradation_score,
    degraded_exclusions,
    dw_built_filter,
    focal_baseline_pct,
    forest_stratum,
    intactness_flag,
    intactness_score,
    pathway_gate,
    percentile_normalise,
    rank_candidates,
    select_spanning_set,
    signature_from_rules,
    thin_spatially,
    zero_star_likelihood,
)
from eetools.site_selection.similarity import (
    FocalEnvelope,
    compute_mess,
    drop_collinear,
    embedding_query_similarity,
    envelope_gate,
    mean_embedding_vector,
    resolve_seed_source,
)
from eetools.site_selection.types import (
    COMPLETABLE_KEY_FIELDS,
    DegradationType,
    DegradedWeights,
    EnvelopeTolerances,
    ReferenceWeights,
    ScreeningTables,
    SiteSelectionResult,
    dataclass_params,
)

SURVEY_PENDING = (
    "survey-sensor confirmation pending (run_site_timeseries hand-off: "
    "result_to_feature_collection -> run_site_timeseries with the survey sensor and "
    "its native scale)"
)


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def identify_reference_sites(
    focal_site: Any,
    *,
    end_year: int | None = None,
    seed_reference_sites: ee.FeatureCollection | None = None,
    auto_seeds: Literal["none", "wdpa_strict"] = "none",
    target_cover_classes: tuple[int, ...] | list[int] | None = None,
    n_candidates: int = 5,
    tier_distances_km: tuple[float, float, float] = (25.0, 50.0, 150.0),
    screen_cell_m: float = 2000.0,
    refine_pool: int = 40,
    analysis_scale_m: int = 100,
    screening: ScreeningTables | None = None,
    tolerances: EnvelopeTolerances | None = None,
    weights: ReferenceWeights | None = None,
    use_embeddings: bool = True,
    embed_pool: int = 100,
    natural_mask: Literal["worldcover", "sbtn"] = "worldcover",
    require_protected: bool = False,
    span_environment: bool = True,
    span_score_floor: float = 0.5,
    span_weight: float = 0.5,
    exclude_fc: ee.FeatureCollection | None = None,
    asset_overrides: dict[str, str] | None = None,
    tile_scale: int = 4,
) -> SiteSelectionResult:
    """Propose rank-ordered TGBS reference sites for a focal site.

    Pipeline (spec §5): focal profile and coarse grid in the focal UTM zone; Stage 1
    envelope gate (VECEA national class, SoilGrids WRB substrate, elevation +/-300 m,
    CHIRPS rainfall +/-25%, HAND, coast); Stage 2 MESS on condition-independent
    predictors; Stage 3 condition screen (MODIS NDVI/NDMI/NBR, 250 m fraction image,
    gHM, BII, WDPA); Stage 3a refinement of the best coarse cells into focal-sized
    cells; Stage 3b embedding similarity; ranking, thinning and SOP spanning, all
    repeated per search-ladder rung until enough candidates pass. Only the Earth
    Engine reductions cost EECU, and they are memoised in ``result.screening``.

    Outputs are *proposed* candidates for field verification, never final
    designations. Survey-sensor confirmation (R7) is a separate step:
    ``result_to_feature_collection(result)`` -> ``run_site_timeseries(...,
    scale=<survey sensor native scale>)``.

    Args:
        focal_site: Focal site as ee.Geometry / ee.Feature / ee.FeatureCollection, a
            shapely geometry (EPSG:4326) or a path to a vector file.
        end_year: Last complete year of the condition window (default: last year).
        seed_reference_sites: Known reference sites (Scenario A). They become the MESS
            reference set and the independent embedding query.
        auto_seeds: 'wdpa_strict' uses Stage-1 survivors inside designated IUCN Ia-II
            areas as the embedding query when no seeds are given (Scenario A').
        target_cover_classes: Optional ESA WorldCover codes counted as natural cover
            (default: tree, shrub, grass, herbaceous wetland, mangrove, moss).
        n_candidates: References wanted (>= 3 for the SOP; raised to the number of
            management units).
        tier_distances_km: Outer limits of search tiers 1-3 (km).
        screen_cell_m: Minimum coarse screening cell (m); grows for large focal sites.
        refine_pool: Coarse cells subdivided in Stage 3a.
        analysis_scale_m: Scale of the terrain/soil/HAND MESS predictors (m).
        screening: ScreeningTables from an earlier run on the same focal site.
        tolerances: EnvelopeTolerances (defaults if None).
        weights: ReferenceWeights (defaults if None).
        use_embeddings: Run the Stage 3b embedding step.
        embed_pool: Fine cells sent to the embedding step.
        natural_mask: 'worldcover' (default) or 'sbtn' natural mask.
        require_protected: Keep only cells at least half inside designated WDPA areas.
        span_environment: Choose references that span the focal environment (SOP).
        span_score_floor: Eligible references must be in this top share of the pool.
        span_weight: Coverage vs rank-score trade-off for secondaries.
        exclude_fc: Optional areas to exclude (other projects, concessions).
        asset_overrides: Optional {'vecea': id, 'soilgrids_wrb': id}.
        tile_scale: reduceRegions tileScale.

    Returns:
        SiteSelectionResult with ranked/pool GeoDataFrames, the SOP selection matrix,
        the layer register, params, provenance, warnings and the ScreeningTables.

    Raises:
        SiteSelectionAssetError: If an uploaded asset cannot be read.
        ScreeningTablesMismatch: If ``screening`` was built with different inputs.
        ValueError: For invalid arguments.
    """
    return _run(
        "reference",
        focal_site,
        end_year=end_year,
        seed_reference_sites=seed_reference_sites,
        auto_seeds=auto_seeds,
        target_cover_classes=target_cover_classes,
        n_candidates=n_candidates,
        tier_distances_km=tier_distances_km,
        screen_cell_m=screen_cell_m,
        refine_pool=refine_pool,
        analysis_scale_m=analysis_scale_m,
        screening=screening,
        tolerances=tolerances,
        ref_weights=weights,
        use_embeddings=use_embeddings,
        embed_pool=embed_pool,
        natural_mask=natural_mask,
        require_protected=require_protected,
        span_environment=span_environment,
        span_score_floor=span_score_floor,
        span_weight=span_weight,
        exclude_fc=exclude_fc,
        asset_overrides=asset_overrides,
        tile_scale=tile_scale,
    )


def identify_degraded_sites(
    focal_site: Any,
    *,
    project_start_year: int,
    end_year: int | None = None,
    degradation_type: DegradationType = "generic",
    signature_mode: Literal["embedding", "rules"] = "embedding",
    n_candidates: int = 3,
    tier_distances_km: tuple[float, float, float] = (25.0, 50.0, 150.0),
    screen_cell_m: float = 2000.0,
    refine_pool: int = 40,
    analysis_scale_m: int = 100,
    screening: ScreeningTables | None = None,
    tolerances: EnvelopeTolerances | None = None,
    weights: DegradedWeights | None = None,
    embed_pool: int = 100,
    pathway_gate_pct: float = PATHWAY_GATE_PCT,
    anchor_review_pct: float = ANCHOR_REVIEW_PCT,
    reference_result: SiteSelectionResult | None = None,
    max_built_frac: float = DEGRADED_MAX_BUILT_FRAC,
    max_dw_built: float | None = DEGRADED_MAX_DW_BUILT,
    natural_mask: Literal["worldcover", "sbtn"] = "worldcover",
    exclude_protected: bool = False,
    exclude_fc: ee.FeatureCollection | None = None,
    asset_overrides: dict[str, str] | None = None,
    tile_scale: int = 4,
) -> SiteSelectionResult:
    """Propose rank-ordered TGBS degraded (zero-star) sites for a focal site.

    The degraded site is the 0% anchor of the TGBS rescale (reference = 100%,
    degraded = 0%; Manual §4.1-4.2), not a look-alike of the focal baseline. After
    the same envelope, MESS, condition screen and search ladder as
    :func:`identify_reference_sites`, fine cells go through constrain -> gate -> rank:

    1. **Anchor constraint (D1):** on NDVI and NDMI, the candidate's baseline-window
       state <= the focal site's, and its current state <= min(focal baseline,
       focal current). The baseline window is the 3 complete years before
       ``project_start_year``.
    2. **Pathway gate (D2):** pct(signature_sim) >= ``pathway_gate_pct``. The
       signature is the cosine similarity of mean Satellite Embeddings in
       ``project_start_year - 1`` (``signature_mode='embedding'``), or a rule-based
       profile of ``degradation_type`` (``'rules'``; forced, with a warning, for
       baselines before 2017). It filters on the *kind* of degradation; it never
       ranks, because a weight would penalise sites further along the same pathway.
    3. **Rank** on STATE (productivity and moisture deficit, unnaturalness, pressure)
       + TRAJECTORY (browning, NBR decline, loss).

    Each ranked site also reports ``focal_baseline_pct_<x>`` (where the focal
    baseline falls on that site's 0-100% scale) and ``anchor_check='review'`` when
    any exceeds ``anchor_review_pct``. If nothing passes D1, the most degraded
    candidates are returned with ``below_baseline=False`` and a warning. Permanent
    water is excluded. Unless ``degradation_type='settlement'``, built-up land is
    excluded too, because GBS focal sites hold no built structures (cropland stays
    acceptable): final cells need ESA WorldCover built-up <= ``max_built_frac``, and
    the top ``DW_BUILT_SHORTLIST_FACTOR x n_candidates`` thinned cells of each rung
    need a Dynamic World built-label share <= ``max_dw_built`` (catches built-up land
    newer than WorldCover's 2021 snapshot). Both filters act before the stop rule, so
    the search ladder widens when they empty a rung. Pass
    ``screening=reference_result.screening`` to reuse the reference run's reductions.

    Args:
        focal_site: Focal site (see identify_reference_sites).
        project_start_year: Year restoration started; sets the signature baseline year
            (the year before) and the anchor baseline window (3 years before).
        end_year: Last complete year of the condition window (default: last year).
        degradation_type: Degradation pathway; drives rules-mode signatures and the
            built-up exclusion.
        signature_mode: 'embedding' (default) or 'rules'.
        n_candidates: Degraded sites wanted.
        tier_distances_km: Outer limits of search tiers 1-3 (km).
        screen_cell_m: Minimum coarse screening cell (m).
        refine_pool: Coarse cells subdivided in Stage 3a.
        analysis_scale_m: Scale of the terrain/soil/HAND MESS predictors (m).
        screening: ScreeningTables from an earlier run on the same focal site.
        tolerances: EnvelopeTolerances (defaults if None).
        weights: DegradedWeights (defaults if None).
        embed_pool: Anchor-constraint survivors sent to the embedding (pathway) step.
        pathway_gate_pct: Minimum pct(signature_sim) to pass the pathway gate ([L]).
        anchor_review_pct: focal_baseline_pct above this flags anchor_check='review'
            ([L]).
        reference_result: Optional reference run on the same focal site; its ranked
            sites' mean NDVI/NDMI state is the 100% value for focal_baseline_pct
            (else the pool p90).
        natural_mask: 'worldcover' (default) or 'sbtn'.
        max_built_frac: Maximum ESA WorldCover (2021) built-up fraction of a final
            candidate cell ([L]; ignored for 'settlement').
        max_dw_built: Maximum Dynamic World built-label share (most frequent label
            over the state window) of a shortlisted cell ([L]); None disables the
            Dynamic World filter. Ignored for 'settlement' and before the S2 era.
        exclude_protected: Drop cells at least half inside designated WDPA areas.
        exclude_fc: Optional areas to exclude.
        asset_overrides: Optional {'vecea': id, 'soilgrids_wrb': id}.
        tile_scale: reduceRegions tileScale.

    Returns:
        SiteSelectionResult (site_type 'degraded').

    Raises:
        SiteSelectionAssetError: If an uploaded asset cannot be read.
        ScreeningTablesMismatch: If ``screening`` was built with different inputs.
        ValueError: For invalid arguments or a baseline year without embeddings.
    """
    return _run(
        "degraded",
        focal_site,
        project_start_year=project_start_year,
        end_year=end_year,
        degradation_type=degradation_type,
        signature_mode=signature_mode,
        n_candidates=n_candidates,
        tier_distances_km=tier_distances_km,
        screen_cell_m=screen_cell_m,
        refine_pool=refine_pool,
        analysis_scale_m=analysis_scale_m,
        screening=screening,
        tolerances=tolerances,
        deg_weights=weights,
        embed_pool=embed_pool,
        pathway_gate_pct=pathway_gate_pct,
        anchor_review_pct=anchor_review_pct,
        reference_result=reference_result,
        max_built_frac=max_built_frac,
        max_dw_built=max_dw_built,
        natural_mask=natural_mask,
        exclude_protected=exclude_protected,
        exclude_fc=exclude_fc,
        asset_overrides=asset_overrides,
        tile_scale=tile_scale,
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def resolve_signature_mode(
    project_start_year: int,
    signature_mode: str,
    latest_embedding_year: int = EMBEDDING_LATEST_YEAR,
) -> tuple[int, str, str | None]:
    """Resolve the degraded-signature baseline year and mode (spec §5.3, Q5).

    The baseline is ``project_start_year - 1`` with no clamping: a baseline before
    2017 has no embedding, so the rule-based signature is used with a warning (never
    a post-start embedding).

    Args:
        project_start_year: Year restoration started.
        signature_mode: Requested mode ('embedding' or 'rules').
        latest_embedding_year: Latest available embedding year.

    Returns:
        (baseline_year, mode, warning or None).

    Raises:
        ValueError: If the mode is unknown or the baseline is after the latest
            embedding year in embedding mode.
    """
    if signature_mode not in ("embedding", "rules"):
        raise ValueError("signature_mode must be 'embedding' or 'rules'.")
    baseline = int(project_start_year) - 1
    msg = None
    if signature_mode != "rules" and baseline < EMBEDDING_FIRST_YEAR:
        signature_mode = "rules"
        msg = (
            f"Baseline year {baseline} precedes the first Satellite Embedding year "
            f"({EMBEDDING_FIRST_YEAR}); using the rule-based degradation signature."
        )
    if signature_mode != "rules" and baseline > latest_embedding_year:
        raise ValueError(
            f"No embedding for baseline_year {baseline} (latest "
            f"{latest_embedding_year}); use signature_mode='rules'."
        )
    return baseline, signature_mode, msg


def _majority(hist_json: Any) -> tuple[int | None, float]:
    if not isinstance(hist_json, str):
        return None, float("nan")
    import json

    h = json.loads(hist_json) or {}
    total = sum(h.values())
    if not total:
        return None, float("nan")
    k, v = max(h.items(), key=lambda kv: (kv[1], -float(kv[0])))
    return int(float(k)), v / total


def _focal_sets(prof: P.FocalProfile, tol: EnvelopeTolerances) -> dict[str, Any]:
    """Derive the focal class set, soil group, HAND class and fallbacks."""
    vshares = prof.shares.get("vecea_class", {})
    class_set = sorted(
        c
        for c, s in vshares.items()
        if s >= tol.focal_class_min_share and c not in VECEA_EXCLUDED_CLASSES
    )
    class_fallback = len(class_set) == 0
    groups = {g for c in class_set if (g := vecea_group(c))}
    group_members = sorted(set(vecea_group_members(groups)) | set(class_set))

    def argmax(d: dict[int, float]) -> int | None:
        return max(d.items(), key=lambda kv: (kv[1], -kv[0]))[0] if d else None

    wrb_code = argmax(prof.shares.get("wrb_group", {}))
    return {
        "class_set": class_set,
        "class_groups": sorted(groups),
        "group_members": group_members,
        "class_fallback": class_fallback,
        "class_mosaic": any(c in VECEA_MOSAIC_CLASSES for c in class_set),
        "pnv_code": argmax(prof.shares.get("pnv_biome", {})),
        "eco_id": argmax(prof.shares.get("eco_id", {})),
        "wrb_code": wrb_code,
        "wrb_related": wrb_related_codes(wrb_code) if wrb_code is not None else [],
        "substrate_fallback": wrb_code is None,
        "texture_code": argmax(prof.shares.get("texture_class", {})),
        "hand_riparian": prof.shares.get("hand_riparian", {}).get(1, 0.0) >= 0.5,
    }


def _predictor_frame(tab: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Per-cell MESS predictor values from the raw tables (see spec §5.2)."""
    out = pd.DataFrame(index=tab.index)
    for p in predictors:
        if p in ("bio01", "bio04", "bio12", "bio15"):
            out[p] = tab.get(f"{p}_mean")
        else:
            out[p] = tab.get(p)
    return out.astype(float)


def _condition_view(tab: pd.DataFrame, site_type: str, coarse: bool) -> pd.DataFrame:
    """Canonical condition columns: coarse reference cells use best-patch summaries."""
    v = pd.DataFrame(index=tab.index)
    names = [
        "ndvi_state",
        "ndvi_trend",
        "ndmi_state",
        "nbr_trend",
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
    for n in names:
        v[n] = tab.get(f"{n}_mean")
    if coarse and site_type == "reference":
        v["ghm"] = tab.get("ghm_p10")
        v["core_frac"] = tab.get("core_frac_p90")
        v["ndvi_state"] = tab.get("ndvi_state_p90")
        v["forest_loss_frac"] = tab.get("forest_loss_frac_p10")
    v["ndvi_state_p10"] = tab.get("ndvi_state_p10")
    for single in ("rue_trend", "viirs", "ndvi_base", "ndmi_base"):
        if single in tab.columns:
            v[single] = tab[single]
    if "canopy_height" in tab.columns:
        v["canopy_height_p50"] = tab["canopy_height"]
    return v.apply(pd.to_numeric, errors="coerce")


# --------------------------------------------------------------------------- #
# Core run                                                                     #
# --------------------------------------------------------------------------- #
def _run(
    site_type: str, focal_site: Any, **kw: Any
) -> SiteSelectionResult:  # noqa: C901
    tol: EnvelopeTolerances = kw.get("tolerances") or EnvelopeTolerances()
    ref_w: ReferenceWeights = kw.get("ref_weights") or ReferenceWeights()
    deg_w: DegradedWeights = kw.get("deg_weights") or DegradedWeights()
    n_candidates = int(kw["n_candidates"])
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1.")
    if kw.get("auto_seeds", "none") not in ("none", "wdpa_strict"):
        raise ValueError("auto_seeds must be 'none' or 'wdpa_strict'.")
    if kw["natural_mask"] not in ("worldcover", "sbtn"):
        raise ValueError("natural_mask must be 'worldcover' or 'sbtn'.")
    degradation_type = kw.get("degradation_type", "generic")
    if site_type == "degraded" and degradation_type not in DEGRADATION_TYPE_PROFILES:
        raise ValueError(
            f"degradation_type must be one of {sorted(DEGRADATION_TYPE_PROFILES)}."
        )
    warn: list[str] = []
    end_year = int(kw.get("end_year") or date.today().year - 1)
    trend_start = end_year - TREND_WINDOW_YEARS + 1
    state_start = end_year - STATE_WINDOW_YEARS + 1
    signature_mode = None
    baseline_year = None
    if site_type == "degraded":
        baseline_year, signature_mode, msg = resolve_signature_mode(
            kw["project_start_year"], kw.get("signature_mode", "embedding")
        )
        if msg:
            warn.append(msg)
    base_win: tuple[int, int] | None = None
    if site_type == "degraded":
        for name in ("pathway_gate_pct", "anchor_review_pct", "max_built_frac"):
            if not 0.0 <= float(kw[name]) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if (
            kw.get("max_dw_built") is not None
            and not 0.0 <= float(kw["max_dw_built"]) <= 1.0
        ):
            raise ValueError("max_dw_built must be in [0, 1] or None.")
        base_win = baseline_window(kw["project_start_year"])
        if base_win is None:
            warn.append(
                f"No complete MODIS year precedes project_start_year "
                f"{kw['project_start_year']}: the anchor constraint uses the current "
                "window only (D_now <= F_now)."
            )

    asset_ids = check_required_assets(kw.get("asset_overrides"))

    # ---- Stage 0: focal geometry, grid sizes, exclusions ---------------------
    focal_ll = load_focal_geometry(focal_site)
    crs = utm_projection_for(focal_ll)
    focal_utm = to_utm(focal_ll, crs)
    focal_area_m2 = geodesic_area_m2(focal_ll)
    screen, fine_size = derive_cell_size(focal_area_m2, kw["screen_cell_m"])
    k = int(round(screen / fine_size))
    focal_ee = P.shapely_to_ee(focal_ll)
    excl_utm = focal_utm.buffer(DEFAULT_EXCLUDE_BUFFER_M)
    exclude_ll = None
    if kw.get("exclude_fc") is not None:
        exclude_ll = load_focal_geometry(kw["exclude_fc"])
        excl_utm = shapely.union(excl_utm, to_utm(exclude_ll, crs))
    excl_ll = to_lonlat(excl_utm, crs)
    tiers = tuple(float(t) for t in kw["tier_distances_km"])
    max_dist = tiers[-1] * 1000.0
    search_ee = focal_ee.buffer(max_dist + 2 * screen, 100)

    seeds = kw.get("seed_reference_sites")
    seeds_ll = load_focal_geometry(seeds) if seeds is not None else None
    seeds_ee = P.shapely_to_ee(seeds_ll) if seeds_ll is not None else None

    prof = P.profile_focal(
        focal_ee,
        crs,
        asset_ids=asset_ids,
        min_focal_class_pixels=tol.min_focal_class_pixels,
        reference_buffer_m=DEFAULT_REFERENCE_BUFFER_M,
        n_samples=DEFAULT_FOCAL_SAMPLES,
        analysis_scale_m=kw["analysis_scale_m"],
        coastal_check=tol.coastal_mode != "off",
        mess_region=seeds_ee,
    )
    sets = _focal_sets(prof, tol)
    if sets["class_fallback"]:
        warn.append(
            "The focal site is outside VECEA coverage: the class gate uses OpenLandMap "
            "PNV biome + RESOLVE ECO_ID at [M] confidence."
        )
    if sets["class_mosaic"]:
        warn.append(
            "Focal class "
            + ",".join(vecea_class_code(c) for c in sets["class_set"])
            + " is a mosaic class: the class gate passes everything; separation falls "
            "to substrate, elevation, MESS and the coastal strata ([L])."
        )
    if sets["substrate_fallback"]:
        warn.append(
            "SoilGrids WRB is missing at the focal site: substrate uses iSDA texture class."
        )
    in_use = {
        "vecea_class": not sets["class_fallback"],
        "pnv_biome": sets["class_fallback"],
        "wrb_group": not sets["substrate_fallback"],
        "texture_class": sets["substrate_fallback"],
    }
    for layer, buf in prof.buffered.items():
        if buf and in_use.get(layer, False):
            warn.append(
                f"Focal site covers < {tol.min_focal_class_pixels} native {layer} pixels; "
                f"class shares use focal + {DEFAULT_REFERENCE_BUFFER_M:.0f} m ([M])."
            )
    coastal = False
    coastal_limit = None
    if tol.coastal_mode != "off" and prof.coast_dist is not None:
        coastal = tol.coastal_mode == "on" or prof.coast_dist <= COASTAL_TRIGGER_M
        if coastal:
            coastal_limit = max(2 * prof.coast_dist, COASTAL_MIN_LIMIT_M)
    elev = prof.elevation
    rain = prof.rainfall
    if np.isnan(elev["median"]) or rain["mean"] is None or np.isnan(rain["mean"]):
        raise ValueError(
            "Could not read focal elevation or rainfall; check the geometry."
        )
    tcc = kw.get("target_cover_classes")
    is_forest = forest_stratum(
        sets["class_groups"],
        class_fallback=sets["class_fallback"],
        pnv_code=sets["pnv_code"],
        target_cover_classes=tcc,
    )

    ctx = P.RunContext(
        focal_ee=focal_ee,
        exclusion_ee=P.shapely_to_ee(excl_ll),
        search_ee=search_ee,
        crs=crs,
        end_year=end_year,
        trend_start_year=trend_start,
        state_start_year=state_start,
        asset_ids=asset_ids,
        analysis_scale_m=int(kw["analysis_scale_m"]),
        tile_scale=int(kw["tile_scale"]),
        natural_mask=kw["natural_mask"],
        target_cover_classes=tuple(tcc) if tcc else None,
        coastal=coastal,
        class_set=sets["class_set"],
        group_members=sets["group_members"],
        class_fallback=sets["class_fallback"],
        pnv_code=sets["pnv_code"],
        eco_id=sets["eco_id"],
        wrb_code=sets["wrb_code"],
        wrb_related=sets["wrb_related"],
        substrate_fallback=sets["substrate_fallback"],
        texture_code=sets["texture_code"],
        hand_riparian=sets["hand_riparian"],
        baseline_start_year=base_win[0] if base_win else None,
        baseline_end_year=base_win[1] if base_win else None,
    )
    if site_type == "degraded":
        kw["_focal_anchor"] = _focal_anchor(ctx, base_win, warn)
        kw["_reference_state"] = _reference_state(kw.get("reference_result"))

    key = {
        "focal_geometry": geometry_hash(focal_ll),
        "exclude_fc": geometry_hash(exclude_ll),
        "exclude_buffer_m": DEFAULT_EXCLUDE_BUFFER_M,
        "grid_crs": crs,
        "screen_m": screen,
        "k": k,
        "tier_distances_km": list(tiers),
        "end_year": end_year,
        "trend_start_year": trend_start,
        "state_start_year": state_start,
        "analysis_scale_m": int(kw["analysis_scale_m"]),
        "natural_mask": kw["natural_mask"],
        "target_cover_classes": sorted(tcc) if tcc else None,
        "asset_ids": asset_ids,
        "datasets": _dataset_ids(),
        "focal_class_min_share": tol.focal_class_min_share,
        "min_focal_class_pixels": tol.min_focal_class_pixels,
        "reference_buffer_m": DEFAULT_REFERENCE_BUFFER_M,
        "coastal": coastal,
        **{f"focal_{k_}": v for k_, v in sets.items()},
        "qa_rules": "MOD13Q1 SummaryQA<=1; MCD43A4 mandatory QA<=1 on b2/6/7, 8-day subsample",
        # Sets the ndvi_base / ndmi_base window (anchor constraint). Completable: a
        # reference run leaves it None and its tables are completed, not rejected.
        "project_start_year": (
            int(kw["project_start_year"]) if site_type == "degraded" else None
        ),
    }
    screening = kw.get("screening")
    if screening is not None:
        screening.check_key(key)
        tables = screening
        warn.append(
            f"Raw reductions reused from ScreeningTables created {screening.created}."
        )
    else:
        tables = ScreeningTables.empty(key)
    store = P.TableStore(tables)
    if screening is not None:
        merged = dict(store.key)
        for field_ in COMPLETABLE_KEY_FIELDS:
            if merged.get(field_) is None and key.get(field_) is not None:
                merged[field_] = key[field_]
        store.key = merged
    fams = P.families(ctx)

    def ensure(level: str, name: str, cells: pd.DataFrame) -> None:
        P.ensure_family(
            store, level, fams[name], cells, P.family_image_fn(name, ctx), ctx
        )

    grid = build_candidate_grid(focal_utm, max_dist, screen, tiers)
    grid = grid[grid["tier"] > 0]
    grid = apply_exclusions(grid, excl_utm, crs, tol.exclude_max_share).sort_index()

    coarse_min_share = tol.coarse_min_share
    if coarse_min_share is None:
        coarse_min_share = max(focal_area_m2 / screen**2, COARSE_MIN_SHARE_FLOOR)
    min_screen_pool = tol.min_screen_pool or 3 * n_candidates
    coarse_rule = k > 1
    focal_env = FocalEnvelope(
        elevation=float(elev["median"]),
        rainfall=float(rain["mean"]),
        class_mosaic=sets["class_mosaic"],
        class_fallback=sets["class_fallback"],
        substrate_fallback=sets["substrate_fallback"],
        coastal_limit_m=coastal_limit,
    )
    predictors = list(ENV_PREDICTORS) + ([COAST_PREDICTOR] if coastal else [])
    ref_vals = {
        p: prof.mess_reference.loc[prof.mess_reference["predictor"] == p, "value"]
        .astype(float)
        .to_numpy()
        for p in predictors
    }
    extra_layers = (
        set(DEGRADATION_TYPE_PROFILES[degradation_type]["layers"])
        if site_type == "degraded"
        else set()
    )
    need_rue = "rue" in extra_layers
    need_viirs = "viirs" in extra_layers and signature_mode == "rules"
    need_canopy = is_forest or ("canopy" in extra_layers and signature_mode == "rules")
    rung_log: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None

    for rung in search_rungs(tol.relaxation):
        sub_level = tol.substrate_match
        if rung.relax_substrate and sub_level == "same":
            sub_level = "related"
        cls_level = "group" if rung.relax_class else tol.class_match
        relaxed = set()
        if sub_level != tol.substrate_match:
            relaxed.add("substrate")
        if cls_level != tol.class_match or tol.class_match == "group":
            relaxed.add("national_class")
        if sets["class_fallback"]:
            relaxed.add("national_class")
        if tol.substrate_match == "related":
            relaxed.add("substrate")

        pool_cells = grid[grid["tier"] <= rung.max_tier]
        ensure("coarse", "stage1", pool_cells)
        ensure("coarse", "climate", pool_cells)
        tab = pool_cells.join(store.coarse, how="left")
        gate = envelope_gate(
            tab,
            focal_env,
            class_level=cls_level,
            substrate_level=sub_level,
            elevation_tol_m=tol.elevation_tol_m,
            rainfall_tol_frac=tol.rainfall_tol_frac,
            hydro_match=tol.hydro_match,
            coarse_rule=coarse_rule,
            coarse_min_share=coarse_min_share,
        )
        surv = tab.loc[gate["passes"]].join(gate.drop(columns="passes"))
        entry: dict[str, Any] = {
            "rung": rung.label,
            "cells": int(len(pool_cells)),
            "stage1_survivors": int(len(surv)),
        }
        rung_log.append(entry)
        is_last = rung.index == len(search_rungs(tol.relaxation)) - 1
        if len(surv) < min_screen_pool and not is_last:
            entry["outcome"] = f"abandoned (< min_screen_pool {min_screen_pool})"
            continue
        if len(surv) == 0:
            entry["outcome"] = "no Stage 1 survivors"
            final = None
            continue

        res = _score_rung(
            site_type,
            rung,
            surv,
            grid,
            store,
            ensure,
            ctx,
            kw,
            ref_w,
            deg_w,
            tol,
            focal_env,
            predictors,
            ref_vals,
            is_forest,
            coarse_rule,
            k,
            screen,
            fine_size,
            cls_level,
            sub_level,
            focal_utm,
            excl_utm,
            degradation_type,
            signature_mode,
            baseline_year,
            need_rue,
            need_viirs,
            need_canopy,
            seeds_ee,
            seeds_ll,
        )
        res["relaxed"] = relaxed
        res["rung"] = rung
        res["sub_level"] = sub_level
        res["cls_level"] = cls_level
        entry["thinned"] = int(len(res["thinned"]))
        final = res
        if len(res["thinned"]) >= n_candidates:
            entry["outcome"] = "stop rule met"
            break
        entry["outcome"] = "too few thinned candidates"

    if final is None:
        raise RuntimeError(
            "No candidate passed the envelope gate on any search-ladder rung; widen "
            "tier_distances_km or relax EnvelopeTolerances. Rungs: "
            f"{rung_log}"
        )
    if rung_log[-1].get("outcome") != "stop rule met":
        warn.append(
            "No search-ladder rung met the stop rule; results come from the last rung "
            f"({final['rung'].label}) and may hold fewer than {n_candidates} candidates."
        )
    if final["rung"].index > 0:
        warn.append(
            f"Search ladder reached rung '{final['rung'].label}'; the pool changed from "
            "Tier 1 strict, so Tier-1 ranks can shift."
        )
    for crit in sorted(final["relaxed"]):
        warn.append(
            f"Mandatory criterion '{crit}' relaxed on this rung; affected candidates carry fallback=True."
        )
    warn.extend(final["warnings"])

    return _assemble(
        site_type,
        final,
        kw,
        warn,
        store,
        prof,
        sets,
        ctx,
        tol,
        ref_w,
        deg_w,
        focal_ll,
        focal_area_m2,
        screen,
        fine_size,
        k,
        coastal_limit,
        rung_log,
        is_forest,
        signature_mode,
        baseline_year,
        degradation_type,
        n_candidates,
        coarse_min_share,
        min_screen_pool,
    )


def _focal_anchor(
    ctx: P.RunContext, base_win: tuple[int, int] | None, warn: list[str]
) -> dict[str, float]:
    """Focal NDVI/NDMI in the baseline and current windows (anchor constraint, D1)."""
    vals = P.reduce_focal_anchor_state(ctx)
    for x in ANCHOR_INDICES:
        missing = [
            w
            for w in (("base",) if base_win else ()) + ("state",)
            if np.isnan(vals[f"{x}_{w}"])
        ]
        if missing:
            warn.append(
                f"Focal {x.upper()} unavailable for window(s) {missing}: that part of "
                "the anchor constraint is skipped."
            )
    return vals


def _reference_state(result: SiteSelectionResult | None) -> dict[str, float]:
    """Mean NDVI/NDMI state of a reference run's ranked sites (D4's 100% value)."""
    if result is None:
        return {}
    if result.site_type != "reference":
        raise ValueError("reference_result must come from identify_reference_sites.")
    out: dict[str, float] = {}
    for x in ANCHOR_INDICES:
        col = f"{x}_state"
        if col in result.ranked.columns and len(result.ranked):
            out[x] = float(pd.to_numeric(result.ranked[col], errors="coerce").mean())
    return out


def _dataset_ids() -> dict[str, str]:
    return {
        "chirps_pentad": CHIRPS_PENTAD_COLLECTION,
        "terraclimate": TERRACLIMATE_COLLECTION,
        "worldclim": WORLDCLIM_BIO_IMAGE,
        "resolve": RESOLVE_ECOREGIONS_COLLECTION,
        "ghm": HUMAN_MODIFICATION_COLLECTION,
        "pnv": PNV_BIOME_IMAGE,
        "merit_hydro": MERIT_HYDRO_IMAGE,
        "isda_texture": ISDA_TEXTURE_IMAGE,
        "isda_sand": ISDA_SAND_IMAGE,
        "mod13q1": MODIS_VI_COLLECTION,
        "mcd43a4": MODIS_NBAR_COLLECTION,
        "worldcover": ESA_WC_COLLECTION,
        "hansen": HANSEN_GFC_COLLECTION,
        "wdpa": WDPA_POLYGONS_COLLECTION,
        "embeddings": SATELLITE_EMBEDDING_COLLECTION,
        "sbtn": SBTN_NATURAL_LANDS_IMAGE,
    }


def _score_rung(  # noqa: C901, PLR0913
    site_type: str,
    rung: Rung,
    surv: pd.DataFrame,
    grid: pd.DataFrame,
    store: P.TableStore,
    ensure: Any,
    ctx: P.RunContext,
    kw: dict[str, Any],
    ref_w: ReferenceWeights,
    deg_w: DegradedWeights,
    tol: EnvelopeTolerances,
    focal_env: FocalEnvelope,
    predictors: list[str],
    ref_vals: dict[str, np.ndarray],
    is_forest: bool,
    coarse_rule: bool,
    k: int,
    screen: float,
    fine_size: float,
    cls_level: str,
    sub_level: str,
    focal_utm: shapely.Geometry,
    excl_utm: shapely.Geometry,
    degradation_type: str,
    signature_mode: str | None,
    baseline_year: int | None,
    need_rue: bool,
    need_viirs: bool,
    need_canopy: bool,
    seeds_ee: ee.Geometry | None,
    seeds_ll: shapely.Geometry | None,
) -> dict[str, Any]:
    """Stages 2-3b and ranking for one rung (pure pandas apart from memoised EE pulls)."""
    rw: list[str] = []
    surv = surv.sort_index()
    cells = grid.loc[surv.index]

    # ---- Stage 2: MESS on coarse survivors -----------------------------------
    ensure("coarse", "env", cells)
    tab = _join_new(_join_new(cells, store.coarse), surv)
    pred = _predictor_frame(tab, predictors)
    kept, collinear = drop_collinear(pred, predictors)
    mess = compute_mess(ref_vals, pred, kept)
    tab = tab.join(mess.scores)
    surv2 = tab.loc[tab["mess"] >= tol.mess_floor].copy()
    if len(surv2) == 0:
        surv2 = tab.iloc[0:0]

    # ---- Stage 3: condition screen on coarse survivors -----------------------
    c2 = grid.loc[surv2.index]
    for fam in ("ndvi", "nbar", "fraction"):
        ensure("coarse", fam, c2)
    if need_rue:
        ensure("coarse", "rue", c2)
    if need_viirs:
        ensure("coarse", "viirs", c2)
    craw = _join_new(surv2, store.coarse)
    cpool = craw.copy()
    cview = _condition_view(craw, site_type, coarse=coarse_rule)
    for c in cview.columns:
        cpool[c] = cview[c]
    cpool = _apply_type_exclusions(cpool, site_type, kw, degradation_type, final=False)
    denom = (
        float(
            pd.to_numeric(
                craw.loc[cpool.index, "ndvi_state_mean"], errors="coerce"
            ).quantile(0.9)
        )
        if len(cpool)
        else float("nan")
    )
    denom_ndmi = (
        float(
            pd.to_numeric(
                craw.loc[cpool.index, "ndmi_state_mean"], errors="coerce"
            ).quantile(0.9)
        )
        if len(cpool)
        else float("nan")
    )
    cpool["env_sim"] = percentile_normalise(cpool["mess"])
    cpool["screen_score"] = _screen_score(cpool, site_type, ref_w, deg_w, is_forest)

    # ---- Stage 3a: refine best coarse cells ----------------------------------
    # Thin coarse cells so edge neighbours are not refined twice (diagonals allowed).
    top_ids = thin_spatially(cpool, "screen_score", screen * 1.01)
    top_ids = top_ids[: int(kw["refine_pool"])]
    if k > 1:
        fine = refine_cells(grid.loc[top_ids], k)
        fine = apply_exclusions(fine, excl_utm, ctx.crs, tol.exclude_max_share)
        fine = add_focal_distance(fine, focal_utm).sort_index()
        ensure("fine", "stage1", fine)
        parent_clim = store.coarse.loc[
            fine["parent_id"],
            [
                c
                for c in ("rain_mean_annual", "cwd", "aridity", "coast_dist")
                if c in store.coarse.columns
            ],
        ].set_index(fine.index)
        ftab = _join_new(_join_new(fine, store.fine), parent_clim)
        fgate = envelope_gate(
            ftab,
            focal_env,
            class_level=cls_level,
            substrate_level=sub_level,
            elevation_tol_m=tol.elevation_tol_m,
            rainfall_tol_frac=tol.rainfall_tol_frac,
            hydro_match=tol.hydro_match,
            coarse_rule=False,
        )
        ftab = _join_new(ftab.loc[fgate["passes"]], fgate.drop(columns="passes"))
        fcells = fine.loc[ftab.index]
        level = "fine"
        if k == 8:
            rw.append(
                f"Fine cells ({fine_size:.0f} m) are smaller than one MCD43A4 pixel "
                f"({MODIS_NBAR_SCALE_M} m); NDMI/NBR use area-weighted means."
            )
    else:
        fcells = grid.loc[top_ids]
        ftab = cpool.loc[top_ids].copy()
        level = "coarse"
    for fam in ("env", "ndvi", "nbar", "fraction"):
        ensure(level, fam, fcells)
    if site_type == "degraded" and ctx.baseline_start_year is not None:
        # Anchor constraint (D1): baseline-window state on the final cells only.
        ensure(level, "ndvi_base", fcells)
        ensure(level, "nbar_base", fcells)
    ensure("labels", "labels", fcells)
    if need_rue:
        ensure(level, "rue", fcells)
    if need_viirs:
        ensure(level, "viirs", fcells)
    if need_canopy:
        ensure(level, "canopy", fcells)
    # Fresh store values first (ftab is a snapshot taken before the condition
    # reductions), then the gate evidence, MESS and parent-climate columns from ftab.
    pool = _join_new(_join_new(fcells, store.table(level)), ftab)
    pool = _join_new(pool, store.labels)
    pool = pool.sort_index()

    # MESS on the final pool with the rung's predictor set
    fpred = _predictor_frame(pool, predictors)
    fmess = compute_mess(ref_vals, fpred, kept)
    pool = pool.drop(
        columns=[
            c
            for c in pool.columns
            if c == "mess" or c == "mess_mod" or c.startswith("mess_")
        ],
        errors="ignore",
    )
    pool = pool.join(fmess.scores)
    pool = pool.loc[pool["mess"] >= tol.mess_floor].copy()
    view = _condition_view(pool, site_type, coarse=False)
    for c in view.columns:
        pool[c] = view[c]
    pool = _apply_type_exclusions(pool, site_type, kw, degradation_type, final=True)
    pool["env_sim"] = percentile_normalise(pool["mess"])
    pool["ndvi_performance"] = (
        pool["ndvi_state"] / denom if denom and denom > 0 else np.nan
    )
    for p in predictors:
        pool[f"pred_{p}"] = fpred[p].reindex(pool.index)

    min_sep = 2.0 * (fine_size if k > 1 else screen)
    info: dict[str, Any] = {
        "collinear_dropped": collinear,
        "mess_predictors": kept,
        "mess_range_widened": sorted(set(mess.widened) | set(fmess.widened)),
        "mess_dropped": sorted(set(mess.dropped) | set(fmess.dropped)),
        "ndvi_performance_denominator": denom,
        "ndmi_pool_p90": denom_ndmi,
        "coarse_stage2_survivors": int(len(surv2)),
        "refined_coarse_cells": int(len(top_ids)),
        "min_separation_m": min_sep,
    }

    if site_type == "reference":
        pool = _rank_reference(
            pool,
            store,
            ctx,
            kw,
            ref_w,
            is_forest,
            min_sep,
            cpool,
            grid,
            seeds_ee,
            seeds_ll,
            info,
            rw,
        )
    else:
        pool = _rank_degraded(
            pool,
            store,
            ctx,
            kw,
            deg_w,
            is_forest,
            signature_mode,
            baseline_year,
            degradation_type,
            info,
            rw,
        )

    thinned = (
        rank_candidates(pool, "rank_score", min_sep)
        if len(pool)
        else pool.assign(rank=[])
    )
    if site_type == "degraded":
        thinned = _filter_dw_built(
            thinned,
            pool,
            fcells,
            level,
            store,
            ensure,
            ctx,
            kw,
            degradation_type,
            info,
            rw,
        )
    return {
        "pool": pool,
        "thinned": thinned,
        "level": level,
        "info": info,
        "warnings": rw,
        "stage1_survivors": int(len(surv)),
    }


def _filter_dw_built(
    thinned: pd.DataFrame,
    pool: pd.DataFrame,
    fcells: pd.DataFrame,
    level: str,
    store: P.TableStore,
    ensure: Any,
    ctx: P.RunContext,
    kw: dict[str, Any],
    degradation_type: str,
    info: dict[str, Any],
    rw: list[str],
) -> pd.DataFrame:
    """Dynamic World built-up filter on the rung's degraded shortlist (memoised).

    Only the top ``DW_BUILT_SHORTLIST_FACTOR x n_candidates`` thinned cells are
    reduced; the rest are not candidates on this rung, so a short result lets the
    stop rule widen the search. Adds ``dw_built_share`` to the shortlisted pool rows.
    """
    max_dw = kw.get("max_dw_built")
    if max_dw is None or degradation_type == "settlement" or not len(thinned):
        info["dw_built_filter"] = {"applied": False}
        return thinned
    if ctx.end_year < DYNAMIC_WORLD_FIRST_YEAR:
        info["dw_built_filter"] = {"applied": False, "reason": "end_year before S2 era"}
        rw.append(
            "Dynamic World built-up filter skipped (end_year before the S2 era); only the "
            "WorldCover built-up filter applies."
        )
        return thinned
    k = DW_BUILT_SHORTLIST_FACTOR * int(kw["n_candidates"])
    short = thinned.head(k)
    ensure(level, "dw_built", fcells.loc[short.index])
    shares = pd.to_numeric(
        store.table(level).loc[short.index, "dw_built_share"], errors="coerce"
    )
    kept, dropped = dw_built_filter(short, shares, float(max_dw))
    pool.loc[short.index, "dw_built_share"] = shares
    info["dw_built_filter"] = {
        "applied": True,
        "max_dw_built": float(max_dw),
        "shortlist": int(len(short)),
        "dropped": int(len(dropped)),
        "kept": int(len(kept)),
    }
    if dropped:
        rw.append(
            f"Dynamic World built-up filter dropped {len(dropped)} of {len(short)} "
            f"shortlisted cells (built-label share > {max_dw})."
        )
    return kept


def _join_new(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Left-join only the columns of ``right`` that ``left`` lacks (no _done flags)."""
    cols = [
        c for c in right.columns if c not in left.columns and not c.startswith("_done_")
    ]
    return left.join(right[cols], how="left")


def _apply_type_exclusions(
    pool: pd.DataFrame,
    site_type: str,
    kw: dict[str, Any],
    degradation_type: str,
    *,
    final: bool,
) -> pd.DataFrame:
    if site_type == "reference":
        if kw.get("require_protected"):
            pool = pool.loc[pool["pa_share"].fillna(0) >= PA_EXCLUDE_MIN_SHARE]
        return pool
    drop = degraded_exclusions(
        pool,
        final=final,
        max_built_frac=float(kw.get("max_built_frac", DEGRADED_MAX_BUILT_FRAC)),
        keep_built=degradation_type == "settlement",
        exclude_protected=bool(kw.get("exclude_protected")),
    )
    return pool.loc[~drop]


def _screen_score(
    pool: pd.DataFrame,
    site_type: str,
    ref_w: ReferenceWeights,
    deg_w: DegradedWeights,
    is_forest: bool,
) -> pd.Series:
    if len(pool) == 0:
        return pd.Series(dtype=float)
    env = 0.5 + 0.5 * pool["env_sim"].fillna(0.5)
    if site_type == "reference":
        return intactness_score(pool, ref_w, is_forest).score * env
    # Degraded coarse screen: state + trajectory (the pathway signature gates fine
    # cells later; it never ranks).
    score = degradation_score(pool, deg_w, is_forest).score
    tie = 1e-6 * percentile_normalise(
        -pd.to_numeric(pool.get("ndvi_state_p10"), errors="coerce")
    ).fillna(0.5)
    return score * env + tie


def _rank_reference(
    pool: pd.DataFrame,
    store: P.TableStore,
    ctx: P.RunContext,
    kw: dict[str, Any],
    w: ReferenceWeights,
    is_forest: bool,
    min_sep: float,
    cpool: pd.DataFrame,
    grid: pd.DataFrame,
    seeds_ee: ee.Geometry | None,
    seeds_ll: shapely.Geometry | None,
    info: dict[str, Any],
    rw: list[str],
) -> pd.DataFrame:
    sc = intactness_score(pool, w, is_forest)
    pool["intactness"] = sc.score
    for c in sc.components.columns:
        pool[f"c_{c}"] = sc.components[c]
    if sc.missing:
        rw.append(
            f"Intactness components with missing values (scored neutral 0.5): {sc.missing}."
        )
    pool["intactness_flag"] = intactness_flag(pool)
    pool["embed_sim"] = np.nan
    pool["agree"] = np.nan
    pool["embed_disagree"] = False
    seed_source, independent = "none", False
    if kw.get("use_embeddings", True) and len(pool):
        year = min(ctx.end_year, EMBEDDING_LATEST_YEAR)
        pre = pool["intactness"] * (0.5 + 0.5 * pool["env_sim"])
        embed_ids = pre.sort_values(ascending=False).index[: int(kw["embed_pool"])]
        ecells = pool.loc[embed_ids, ["ix", "iy", "size"]]
        emb = P.ensure_embeddings(store, ecells, year, ctx)
        strict = cpool.index[
            pd.to_numeric(cpool.get("pa_strict_share"), errors="coerce").fillna(0)
            >= WDPA_STRICT_MIN_SHARE
        ]
        seed_source, independent = resolve_seed_source(
            seeds_ee is not None,
            kw.get("auto_seeds", "none"),
            len(strict),
            WDPA_STRICT_MIN_CELLS,
        )
        seed_cells: list[str] = []
        if seed_source == "user" and seeds_ee is not None:
            query = P.region_mean_embedding(ctx, seeds_ee, year)
            query = mean_embedding_vector(query[None, :])
        elif seed_source == "wdpa_strict":
            ids = (
                cpool.loc[strict]
                .sort_values(
                    "intactness" if "intactness" in cpool else "screen_score",
                    ascending=False,
                )
                .index[:WDPA_STRICT_MAX_SEED_CELLS]
            )
            semb = P.ensure_embeddings(
                store, grid.loc[ids, ["ix", "iy", "size"]], year, ctx
            )
            query = mean_embedding_vector(semb.to_numpy(float))
            seed_cells = list(ids)
            weak = cpool.loc[ids, "screen_score"].isna().any()
            if weak:
                rw.append(
                    "Some WDPA-strict seed cells lack condition data; check for paper parks."
                )
        else:
            tier1 = [c for c in embed_ids if pool.loc[c, "tier"] == 1]
            seeds_pick = (tier1 or list(embed_ids))[:3]
            query = mean_embedding_vector(emb.loc[seeds_pick].to_numpy(float))
            seed_cells = list(seeds_pick)
            rw.append(
                "Embedding query uses intactness pseudo-seeds (not independent): 'agree' "
                "is reported but not used in rank_score. Supply seed_reference_sites or "
                "auto_seeds='wdpa_strict' for independent evidence."
            )
        sim = embedding_query_similarity(emb, query)
        pool.loc[embed_ids, "embed_sim"] = sim
        excluded = _near_seeds(
            pool.loc[embed_ids], seed_cells, seeds_ll, ctx.crs, min_sep
        )
        eligible = [c for c in embed_ids if c not in excluded]
        if eligible:
            mpct = percentile_normalise(pool.loc[eligible, "mess"])
            spct = percentile_normalise(pool.loc[eligible, "embed_sim"])
            pool.loc[eligible, "agree"] = ((mpct >= 0.75) & (spct >= 0.75)).astype(
                float
            )
        if not independent:
            med = pool.loc[embed_ids, "embed_sim"].median()
            pool.loc[embed_ids, "embed_disagree"] = (
                pool.loc[embed_ids, "embed_sim"] < med
            )
    mult = 1.0 + 0.1 * pool["agree"].fillna(0.0) * (1.0 if independent else 0.0)
    pool["rank_score"] = pool["intactness"] * (0.5 + 0.5 * pool["env_sim"]) * mult
    info["seed_source"] = seed_source
    info["agree_independent"] = independent
    return pool


def _near_seeds(
    cells: pd.DataFrame,
    seed_cells: list[str],
    seeds_ll: shapely.Geometry | None,
    crs: str,
    min_sep: float,
) -> set[str]:
    out = set(c for c in seed_cells if c in cells.index)
    pts = shapely.points(cells["cx"].to_numpy(float), cells["cy"].to_numpy(float))
    if seeds_ll is not None:
        seeds_utm = to_utm(seeds_ll, crs)
        d = shapely.distance(pts, seeds_utm)
        out |= set(cells.index[d < min_sep])
    if seed_cells:
        sc = cells.loc[[c for c in seed_cells if c in cells.index]]
        for _, r in sc.iterrows():
            d = np.hypot(cells["cx"] - r["cx"], cells["cy"] - r["cy"])
            out |= set(cells.index[d < min_sep])
    return out


def _rank_degraded(
    pool: pd.DataFrame,
    store: P.TableStore,
    ctx: P.RunContext,
    kw: dict[str, Any],
    w: DegradedWeights,
    is_forest: bool,
    signature_mode: str | None,
    baseline_year: int | None,
    degradation_type: str,
    info: dict[str, Any],
    rw: list[str],
) -> pd.DataFrame:
    """Constrain -> gate -> rank (spec §5.3, amended §0a).

    D1 anchor constraint on the fine cells, then the pathway signature on the
    survivors (embeddings for the top ``embed_pool`` survivors, or the rules
    profile), the D2 gate, and state + trajectory ranking over what passes. D4
    reports where the focal baseline falls on each candidate's 0-100% scale.
    """
    focal: dict[str, float] = kw.get("_focal_anchor") or {}
    n_candidates = int(len(pool))

    # ---- D1: anchor constraint ------------------------------------------------
    anchor = anchor_constraint(pool, focal)
    for c in anchor.columns:
        pool[c] = anchor[c]
    survivors = pool.loc[pool["below_baseline"]]
    anchor_fallback = bool(n_candidates and len(survivors) == 0)
    if anchor_fallback:
        rw.append(
            "No candidate is at least as degraded as the focal site on NDVI and NDMI "
            "(baseline and current windows): the focal site may be the most degraded "
            "place in the search domain. The most degraded candidates are returned "
            "with below_baseline=False, and the focal baseline will rescale below 0%. "
            "TGBS §4.1 fallback: the lowest-rated site available; review by hand."
        )
        survivors = pool
    pool = survivors.copy()
    n_anchor = int(pool["below_baseline"].sum())

    # ---- Pathway signature on the survivors ------------------------------------
    pool["signature_sim"] = np.nan
    if signature_mode == "rules":
        pool["signature_sim"] = signature_from_rules(
            pool, DEGRADATION_TYPE_PROFILES[degradation_type]["terms"]
        )
    elif len(pool) and baseline_year is not None:
        pre = degradation_score(pool, w, is_forest).score * (
            0.5 + 0.5 * pool["env_sim"]
        )
        embed_ids = pre.sort_values(ascending=False).index[: int(kw["embed_pool"])]
        emb = P.ensure_embeddings(
            store, pool.loc[embed_ids, ["ix", "iy", "size"]], int(baseline_year), ctx
        )
        query = P.region_mean_embedding(ctx, ctx.focal_ee, int(baseline_year))
        if np.isnan(query).any():
            raise ValueError(
                f"No Satellite Embedding over the focal site in {baseline_year}."
            )
        pool.loc[embed_ids, "signature_sim"] = embedding_query_similarity(emb, query)
        pool = pool.loc[embed_ids].copy()

    # ---- D2: pathway gate -------------------------------------------------------
    pool["signature_pct"] = percentile_normalise(pool["signature_sim"])
    pool["pathway_gate"] = pathway_gate(
        pool["signature_pct"], float(kw["pathway_gate_pct"])
    )
    passed = pool["pathway_gate"] == "pass"
    n_gate = int(passed.sum())
    if len(pool) and n_gate == 0:
        rw.append(
            "No candidate passed the pathway gate (pct(signature_sim) >= "
            f"{kw['pathway_gate_pct']}); candidates are ranked without it "
            "(pathway_gate='failed')."
        )
    else:
        pool = pool.loc[passed].copy()

    # ---- Rank: state + trajectory over the survivors --------------------------
    pool["env_sim"] = percentile_normalise(pool["mess"])
    sc = degradation_score(pool, w, is_forest)
    if sc.missing:
        rw.append(
            f"Degradation components with missing values (scored neutral 0.5): {sc.missing}."
        )
    pool["degradation"] = sc.score
    for c in sc.components.columns:
        pool[f"c_{c}"] = sc.components[c]
    pool["rank_score"] = pool["degradation"] * (0.5 + 0.5 * pool["env_sim"])
    pool["degradation_pct"] = percentile_normalise(pool["degradation"])
    pool["zero_star_likelihood"] = zero_star_likelihood(
        pool["degradation_pct"], pool["ndvi_performance"]
    )

    # ---- D4: anchor check (reported, never a filter) ----------------------------
    ref_state = kw.get("_reference_state") or {}
    pool_ref = {
        "ndvi": info.get("ndvi_performance_denominator", np.nan),
        "ndmi": info.get("ndmi_pool_p90", np.nan),
    }
    reference = {x: ref_state.get(x, pool_ref.get(x, np.nan)) for x in ANCHOR_INDICES}
    pcts = focal_baseline_pct(pool, focal, reference)
    for c in pcts.columns:
        pool[c] = pcts[c]
    pool["anchor_check"] = anchor_check(pcts, float(kw["anchor_review_pct"]))

    info["signature_mode"] = signature_mode
    info["baseline_year"] = baseline_year
    info["anchor"] = {
        "indices": list(ANCHOR_INDICES),
        "baseline_window": (
            [ctx.baseline_start_year, ctx.baseline_end_year]
            if ctx.baseline_start_year is not None
            else None
        ),
        "focal": focal,
        "candidates": n_candidates,
        "below_baseline": int(anchor["below_baseline"].sum()) if n_candidates else 0,
        "fallback": anchor_fallback,
        "pathway_gate_pct": float(kw["pathway_gate_pct"]),
        "pathway_gate_passed": n_gate,
        "anchor_survivors_scored": n_anchor,
        "anchor_review_pct": float(kw["anchor_review_pct"]),
        "reference_values": reference,
        "reference_source": (
            "reference_result"
            if ref_state
            else "pool p90 (ndvi_performance denominator)"
        ),
    }
    return pool


def _assemble(  # noqa: C901
    site_type: str,
    final: dict[str, Any],
    kw: dict[str, Any],
    warn: list[str],
    store: P.TableStore,
    prof: P.FocalProfile,
    sets: dict[str, Any],
    ctx: P.RunContext,
    tol: EnvelopeTolerances,
    ref_w: ReferenceWeights,
    deg_w: DegradedWeights,
    focal_ll: shapely.Geometry,
    focal_area_m2: float,
    screen: float,
    fine_size: float,
    k: int,
    coastal_limit: float | None,
    rung_log: list[dict[str, Any]],
    is_forest: bool,
    signature_mode: str | None,
    baseline_year: int | None,
    degradation_type: str,
    n_candidates: int,
    coarse_min_share: float,
    min_screen_pool: int,
) -> SiteSelectionResult:
    pool: pd.DataFrame = final["pool"]
    thinned: pd.DataFrame = final["thinned"].copy()
    relaxed: set[str] = final["relaxed"]
    info = final["info"]
    label_col = "pnv_biome" if sets["class_fallback"] else "vecea_class"

    def labels(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cls = [
            _majority(v)
            for v in df.get(label_col, pd.Series(index=df.index, dtype=object))
        ]
        wrb = [
            _majority(v)
            for v in df.get("wrb_group", pd.Series(index=df.index, dtype=object))
        ]
        df["class_code"] = [c for c, _ in cls]
        df["national_class"] = [
            (
                (vecea_class_code(c) if not sets["class_fallback"] else f"PNV_{c}")
                if c is not None
                else None
            )
            for c, _ in cls
        ]
        df["class_source"] = "pnv_ecoregion" if sets["class_fallback"] else "vecea"
        df["wrb_code"] = [c for c, _ in wrb]
        df["wrb_group"] = [wrb_name(c) if c is not None else None for c, _ in wrb]
        df["requires_confirmation"] = True
        return df

    thinned = labels(thinned)
    units: dict[str, set[str]] = {}
    if sets["class_fallback"]:
        units = {"pnv_ecoregion": {"pnv_ecoregion"}}
    elif sets["class_mosaic"]:
        units = {}
    else:
        units = {vecea_class_code(c): {vecea_class_code(c)} for c in sets["class_set"]}
    group_relaxed = final["cls_level"] == "group"

    def unit_keys(row: pd.Series) -> set[str]:
        if sets["class_fallback"]:
            return (
                {"pnv_ecoregion"} if row.get("pnv_eco_same_mean", 0) >= 0.5 else set()
            )
        code = row.get("class_code")
        if code is None or pd.isna(code):
            return set()
        out = set()
        for c in sets["class_set"]:
            if int(code) == c or (
                group_relaxed and vecea_group(int(code)) == vecea_group(c)
            ):
                out.add(vecea_class_code(c))
        return out

    thinned["unit_keys"] = [unit_keys(r) for _, r in thinned.iterrows()]
    focal_class_label = (
        f"PNV {sets['pnv_code']} / ECO_ID {sets['eco_id']}"
        if sets["class_fallback"]
        else ",".join(vecea_class_code(c) for c in sets["class_set"])
    )
    mctx = MatrixContext(
        site_type=site_type,
        elevation=prof.elevation["median"],
        elevation_p5=_nz(prof.elevation["p5"], prof.elevation["median"]),
        elevation_p95=_nz(prof.elevation["p95"], prof.elevation["median"]),
        rainfall=prof.rainfall["mean"],
        rainfall_p5=_nz(prof.rainfall["p5"], prof.rainfall["mean"]),
        rainfall_p95=_nz(prof.rainfall["p95"], prof.rainfall["mean"]),
        elevation_tol_m=tol.elevation_tol_m,
        rainfall_tol_frac=tol.rainfall_tol_frac,
        focal_area_ha=focal_area_m2 / 1e4,
        is_forest=is_forest,
        class_fallback=sets["class_fallback"],
        class_mosaic=sets["class_mosaic"],
        substrate_fallback=sets["substrate_fallback"],
        substrate_off=final["sub_level"] == "off",
        relaxed_criteria=relaxed,
        focal_class_label=focal_class_label,
        focal_substrate_label=(
            wrb_name(sets["wrb_code"]) if sets["wrb_code"] is not None else "n/a"
        ),
        pool=pool,
    )
    matrix = build_selection_matrix(thinned, mctx)
    span_mode = None
    env_coverage = float("nan")
    unit_status: dict[str, bool] = {}
    if site_type == "reference":
        span_samples = prof.span_samples
        elig = matrix["mandatory_met_relaxed"] if len(matrix) else pd.Series(dtype=bool)
        nz = matrix["no_mandatory_zero"] if len(matrix) else pd.Series(dtype=bool)
        cand = thinned.copy()
        for p in info["mess_predictors"]:
            cand[p] = cand.get(f"pred_{p}")
        pool_preds = pool[[f"pred_{p}" for p in info["mess_predictors"]]].rename(
            columns=lambda c: c[5:]
        )
        sp = select_spanning_set(
            cand,
            span_samples,
            pool_preds,
            info["mess_predictors"],
            n_candidates,
            eligible=elig,
            no_mandatory_zero=nz,
            units=units,
            span_weight=float(kw["span_weight"]),
            span_score_floor=float(kw["span_score_floor"]),
            span_environment=bool(kw["span_environment"]),
        )
        ranked = sp.selected
        span_mode, env_coverage, unit_status = sp.span_mode, sp.env_coverage, sp.units
        warn.extend(sp.warnings)
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        n_met = (
            int(matrix.loc[ranked.index, "mandatory_met"].sum()) if len(ranked) else 0
        )
        n_rel = (
            int(matrix.loc[ranked.index, "mandatory_met_relaxed"].sum())
            if len(ranked)
            else 0
        )
        if n_met < 3:
            warn.append(
                f"Only {n_met} ranked references have mandatory_met ({n_rel} meet "
                "mandatory_met_relaxed). SOP fallbacks (not applied): recovering sites "
                "with known histories, best portions of the project site, pooled "
                "regional data."
            )
        if len(ranked) and (ranked["intactness_flag"].iloc[0] == "weak"):
            warn.append(
                "Even the top reference candidate is weakly intact (intactness_flag='weak'); R2: most intact available."
            )
    else:
        ranked = thinned.head(n_candidates).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        if (
            "anchor_check" in ranked.columns
            and (ranked["anchor_check"] == "review").any()
        ):
            ids = ", ".join(ranked.index[ranked["anchor_check"] == "review"])
            warn.append(
                f"anchor_check='review' for {ids}: the focal baseline sits above "
                f"{kw['anchor_review_pct']:.0%} of the scale for these anchors, so they "
                "may be far more degraded than the focal site's starting point. Confirm "
                "the same vegetation formation and degradation pathway."
            )
        if ctx.end_year >= DYNAMIC_WORLD_FIRST_YEAR and len(ranked):
            try:
                dw = P.reduce_dw_context(ctx, ranked[["ix", "iy", "size"]])
                for c in dw.columns:
                    ranked[c] = dw[c].reindex(ranked.index)
                    if "__focal__" in dw.index:
                        ranked[f"{c}_focal"] = dw.loc["__focal__", c]
            except Exception as exc:  # noqa: BLE001 - context only, never fatal
                warn.append(f"Dynamic World ranked-site context failed: {exc}")
        else:
            warn.append("Dynamic World context skipped (end_year before the S2 era).")
    warn.append(SURVEY_PENDING)

    ranked["rung"] = final["rung"].label
    ranked["dist_to_focal_km"] = ranked["dist_to_focal_m"] / 1000.0
    pool = pool.copy()
    pool["rung"] = final["rung"].label
    pool["dist_to_focal_km"] = pool["dist_to_focal_m"] / 1000.0
    pool["thinned_rank"] = thinned["rank"].reindex(pool.index)
    pool = labels(pool)
    ranked = ranked.drop(columns=["unit_keys"], errors="ignore")
    ranked = ranked.join(
        matrix[["mandatory_met", "mandatory_met_relaxed", "fallback", "matrix_total"]],
        how="left",
    )
    drop_cols = [
        c for c in ranked.columns if c.startswith("_done_") or c.endswith("_v")
    ]
    ranked = ranked.drop(columns=drop_cols)
    pool = pool.drop(
        columns=[c for c in pool.columns if c.startswith("_done_") or c.endswith("_v")]
    )
    ranked_gdf = cell_geodataframe(ranked, ctx.crs)
    pool_gdf = cell_geodataframe(pool, ctx.crs)
    matrix_out = matrix.loc[ranked.index] if len(ranked) else matrix
    matrix_out = matrix_out.assign(rank=ranked["rank"]).reset_index()

    focal_profile = prof.mess_reference.assign(sample_set="mess")
    span_long = (
        prof.span_samples.reset_index()
        .melt(id_vars="index", var_name="predictor", value_name="value")
        .rename(columns={"index": "sample_id"})
    )
    focal_profile = pd.concat(
        [focal_profile, span_long.assign(sample_set="span")], ignore_index=True
    )

    params: dict[str, Any] = {
        "site_type": site_type,
        "focal_geojson": geojson_of(focal_ll),
        "focal_area_ha": focal_area_m2 / 1e4,
        "grid_crs": ctx.crs,
        "screen_cell_m": screen,
        "fine_cell_m": fine_size,
        "k": k,
        "end_year": ctx.end_year,
        "trend_start_year": ctx.trend_start_year,
        "state_start_year": ctx.state_start_year,
        "recent_loss_years": RECENT_LOSS_YEARS,
        "n_candidates": n_candidates,
        "tier_distances_km": list(kw["tier_distances_km"]),
        "refine_pool": kw["refine_pool"],
        "embed_pool": kw.get("embed_pool"),
        "analysis_scale_m": kw["analysis_scale_m"],
        "natural_mask": kw["natural_mask"],
        "target_cover_classes": kw.get("target_cover_classes"),
        "tolerances": dataclass_params(tol),
        "weights": dataclass_params(ref_w if site_type == "reference" else deg_w),
        "coarse_min_share": coarse_min_share,
        "min_screen_pool": min_screen_pool,
        "fine_gate_rule": "in-set share >= 0.5; elevation median within tolerance",
        "coarse_gate_rule": (
            "in-set share >= coarse_min_share; elevation p10-p90 overlaps focal +/- tol"
            if k > 1
            else "k == 1: fine rules apply to the screening grid"
        ),
        "rungs_tried": rung_log,
        "rung_reached": final["rung"].label,
        "relaxed_criteria": sorted(relaxed),
        "class_level": final["cls_level"],
        "substrate_level": final["sub_level"],
        "focal_class_set": [vecea_class_code(c) for c in sets["class_set"]],
        "focal_class_groups": sets["class_groups"],
        "focal_class_shares": {
            vecea_class_code(c): round(s, 4)
            for c, s in prof.shares.get("vecea_class", {}).items()
        },
        "focal_wrb_group": (
            wrb_name(sets["wrb_code"]) if sets["wrb_code"] is not None else None
        ),
        "focal_wrb_shares": {
            wrb_name(c): round(s, 4)
            for c, s in prof.shares.get("wrb_group", {}).items()
        },
        "focal_hand_class": "riparian" if sets["hand_riparian"] else "upland",
        "class_shares_buffered": prof.buffered,
        "native_pixel_counts": prof.native_counts,
        "class_fallback": sets["class_fallback"],
        "substrate_fallback": sets["substrate_fallback"],
        "class_mosaic": sets["class_mosaic"],
        "focal_elevation": prof.elevation,
        "focal_rainfall": prof.rainfall,
        "coastal_mode": tol.coastal_mode,
        "coastal_limit_m": coastal_limit,
        "focal_coast_dist_m": prof.coast_dist,
        "is_forest_stratum": is_forest,
        "mess_range_floors": {k_: list(v) for k_, v in MESS_RANGE_FLOORS.items()},
        "span_mode": span_mode,
        "env_coverage": env_coverage,
        "management_units": unit_status,
        "screening_created": store.created,
        "screening_reused_cells": store.reused_cells,
        "reductions": store.reductions,
        **info,
    }
    if site_type == "reference":
        params.update(
            {
                "auto_seeds": kw.get("auto_seeds"),
                "use_embeddings": kw.get("use_embeddings"),
                "require_protected": kw.get("require_protected"),
                "span_environment": kw.get("span_environment"),
                "span_score_floor": kw.get("span_score_floor"),
                "span_weight": kw.get("span_weight"),
            }
        )
    else:
        params.update(
            {
                "project_start_year": kw.get("project_start_year"),
                "degradation_type": degradation_type,
                "signature_mode": signature_mode,
                "baseline_year": baseline_year,
                "exclude_protected": kw.get("exclude_protected"),
                "pathway_gate_pct": kw.get("pathway_gate_pct"),
                "anchor_review_pct": kw.get("anchor_review_pct"),
                "reference_result_supplied": kw.get("reference_result") is not None,
                "max_built_frac": kw.get("max_built_frac"),
                "max_dw_built": kw.get("max_dw_built"),
            }
        )
    provenance = _provenance(ctx)
    extra = []
    if sets["class_fallback"]:
        extra.append(
            _reg(
                "class fallback: PNV biome + RESOLVE ECO_ID",
                "OpenLandMap PNV v01; RESOLVE 2017",
                "M",
            )
        )
    if sets["substrate_fallback"]:
        extra.append(
            _reg("substrate fallback: iSDA texture class", ISDA_TEXTURE_IMAGE, "M")
        )
    if sets["class_mosaic"]:
        extra.append(_reg("national class too broad (mosaic)", focal_class_label, "L"))
    extra.append(
        _reg(
            "substrate: parent material unverified (GLiM deferred)",
            "SoilGrids WRB 2.0",
            "M",
        )
    )
    extra.append(
        _reg(
            "WorldCover snapshot (degradation after 2021 missed by land-cover terms)",
            f"ESA WorldCover {WORLDCOVER_YEAR}",
            "M",
        )
    )
    extra.append(
        _reg("core_frac kernel", "1 km focal mean of the 250 m natural fraction", "M")
    )
    if kw.get("screening") is not None:
        extra.append(_reg("ScreeningTables reused", f"created {store.created}", "H"))
    if site_type == "degraded":
        anc = info.get("anchor", {})
        extra.append(
            _reg(
                "anchor constraint (D1): NDVI + NDMI, D_base <= F_base and "
                "D_now <= min(F_base, F_now)",
                f"baseline window {anc.get('baseline_window')} (MODIS proxy; repeat "
                "on the survey sensor)",
                "M",
            )
        )
        extra.append(
            _reg(
                f"pathway gate (D2): pct(signature_sim) >= {kw.get('pathway_gate_pct')}",
                "relative threshold, uncalibrated",
                "L",
            )
        )
        if degradation_type != "settlement":
            extra.append(
                _reg(
                    f"built-up filter: ESA WorldCover built_frac <= {kw.get('max_built_frac')} "
                    "(final cells)",
                    "WorldCover v200 (2021 snapshot); GBS focal sites hold no built structures",
                    "L",
                )
            )
            dwf = info.get("dw_built_filter", {})
            extra.append(
                _reg(
                    f"built-up filter: Dynamic World built-label share <= {kw.get('max_dw_built')} "
                    "(shortlist)",
                    (
                        f"most frequent label over the state window; shortlist "
                        f"{dwf.get('shortlist')}, dropped {dwf.get('dropped')}"
                        if dwf.get("applied")
                        else f"not applied ({dwf.get('reason', 'disabled')})"
                    ),
                    "L",
                )
            )
        extra.append(
            _reg(
                f"anchor_check (D4): focal_baseline_pct > {kw.get('anchor_review_pct')}",
                f"100% value from {anc.get('reference_source')}; TGBS App. F gives no "
                "RS star thresholds",
                "L",
            )
        )
    if site_type == "reference":
        conf = {"user": "H", "wdpa_strict": "M", "pseudo": "L"}.get(
            info.get("seed_source", ""), "L"
        )
        extra.append(
            _reg(
                f"embedding query seed_source={info.get('seed_source')}",
                f"agree_independent={info.get('agree_independent')}",
                conf,
            )
        )
    layer_register = build_layer_register(
        provenance, params, date.today().isoformat(), extra
    )

    return SiteSelectionResult(
        site_type=site_type,  # type: ignore[arg-type]
        ranked=ranked_gdf,
        pool=pool_gdf,
        focal_profile=focal_profile,
        params=params,
        provenance=provenance,
        selection_matrix=matrix_out,
        layer_register=layer_register,
        screening=store.to_tables(),
        warnings=list(dict.fromkeys(warn)),
    )


def _nz(v: float | None, default: float) -> float:
    return default if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def _reg(layer: str, version: str, conf: str) -> dict[str, Any]:
    return {
        "layer_or_query": layer,
        "version_or_date": version,
        "extent_or_parameters": "",
        "access_date": date.today().isoformat(),
        "result": "noted",
        "confidence": conf,
    }


def _provenance(ctx: P.RunContext) -> dict[str, str]:
    return {
        "VECEA national vegetation": f"{ctx.asset_ids.get('vecea')} | {VECEA_VERSION} | doi:{VECEA_DOI} | van Breugel et al. 2015; licence terms unresolved (keep private)",
        "SoilGrids WRB": f"{ctx.asset_ids.get('soilgrids_wrb')} | SoilGrids 2.0 | CC BY 4.0, Poggio et al. 2021 (ISRIC)",
        "CHIRPS v3 pentad": f"{CHIRPS_PENTAD_COLLECTION} | climatology 1991-2020",
        "TerraClimate": f"{TERRACLIMATE_COLLECTION} | climatology 1991-2020",
        "WorldClim V1 BIO": WORLDCLIM_BIO_IMAGE,
        "Copernicus DEM GLO-30": "COPERNICUS/DEM/GLO30_2024_1",
        "MERIT Hydro HAND": MERIT_HYDRO_IMAGE,
        "iSDAsoil texture/sand/clay": "ISDASOIL/Africa/v1",
        "OpenLandMap PNV biome": PNV_BIOME_IMAGE,
        "RESOLVE Ecoregions": RESOLVE_ECOREGIONS_COLLECTION,
        "MODIS MOD13Q1": f"{MODIS_VI_COLLECTION} | {ctx.trend_start_year}-{ctx.end_year}",
        "MODIS MCD43A4": f"{MODIS_NBAR_COLLECTION} | {ctx.trend_start_year}-{ctx.end_year}",
        "ESA WorldCover": f"{ESA_WC_COLLECTION} | {WORLDCOVER_YEAR}",
        "Hansen GFC": HANSEN_GFC_COLLECTION,
        "TNC gHM v3": f"{HUMAN_MODIFICATION_COLLECTION} | 2022",
        "BII (bii4africa)": "sat-io BII 1 km",
        "WDPA": f"{WDPA_POLYGONS_COLLECTION} | non-commercial without UNEP-WCMC permission; attribution required",
        "Satellite Embedding V1": f"{SATELLITE_EMBEDDING_COLLECTION} | CC BY 4.0, attribution to Google and Google DeepMind",
        "Dynamic World (ranked-site context only)": DW_COLLECTION,
        "VIIRS (settlement profile only)": VIIRS_NTL_COLLECTION,
        "Meta canopy height (forest strata)": "projects/sat-io/open-datasets/facebook/meta-canopy-height",
    }


# --------------------------------------------------------------------------- #
# Hand-off and export                                                          #
# --------------------------------------------------------------------------- #
def result_to_feature_collection(
    result: SiteSelectionResult, include_focal: bool = True
) -> ee.FeatureCollection:
    """Convert ranked sites (and the focal site) to an ee.FeatureCollection.

    This is the R7 survey-sensor confirmation hand-off: pass the output to
    ``run_site_timeseries`` (or ``run_baseline_assessment``) with the survey sensor's
    builder and its native ``scale`` (10-30 m).

    Args:
        result: SiteSelectionResult.
        include_focal: Prepend the focal site as ``site_role='focal'``.

    Returns:
        ee.FeatureCollection with properties site_name, site_role ('focal',
        'reference' or 'degraded'), rank and cell_id.
    """
    feats = []
    if include_focal:
        feats.append(
            ee.Feature(
                ee.Geometry(result.params["focal_geojson"]),
                {
                    "site_name": "focal",
                    "site_role": "focal",
                    "rank": 0,
                    "cell_id": "focal",
                },
            )
        )
    for cid, row in result.ranked.iterrows():
        feats.append(
            ee.Feature(
                ee.Geometry(shapely.geometry.mapping(row.geometry)),
                {
                    "site_name": f"{result.site_type}_{int(row['rank'])}",
                    "site_role": result.site_type,
                    "rank": int(row["rank"]),
                    "cell_id": str(cid),
                },
            )
        )
    return ee.FeatureCollection(feats)


def export_site_selection(
    result: SiteSelectionResult,
    folder: str | Path,
    project: str = "Project",
    site: str = "Site",
    version: int = 1,
    run_date: str | None = None,
) -> dict[str, Path]:
    """Write a result to local GPKG + CSV files (pure geopandas, no Earth Engine).

    Names follow the SOP convention ``<Project>_<Site>_<Product>_<YYYYMMDD>_v<n>``.

    Args:
        result: SiteSelectionResult.
        folder: Output directory (created if missing).
        project: Project code for file names.
        site: Site code for file names.
        version: Version number.
        run_date: YYYYMMDD stamp (default today).

    Returns:
        Dict of product name -> written path.
    """
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    stamp = run_date or date.today().strftime("%Y%m%d")
    kind = result.site_type

    def name(product: str, ext: str) -> Path:
        return out / f"{project}_{site}_{product}_{stamp}_v{version}.{ext}"

    paths: dict[str, Path] = {}
    gpkg = name(f"{kind}_sites", "gpkg")
    _gpkg_safe(result.ranked).to_file(gpkg, layer="ranked", driver="GPKG")
    _gpkg_safe(result.pool).to_file(gpkg, layer="pool", driver="GPKG")
    paths["gpkg"] = gpkg
    for product, df in (
        (f"{kind}_ranked", pd.DataFrame(result.ranked.drop(columns="geometry"))),
        (f"{kind}_selection_matrix", result.selection_matrix),
        (f"{kind}_layer_register", result.layer_register),
    ):
        p = name(product, "csv")
        df.to_csv(p, index=True)
        paths[product] = p
    return paths


def _gpkg_safe(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    for c in out.columns:
        if c == "geometry":
            continue
        if out[c].dtype == object:
            out[c] = out[c].map(
                lambda v: (
                    v if v is None or isinstance(v, (str, int, float, bool)) else str(v)
                )
            )
    return out
