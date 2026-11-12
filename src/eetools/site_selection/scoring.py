"""Condition scoring, ranking, thinning, spanning, SOP matrix and layer register.

Everything here is pure pandas/numpy: no Earth Engine calls, zero EECU.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from eetools.constants import (
    ANCHOR_INDICES,
    CONVERSION_DEGRADATION_TYPES,
    DEGRADED_BASELINE_WINDOW_YEARS,
    DEGRADED_EXCLUDE_MAX_FRAC,
    DW_LANDUSE_FIRST_YEAR,
    FOREST_MIN_CANOPY_M,
    INTACTNESS_WEAK_NATURAL_FRAC,
    INTACTNESS_WEAK_NATURALNESS,
    LANDUSE_MIN_MODIFIED_SHARE,
    LANDUSE_MIN_VALID_SHARE,
    MODIS_FIRST_FULL_YEAR,
    PA_EXCLUDE_MIN_SHARE,
    PNV_FOREST_STRATUM_BIOMES,
    PREDICTOR_LABELS,
    SOP_MATRIX_THRESHOLDS,
    SPAN_ENV_COVERAGE_SD,
    TOP_CANDIDATES_CONDITION_WEIGHT,
    TOP_CANDIDATES_N,
    VECEA_FOREST_STRATUM_GROUPS,
    ZERO_STAR_HIGH_PCT,
    ZERO_STAR_MAX_PERFORMANCE,
    ZERO_STAR_MODERATE_PCT,
)
from eetools.site_selection.types import DegradedWeights, ReferenceWeights

NEUTRAL = 0.5  # percentile used for a component that is missing for a cell


# --------------------------------------------------------------------------- #
# Percentiles and component scores                                             #
# --------------------------------------------------------------------------- #
def percentile_normalise(values: pd.Series) -> pd.Series:
    """Within-pool percentile in [0, 1] (1 = highest value), ties averaged.

    ``(average rank - 1) / (n - 1)`` so the column always spans 0-1 exactly; a
    single valid value maps to 1.0 and NaN stays NaN.

    Args:
        values: Numeric series.

    Returns:
        Series of percentiles with the same index.
    """
    s = pd.to_numeric(values, errors="coerce").astype(float)
    n = int(s.notna().sum())
    ranks = s.rank(method="average")
    if n <= 1:
        return ranks.where(ranks.isna(), 1.0)
    return (ranks - 1.0) / (n - 1.0)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _has(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns and df[name].notna().any()


@dataclass
class ScoreResult:
    """A weighted score with its components.

    Attributes:
        score: Weighted sum per candidate.
        components: Per-candidate component values (0-1) used in the sum.
        weights: Effective weights after stratum/missing-layer adjustments.
        missing: Components that were NaN for some cells (filled with NEUTRAL).
    """

    score: pd.Series
    components: pd.DataFrame
    weights: dict[str, float]
    missing: list[str] = field(default_factory=list)


def _combine(
    components: dict[str, pd.Series], weights: dict[str, float]
) -> ScoreResult:
    weights = {k: v for k, v in weights.items() if v > 0}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    comp = pd.DataFrame({k: components[k] for k in weights})
    missing = [k for k in comp.columns if comp[k].isna().any()]
    filled = comp.fillna(NEUTRAL)
    score = sum(filled[k] * w for k, w in weights.items())
    return ScoreResult(pd.Series(score, index=comp.index), comp, weights, missing)


def intactness_score(
    df: pd.DataFrame,
    weights: ReferenceWeights | None = None,
    is_forest: bool = False,
) -> ScoreResult:
    """Reference intactness score (spec §5.3), percentiles over the pool in ``df``.

    Components: pct(1 - ghm), pct(natural_frac), pct(core_frac), pct(ndvi_state),
    pct(ndmi_state), pct(min(ndvi_trend, 0)), pct(min(nbr_trend, 0)),
    pct(natural_persist) (Dynamic World persistent-natural share) and the absolute
    pa_overlap. Percentiles only order cells that already passed the absolute
    land-use gate; they never decide eligibility. Forest strata move ``forest_loss_shift`` from core to
    pct(1 - forest_loss_frac) and ``canopy_shift`` from performance to
    pct(canopy_height_p50) (only when canopy height is available).

    Args:
        df: Candidate pool with the condition columns.
        weights: ReferenceWeights (defaults if None).
        is_forest: Forest integrity stratum (see forest_stratum).

    Returns:
        ScoreResult with the intactness score.
    """
    w = weights or ReferenceWeights()
    comps = {
        "naturalness": percentile_normalise(1.0 - _col(df, "ghm")),
        "natural_cover": percentile_normalise(_col(df, "natural_frac")),
        "core": percentile_normalise(_col(df, "core_frac")),
        "performance": percentile_normalise(_col(df, "ndvi_state")),
        "moisture": percentile_normalise(_col(df, "ndmi_state")),
        "persistence": percentile_normalise(_col(df, "ndvi_trend").clip(upper=0)),
        "burn_persist": percentile_normalise(_col(df, "nbr_trend").clip(upper=0)),
        "natural_persist": percentile_normalise(_col(df, "natural_persist")),
        "protection": _col(df, "pa_overlap").fillna(0.0).clip(0, 1),
    }
    wts = {
        "naturalness": w.naturalness,
        "natural_cover": w.natural_cover,
        "core": w.core,
        "performance": w.performance,
        "moisture": w.moisture,
        "persistence": w.persistence,
        "burn_persist": w.burn_persist,
        "natural_persist": w.natural_persist,
        "protection": w.protection,
    }
    if is_forest:
        comps["forest_persistence"] = percentile_normalise(
            1.0 - _col(df, "forest_loss_frac")
        )
        wts["core"] -= w.forest_loss_shift
        wts["forest_persistence"] = w.forest_loss_shift
        if _has(df, "canopy_height_p50"):
            comps["canopy"] = percentile_normalise(_col(df, "canopy_height_p50"))
            wts["performance"] -= w.canopy_shift
            wts["canopy"] = w.canopy_shift
    return _combine(comps, wts)


def signature_from_rules(
    df: pd.DataFrame, terms: Sequence[tuple[str, int]]
) -> pd.Series:
    """Rule-based degradation signature: mean of oriented within-pool percentiles.

    Args:
        df: Candidate pool.
        terms: (column, orientation) pairs; +1 means higher = more like the type.
            Columns absent from ``df`` are skipped.

    Returns:
        Series in [0, 1] (NaN when no term is available for a cell).
    """
    parts = [
        percentile_normalise(_col(df, c) * sign) for c, sign in terms if c in df.columns
    ]
    if not parts:
        return pd.Series(np.nan, index=df.index, name="signature_sim")
    return pd.concat(parts, axis=1).mean(axis=1).rename("signature_sim")


def degradation_score(
    df: pd.DataFrame,
    weights: DegradedWeights | None = None,
    is_forest: bool = False,
) -> ScoreResult:
    """Degraded-site score: STATE + TRAJECTORY (spec §5.3, amended §0a D3).

    The signature is not a ranking term: it is the pathway gate (:func:`pathway_gate`),
    applied before scoring. ``loss`` applies to forest strata only; otherwise its
    weight goes proportionally to the four STATE terms.

    Args:
        df: Candidate pool (percentiles are taken over it).
        weights: DegradedWeights (defaults if None).
        is_forest: Forest integrity stratum (see forest_stratum).

    Returns:
        ScoreResult with the degradation score.
    """
    w = weights or DegradedWeights()
    comps = {
        "perf_deficit": percentile_normalise(-_col(df, "ndvi_state")),
        "moist_deficit": percentile_normalise(-_col(df, "ndmi_state")),
        "unnatural": percentile_normalise(-_col(df, "natural_frac")),
        "pressure": percentile_normalise(_col(df, "ghm")),
        "browning": percentile_normalise(-_col(df, "ndvi_trend")),
        "nbr_decline": percentile_normalise(-_col(df, "nbr_trend")),
        "loss": percentile_normalise(_col(df, "forest_loss_frac")),
    }
    wts = {
        "perf_deficit": w.perf_deficit,
        "moist_deficit": w.moist_deficit,
        "unnatural": w.unnatural,
        "pressure": w.pressure,
        "browning": w.browning,
        "nbr_decline": w.nbr_decline,
        "loss": w.loss,
    }
    if not is_forest:
        state = ["perf_deficit", "moist_deficit", "unnatural", "pressure"]
        state_total = sum(wts[k] for k in state)
        for k in state:
            wts[k] += wts["loss"] * wts[k] / state_total
        wts["loss"] = 0.0
    return _combine(comps, wts)


# --------------------------------------------------------------------------- #
# Degraded anchor (2026-11-07 amendment, spec §0a)                             #
# --------------------------------------------------------------------------- #
def baseline_window(project_start_year: int) -> tuple[int, int] | None:
    """Baseline window for the anchor constraint: the complete years before start.

    ``[start - DEGRADED_BASELINE_WINDOW_YEARS, start - 1]``, clipped to the first
    complete MODIS year. None when no MODIS year precedes the start.

    Args:
        project_start_year: Year restoration started.

    Returns:
        (first_year, last_year) inclusive, or None.
    """
    last = int(project_start_year) - 1
    first = max(last - DEGRADED_BASELINE_WINDOW_YEARS + 1, MODIS_FIRST_FULL_YEAR)
    return (first, last) if last >= first else None


def anchor_constraint(
    df: pd.DataFrame,
    focal: Mapping[str, float],
    indices: Sequence[str] = ANCHOR_INDICES,
) -> pd.DataFrame:
    """D1: is each candidate at least as degraded as the focal site, in both windows?

    For each index ``x``: ``D.x_base <= F.x_base`` and
    ``D.x_state <= min(F.x_base, F.x_state)``. Like-for-like windows keep the focal
    baseline at or above 0% on the TGBS rescale whether the downstream rescale uses
    a fixed or a time-varying degraded value; the ``min`` keeps the anchor at or
    below the focal site throughout. A focal value that is NaN drops that test (the
    caller warns); a candidate value that is NaN fails it (it cannot be verified).

    Args:
        df: Candidates with ``<x>_state`` and ``<x>_base`` columns.
        focal: Focal values keyed ``<x>_state`` / ``<x>_base``.
        indices: Anchor indices (default ANCHOR_INDICES = ndvi, ndmi).

    Returns:
        DataFrame with ``below_baseline_<x>`` per index and ``below_baseline`` (all).
    """
    out = pd.DataFrame(index=df.index)
    for x in indices:
        f_base = focal.get(f"{x}_base", np.nan)
        f_now = focal.get(f"{x}_state", np.nan)
        ok = pd.Series(True, index=df.index)
        if pd.notna(f_base):
            ok &= (_col(df, f"{x}_base") <= f_base).fillna(False)
        ceiling = (
            np.nanmin([f_base, f_now]) if pd.notna([f_base, f_now]).any() else np.nan
        )
        if pd.notna(ceiling):
            ok &= (_col(df, f"{x}_state") <= ceiling).fillna(False)
        out[f"below_baseline_{x}"] = ok.astype(bool)
    out["below_baseline"] = out.all(axis=1) if len(indices) else True
    return out


def pathway_gate(signature_pct: pd.Series, threshold: float) -> pd.Series:
    """D2: 'pass' where pct(signature_sim) >= threshold, else 'failed'.

    Args:
        signature_pct: Within-pool percentile of signature_sim (NaN fails).
        threshold: Minimum percentile (``pathway_gate_pct``).

    Returns:
        Series of 'pass' / 'failed'.
    """
    passed = (pd.to_numeric(signature_pct, errors="coerce") >= threshold).fillna(False)
    return pd.Series(np.where(passed, "pass", "failed"), index=signature_pct.index)


def focal_baseline_pct(
    df: pd.DataFrame,
    focal: Mapping[str, float],
    reference: Mapping[str, float],
    indices: Sequence[str] = ANCHOR_INDICES,
) -> pd.DataFrame:
    """D4: where the focal baseline sits on each candidate's 0-100% scale.

    ``(F.x_base - D.x_state) / (R.x - D.x_state)``. Uses the focal current state if
    the baseline is unavailable. Reported only, never a filter.

    Args:
        df: Candidates with ``<x>_state``.
        focal: Focal values keyed ``<x>_base`` / ``<x>_state``.
        reference: Reference (100%) value per index, keyed ``x``.
        indices: Anchor indices.

    Returns:
        DataFrame with ``focal_baseline_pct_<x>`` (fractions; NaN if undefined).
    """
    out = pd.DataFrame(index=df.index)
    for x in indices:
        f = focal.get(f"{x}_base", np.nan)
        if pd.isna(f):
            f = focal.get(f"{x}_state", np.nan)
        r = reference.get(x, np.nan)
        d = _col(df, f"{x}_state")
        span = r - d
        out[f"focal_baseline_pct_{x}"] = ((f - d) / span).where(span > 0)
    return out


def anchor_check(pcts: pd.DataFrame, threshold: float) -> pd.Series:
    """D4: 'review' where any focal_baseline_pct exceeds ``threshold``, else 'ok'.

    Args:
        pcts: Output of :func:`focal_baseline_pct`.
        threshold: ``anchor_review_pct``.

    Returns:
        Series of 'review' / 'ok'.
    """
    review = (pcts > threshold).any(axis=1) if pcts.shape[1] else False
    return pd.Series(np.where(review, "review", "ok"), index=pcts.index)


def intactness_flag(df: pd.DataFrame) -> pd.Series:
    """'weak' where gHM >= 0.4 (1 - ghm < 0.6, the integrity break) or natural_frac < 0.6.

    Args:
        df: Candidates with ghm and natural_frac.

    Returns:
        Series of 'weak' / 'ok'.
    """
    weak = (1.0 - _col(df, "ghm") < INTACTNESS_WEAK_NATURALNESS) | (
        _col(df, "natural_frac") < INTACTNESS_WEAK_NATURAL_FRAC
    )
    return pd.Series(np.where(weak, "weak", "ok"), index=df.index)


def zero_star_likelihood(
    degradation_pct: pd.Series, ndvi_performance: pd.Series
) -> pd.Series:
    """'high' if pct >= 0.9 and performance <= 0.5; 'moderate' if pct >= 0.75; else 'low'.

    Args:
        degradation_pct: Within-pool percentile of the degradation score.
        ndvi_performance: ndvi_state / single pool denominator.

    Returns:
        Series of 'high' / 'moderate' / 'low'.
    """
    pct = degradation_pct.astype(float)
    perf = ndvi_performance.astype(float)
    out = np.where(
        (pct >= ZERO_STAR_HIGH_PCT) & (perf <= ZERO_STAR_MAX_PERFORMANCE),
        "high",
        np.where(pct >= ZERO_STAR_MODERATE_PCT, "moderate", "low"),
    )
    return pd.Series(out, index=pct.index)


# --------------------------------------------------------------------------- #
# Ranking and thinning                                                         #
# --------------------------------------------------------------------------- #
def forest_stratum(
    class_groups: Iterable[str],
    *,
    class_fallback: bool,
    pnv_code: int | None,
    target_cover_classes: Sequence[int] | None = None,
) -> bool:
    """Whether a run uses the forest integrity stratum.

    The forest stratum requires median canopy height >= ``FOREST_MIN_CANOPY_M`` for
    integrity 2 and adds the forest-only intactness extras. Only true forest groups
    qualify (``VECEA_FOREST_STRATUM_GROUPS``: forest, riverine, bamboo; or a PNV forest
    biome outside VECEA coverage). Woodland and wooded grassland use the non-forest
    path, so intactness does not mean "more trees" (spec §1.2).

    Args:
        class_groups: VECEA_CLASS_GROUPS groups of the focal class set.
        class_fallback: The class gate uses PNV + RESOLVE (focal outside VECEA).
        pnv_code: Focal OpenLandMap PNV biome code (used only with class_fallback).
        target_cover_classes: Optional WorldCover natural classes; tree cover (10) or
            mangrove (95) forces the forest stratum.

    Returns:
        True for the forest stratum.
    """
    forest = bool(set(class_groups) & VECEA_FOREST_STRATUM_GROUPS) or (
        class_fallback and pnv_code in PNV_FOREST_STRATUM_BIOMES
    )
    if target_cover_classes and ({10, 95} & set(target_cover_classes)):
        forest = True
    return forest


def degraded_exclusion_reasons(
    pool: pd.DataFrame,
    *,
    final: bool,
    max_built_frac: float,
    keep_built: bool = False,
    exclude_protected: bool = False,
    max_dw_built: float | None = None,
) -> pd.Series:
    """Why each cell is excluded from a degraded pool (None = kept).

    Permanent water (``water_frac >= DEGRADED_EXCLUDE_MAX_FRAC``) -> 'water'. Built-up
    land, unless ``keep_built`` (degradation_type 'settlement'): on the coarse screen
    ``built_frac >= DEGRADED_EXCLUDE_MAX_FRAC`` (a 2 km cell nearly always holds some
    built-up land); on final cells WorldCover ``built_frac > max_built_frac`` ->
    'built_wc' and Dynamic World ``dw_built_share > max_dw_built`` -> 'built_dw' (NaN
    is kept: a missing value is not evidence of built-up land), because GBS focal sites
    hold no built structures. ``exclude_protected`` -> 'protected'. The first matching
    reason wins. Applied before thinning (gate, then thin).

    Args:
        pool: Cells with water_frac, built_frac and (optionally) pa_share and
            dw_built_share columns.
        final: Final candidate cells (fine cells, or coarse cells when k == 1).
        max_built_frac: Maximum WorldCover built-up fraction for final cells.
        keep_built: Skip the built-up rules (settlement pathway).
        exclude_protected: Also drop cells >= PA_EXCLUDE_MIN_SHARE inside a PA.
        max_dw_built: Maximum Dynamic World built-label share for final cells; None
            disables the Dynamic World test.

    Returns:
        Series of reason strings (None where the cell is kept), aligned to ``pool``.
    """

    def col(name: str) -> pd.Series:
        if name not in pool.columns:
            return pd.Series(0.0, index=pool.index)
        return pd.to_numeric(pool[name], errors="coerce").fillna(0.0)

    reason = pd.Series(None, index=pool.index, dtype=object)

    def mark(mask: pd.Series, label: str) -> None:
        reason[mask & reason.isna()] = label

    mark(col("water_frac") >= DEGRADED_EXCLUDE_MAX_FRAC, "water")
    if not keep_built:
        built = col("built_frac")
        if final:
            mark(built > max_built_frac, "built_wc")
            if max_dw_built is not None and "dw_built_share" in pool.columns:
                dwb = pd.to_numeric(pool["dw_built_share"], errors="coerce")
                mark((dwb > max_dw_built).fillna(False), "built_dw")
        else:
            mark(built >= DEGRADED_EXCLUDE_MAX_FRAC, "built_wc")
    if exclude_protected:
        mark(col("pa_share") >= PA_EXCLUDE_MIN_SHARE, "protected")
    return reason


def degraded_exclusions(
    pool: pd.DataFrame,
    *,
    final: bool,
    max_built_frac: float,
    keep_built: bool = False,
    exclude_protected: bool = False,
    max_dw_built: float | None = None,
) -> pd.Series:
    """Cells to drop from a degraded pool (True = drop); see degraded_exclusion_reasons.

    Args:
        pool: Cells with water_frac, built_frac and optional pa_share / dw_built_share.
        final: Final candidate cells (fine cells, or coarse cells when k == 1).
        max_built_frac: Maximum WorldCover built-up fraction for final cells.
        keep_built: Skip the built-up rules (settlement pathway).
        exclude_protected: Also drop cells >= PA_EXCLUDE_MIN_SHARE inside a PA.
        max_dw_built: Maximum Dynamic World built-label share for final cells.

    Returns:
        Boolean Series aligned to ``pool`` (True = drop).
    """
    return degraded_exclusion_reasons(
        pool,
        final=final,
        max_built_frac=max_built_frac,
        keep_built=keep_built,
        exclude_protected=exclude_protected,
        max_dw_built=max_dw_built,
    ).notna()


def _natural_persist_column(natural_persist_years: str) -> str:
    if natural_persist_years not in ("all", "all_but_one"):
        raise ValueError("natural_persist_years must be 'all' or 'all_but_one'.")
    return f"dw_nat_share_{natural_persist_years}"


def landuse_history(
    df: pd.DataFrame,
    *,
    end_year: int,
    natural_persist_years: str = "all",
    min_natural_persist: float = 0.8,
    max_built_share: float = 0.05,
    baseline_end_year: int | None = None,
    min_modified_share: float = LANDUSE_MIN_MODIFIED_SHARE,
    min_valid_share: float = LANDUSE_MIN_VALID_SHARE,
) -> pd.DataFrame:
    """Absolute land-use history class per cell from the ``dw_landuse`` shares.

    Classes, in priority order:

    - ``no_dw``: fewer than ``min_valid_share`` of pixels observed in n - 1 years;
    - ``persistent_natural``: natural share (all / all-but-one years) >=
      ``min_natural_persist``. It outranks ``built``: a large cell can be mostly
      persistent natural cover and still hold a few percent of homesteads and roads;
      built-up land is filtered separately for degraded runs (``dw_built_share``);
    - ``built``: Dynamic World built-label share > ``max_built_share``;
    - ``converted`` / ``persistent_modified``: converted + persistently modified
      share >= ``min_modified_share``, labelled by the larger part;
    - ``mixed``: everything else.

    The two workflows read the same classes, so the reference gate
    (persistent_natural) and the conversion pathway test (modified by the end of the
    baseline window) are disjoint by construction.

    Args:
        df: Cells with the ``dw_landuse`` columns.
        end_year: Last year of the land-use history.
        natural_persist_years: 'all' or 'all_but_one'.
        min_natural_persist: Persistent-natural share for the reference gate.
        max_built_share: Built-label share above which a cell is 'built'.
        baseline_end_year: Last year of the degraded baseline window (None: no
            ``modified_by_baseline`` column beyond the persistent share).
        min_modified_share: Converted + persistently modified share for the
            converted / persistent_modified classes.
        min_valid_share: Observed share below which a cell is 'no_dw'.

    Returns:
        DataFrame with natural_persist, converted_share, modified_share,
        modified_by_baseline, dw_conversion_year and landuse_class.
    """
    out = pd.DataFrame(index=df.index)
    nat = _col(df, _natural_persist_column(natural_persist_years))
    persist = _col(df, "dw_mod_share_persist")
    conv_end = _col(df, f"dw_converted_by_{int(end_year)}")
    if int(end_year) <= DW_LANDUSE_FIRST_YEAR:
        conv_end = pd.Series(0.0, index=df.index)
    out["natural_persist"] = nat
    out["converted_share"] = conv_end
    out["modified_share"] = (persist.fillna(0) + conv_end.fillna(0)).where(
        persist.notna() | conv_end.notna()
    )
    if baseline_end_year is None:
        by_base = persist
    elif baseline_end_year <= DW_LANDUSE_FIRST_YEAR:
        by_base = persist
    else:
        y = min(int(baseline_end_year), int(end_year))
        conv_base = _col(df, f"dw_converted_by_{y}")
        by_base = (persist.fillna(0) + conv_base.fillna(0)).where(
            persist.notna() | conv_base.notna()
        )
    out["modified_by_baseline"] = by_base
    out["dw_conversion_year"] = _col(df, "dw_conversion_year")
    valid = _col(df, "dw_valid_share")
    built = _col(df, "dw_built_share")
    cls = pd.Series("mixed", index=df.index, dtype=object)
    modified = out["modified_share"] >= min_modified_share
    cls[modified & (conv_end.fillna(0) >= persist.fillna(0))] = "converted"
    cls[modified & (conv_end.fillna(0) < persist.fillna(0))] = "persistent_modified"
    cls[built > max_built_share] = "built"
    cls[nat >= min_natural_persist] = "persistent_natural"
    cls[valid.isna() | (valid < min_valid_share)] = "no_dw"
    out["landuse_class"] = cls
    return out


def conversion_pathway(
    history: pd.DataFrame,
    degradation_type: str,
    *,
    forest_loss_frac: pd.Series | None = None,
    min_modified_share: float = LANDUSE_MIN_MODIFIED_SHARE,
    forest_loss_min: float = 0.10,
) -> pd.Series | None:
    """Absolute degraded pathway test for conversion pathways (replaces D2 there).

    A cell is on a conversion pathway when at least ``min_modified_share`` of it was
    converted (natural -> crops, built or bare) at or before the end of the baseline
    window, or is persistently modified. For 'deforestation' a Hansen forest loss of at
    least ``forest_loss_min`` also passes, because Dynamic World does not separate trees
    from regrowing grass or shrub. Returns None for pathways inside natural cover
    (overgrazing, rangeland), which keep the signature gate (D2) and D1.

    Args:
        history: Output of :func:`landuse_history`.
        degradation_type: Degradation pathway.
        forest_loss_frac: Hansen loss share per cell (deforestation only).
        min_modified_share: Modified share needed.
        forest_loss_min: Loss share that also passes for deforestation.

    Returns:
        Boolean Series (True = on the pathway), or None for non-conversion pathways.
    """
    if degradation_type not in CONVERSION_DEGRADATION_TYPES:
        return None
    ok = (
        pd.to_numeric(history["modified_by_baseline"], errors="coerce")
        >= min_modified_share
    ).fillna(False)
    if degradation_type == "deforestation" and forest_loss_frac is not None:
        loss = pd.to_numeric(forest_loss_frac, errors="coerce").reindex(history.index)
        ok |= (loss >= forest_loss_min).fillna(False)
    return ok


def pathway_example(history: pd.DataFrame) -> pd.Series:
    """True where most of a cell's modification happened after the baseline window.

    Such a cell was still largely natural at baseline, so it shows the pathway but sets
    the baseline-year 0% too high: report it as a pathway example, not an anchor. A
    cell that is mostly persistent cropland with a small recent conversion is not one.

    Args:
        history: Output of :func:`landuse_history` (modified_share,
            modified_by_baseline).

    Returns:
        Boolean Series.
    """
    total = pd.to_numeric(history.get("modified_share"), errors="coerce")
    by_base = pd.to_numeric(history.get("modified_by_baseline"), errors="coerce")
    if total is None or by_base is None:
        return pd.Series(False, index=history.index)
    return ((total - by_base) > by_base).fillna(False)


def near_references(
    pool: pd.DataFrame, references: pd.DataFrame, min_separation_m: float
) -> pd.Series:
    """Cells closer than ``min_separation_m`` to any reference centroid (True = near).

    Used for pathways inside natural cover, where land-use classes cannot keep the
    degraded and reference sets apart.

    Args:
        pool: Cells with cx, cy (grid metres).
        references: Reference cells with cx, cy in the same grid.
        min_separation_m: Separation distance.

    Returns:
        Boolean Series aligned to ``pool``.
    """
    near = pd.Series(False, index=pool.index)
    if references is None or not len(references) or not len(pool):
        return near
    px = pool["cx"].to_numpy(float)
    py = pool["cy"].to_numpy(float)
    for rx, ry in zip(
        references["cx"].to_numpy(float), references["cy"].to_numpy(float), strict=True
    ):
        near |= np.hypot(px - rx, py - ry) < min_separation_m
    return near


def select_top_candidates(
    pool: pd.DataFrame,
    site_type: str,
    min_separation_m: float,
    *,
    n: int = TOP_CANDIDATES_N,
    condition_weight: float = TOP_CANDIDATES_CONDITION_WEIGHT,
    shortlist: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Best ``n`` distinct eligible sites, led by condition and environmental similarity.

    ``top_score = condition_weight * pct(condition) + (1 - condition_weight) * env_sim``,
    where the condition is ``intactness`` (reference) or ``degradation`` (degraded) and
    both percentiles are over the eligible cells (``env_sim`` already is one). Ties are
    broken by ``rank_score``, the score behind the shortlist's thinned rank. The cells
    are then spaced with the shortlist rule (centroid distance >= ``min_separation_m``),
    so every pick is a separate site. Unlike ``rank_score``, where similarity can at
    most halve a score, condition and similarity carry the given weights. Pure pandas.

    Args:
        pool: Final cells indexed by cell_id with ``pool_status``, the condition score,
            ``env_sim``, ``rank_score`` and grid centroids ``cx`` / ``cy``.
        site_type: 'reference' or 'degraded'.
        min_separation_m: Spacing distance (the shortlist's ``min_separation_m``).
        n: Number of sites.
        condition_weight: Weight of the condition percentile (0-1).
        shortlist: Ranked (proposed) sites indexed by cell_id with ``rank`` (and
            ``role``); used to flag proposed sites.

    Returns:
        Up to ``n`` cells in order with ``top_rank``, ``top_score``,
        ``<condition>_pct``, ``proposed``, ``proposed_rank`` and ``proposed_role``
        first, then the absolute condition flags (``TOP_CANDIDATE_FLAGS``) and
        ``mandatory_met`` when the pool has them, then the other pool columns. Flags
        are shown, never used to filter or rank.
    """
    if not 0.0 <= condition_weight <= 1.0:
        raise ValueError("condition_weight must be between 0 and 1.")
    cond = "intactness" if site_type == "reference" else "degradation"
    lead = ["top_rank", "top_score", f"{cond}_pct", "proposed", "proposed_rank"]
    lead.append("proposed_role")
    lead += [
        c
        for c in (*TOP_CANDIDATE_FLAGS.get(site_type, ()), "mandatory_met")
        if c in pool
    ]
    if "pool_status" not in pool or cond not in pool:
        return pd.DataFrame(columns=lead + list(pool.columns))
    elig = pool.loc[(pool["pool_status"] == "eligible") & pool[cond].notna()].copy()
    if not len(elig):
        return pd.DataFrame(columns=lead + list(pool.columns))
    elig[f"{cond}_pct"] = elig[cond].rank(pct=True)
    env = pd.to_numeric(elig.get("env_sim"), errors="coerce").fillna(0.0)
    elig["top_score"] = (
        condition_weight * elig[f"{cond}_pct"] + (1.0 - condition_weight) * env
    )
    tie = pd.to_numeric(elig.get("rank_score"), errors="coerce").rank(pct=True)
    key = elig["top_score"] + 1e-9 * tie.fillna(0.0)
    kept = thin_spatially(
        elig.assign(top_sort_key=key), "top_sort_key", min_separation_m
    )
    top = elig.loc[kept[:n]].copy()
    top["top_rank"] = np.arange(1, len(top) + 1)
    if shortlist is not None and len(shortlist):
        top["proposed"] = top.index.isin(shortlist.index)
        top["proposed_rank"] = shortlist["rank"].reindex(top.index)
        role = shortlist["role"] if "role" in shortlist else None
        top["proposed_role"] = role.reindex(top.index) if role is not None else None
    else:
        top["proposed"] = False
        top["proposed_rank"] = np.nan
        top["proposed_role"] = None
    return top[lead + [c for c in top.columns if c not in lead]]


def gate_then_thin(
    pool: pd.DataFrame, eligible: pd.Series, score_col: str, min_separation_m: float
) -> pd.DataFrame:
    """Drop ineligible cells first, then thin spatially (never the reverse).

    Thinning before a filter lets a rejected cell suppress its eligible neighbours,
    which are then never reconsidered (the v0.5.0 Dynamic World built filter did
    this). Every gate in the workflows runs through this order.

    Args:
        pool: Scored candidates.
        eligible: Boolean Series aligned to ``pool`` (True = keep).
        score_col: Score column.
        min_separation_m: Thinning distance.

    Returns:
        Thinned eligible candidates in rank order with a 'rank' column.
    """
    keep = eligible.reindex(pool.index, fill_value=False).astype(bool)
    sub = pool.loc[keep]
    if not len(sub):
        return sub.assign(rank=pd.Series(dtype=int))
    return rank_candidates(sub, score_col, min_separation_m)


def stop_rule_met(
    n_thinned: int,
    n_candidates: int,
    units: Iterable[str] = (),
    covered: Iterable[str] = (),
    *,
    require_units: bool = True,
) -> bool:
    """Search-ladder stop rule: enough eligible thinned candidates (and every unit).

    Args:
        n_thinned: Eligible thinned candidates on the rung.
        n_candidates: Candidates wanted.
        units: Management units (focal classes) that need a candidate.
        covered: Units with at least one eligible thinned candidate.
        require_units: Also require every unit to be covered.

    Returns:
        True when the rung is good enough to stop the ladder.
    """
    if n_thinned < n_candidates:
        return False
    if require_units and set(units) - set(covered):
        return False
    return True


def thin_spatially(
    df: pd.DataFrame,
    score_col: str,
    min_separation_m: float,
    x: str = "cx",
    y: str = "cy",
) -> pd.Index:
    """Greedy non-maximum suppression: keep the best cell, drop near neighbours.

    Cells are visited by descending score (ties broken by cell id, so the result is
    order-independent); a cell is kept unless a kept cell lies closer than
    ``min_separation_m`` (centroid distance, grid metres).

    Args:
        df: Candidates with score and centroid columns.
        score_col: Column to rank by (NaN scores are dropped).
        min_separation_m: Minimum centroid separation between kept cells.
        x: Easting column.
        y: Northing column.

    Returns:
        Index of kept cells in rank order.
    """
    ordered = sort_by_score(df.dropna(subset=[score_col]), score_col)
    xs = ordered[x].to_numpy(float)
    ys = ordered[y].to_numpy(float)
    kept: list[int] = []
    kx: list[float] = []
    ky: list[float] = []
    tol = 1e-6
    for i in range(len(ordered)):
        if kept:
            d = np.hypot(np.asarray(kx) - xs[i], np.asarray(ky) - ys[i])
            if np.any(d < min_separation_m - tol):
                continue
        kept.append(i)
        kx.append(xs[i])
        ky.append(ys[i])
    return ordered.index[kept]


# Temporary tie-break column; a reserved name so it never overwrites a score column
# (it was "_key", which silently replaced a score column of that name).
_TIEBREAK_COL = "__sort_by_score_index__"


def sort_by_score(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Sort descending by score with the index as a deterministic tie-breaker.

    Args:
        df: Candidates.
        score_col: Score column.

    Returns:
        Sorted copy.
    """
    if score_col == _TIEBREAK_COL:
        raise ValueError(f"score_col may not be named {_TIEBREAK_COL!r}")
    tmp = df.assign(**{_TIEBREAK_COL: df.index.astype(str)})
    return tmp.sort_values([score_col, _TIEBREAK_COL], ascending=[False, True]).drop(
        columns=_TIEBREAK_COL
    )


def rank_candidates(
    df: pd.DataFrame, score_col: str, min_separation_m: float
) -> pd.DataFrame:
    """Sort by score, thin spatially, and number the survivors 1..n.

    Args:
        df: Scored candidates.
        score_col: Score column.
        min_separation_m: Thinning distance.

    Returns:
        The thinned candidates in rank order with a 'rank' column.
    """
    kept = thin_spatially(df, score_col, min_separation_m)
    out = df.loc[kept].copy()
    out["rank"] = np.arange(1, len(out) + 1)
    return out


# --------------------------------------------------------------------------- #
# Environmental-range spanning                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class SpanResult:
    """Output of select_spanning_set.

    Attributes:
        selected: Selected candidates in order, with 'role', 'covers_unit', 'span_note'.
        span_mode: 'spanning', 'rank_fallback' or 'rank' (span_environment off).
        env_coverage: Share of focal samples within 1 pooled SD of a selected site.
        warnings: Messages for the result.
        units: Management units and whether each was covered.
    """

    selected: pd.DataFrame
    span_mode: str
    env_coverage: float
    warnings: list[str]
    units: dict[str, bool]


def _zspace(
    candidates: pd.DataFrame,
    focal: pd.DataFrame,
    pool: pd.DataFrame,
    predictors: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    preds = [
        p
        for p in predictors
        if p in candidates.columns and p in focal.columns and p in pool.columns
    ]
    sd = pool[preds].astype(float).std(ddof=0).replace(0, np.nan)
    mu = pool[preds].astype(float).mean()
    preds = [p for p in preds if pd.notna(sd[p])]
    cz = ((candidates[preds].astype(float) - mu[preds]) / sd[preds]).to_numpy()
    fz = ((focal[preds].astype(float) - mu[preds]) / sd[preds]).to_numpy()
    cz = np.nan_to_num(cz, nan=0.0)
    fz = fz[~np.isnan(fz).any(axis=1)] if len(fz) else fz
    return cz, fz, preds


def _mean_nearest(fz: np.ndarray, selected: np.ndarray) -> float:
    if len(fz) == 0 or len(selected) == 0:
        return np.inf
    d = np.linalg.norm(fz[:, None, :] - selected[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def _span_note(cz_row: np.ndarray, focal_z: np.ndarray, preds: list[str]) -> str:
    if len(focal_z) == 0 or not preds:
        return "focal environment not sampled"
    diff = cz_row - np.median(focal_z, axis=0)
    order = np.argsort(-np.abs(diff))
    parts = [
        f"{'higher' if diff[i] > 0 else 'lower'} {PREDICTOR_LABELS.get(preds[i], preds[i])}"
        for i in order[:2]
        if abs(diff[i]) >= 0.5
    ]
    return ", ".join(parts) if parts else "close to the focal environment"


def select_spanning_set(
    candidates: pd.DataFrame,
    focal_samples: pd.DataFrame,
    pool: pd.DataFrame,
    predictors: Sequence[str],
    n: int,
    *,
    eligible: pd.Series,
    no_mandatory_zero: pd.Series,
    units: Mapping[str, set[str]] | None = None,
    span_weight: float = 0.5,
    span_score_floor: float = 0.5,
    span_environment: bool = True,
    score_col: str = "rank_score",
) -> SpanResult:
    """Pick >= 3 references that together span the focal site's environment (SOP).

    Eligible = top ``span_score_floor`` of the thinned pool by score AND meeting the
    mandatory rule (``eligible``). The primary is the best eligible candidate; each
    secondary maximises ``span_weight * pct(coverage_gain) + (1 - span_weight) *
    pct(score)``, where coverage_gain is the drop in mean z-distance from focal samples
    to their nearest selected site. Each management unit (focal class) must be covered
    by at least one selected site whose class matches it. With fewer than 3 eligible
    candidates the selection falls back to rank order over candidates with no
    mandatory 0 (``span_mode='rank_fallback'``).

    Args:
        candidates: Thinned candidates in rank order (index cell_id), with predictor
            columns, ``score_col`` and optional 'unit_keys' (set of units covered).
        focal_samples: Joint focal samples (one row per pixel, predictor columns).
        pool: Pool used for z-scoring (pool SD).
        predictors: Predictor columns used for coverage.
        n: Number of references wanted (raised to the number of units).
        eligible: Boolean series: candidate meets the mandatory eligibility rule.
        no_mandatory_zero: Boolean series: candidate has no mandatory 0.
        units: Unit name -> set of acceptable class keys; None/empty = no units.
        span_weight: Coverage vs score trade-off.
        span_score_floor: Score percentile (within the thinned pool) required.
        span_environment: False -> pure rank order.
        score_col: Score column.

    Returns:
        SpanResult.
    """
    warnings: list[str] = []
    units = dict(units or {})
    if len(units) > n:
        warnings.append(
            f"n_candidates raised from {n} to {len(units)} so each of the "
            f"{len(units)} management units gets a reference (SOP step 2)."
        )
        n = len(units)
    cands = candidates.copy()
    unit_keys = cands.get("unit_keys")
    if unit_keys is None:
        unit_keys = pd.Series([set() for _ in range(len(cands))], index=cands.index)

    score_pct = percentile_normalise(cands[score_col])
    elig = cands.index[
        (score_pct >= span_score_floor)
        & eligible.reindex(cands.index, fill_value=False)
    ]
    mode = "spanning" if span_environment else "rank"
    if span_environment and len(elig) < 3:
        mode = "rank_fallback"
        warnings.append(
            f"Only {len(elig)} candidates meet the mandatory SOP criteria in the top "
            "half of the pool; references fall back to rank order. SOP fallbacks to "
            "consider (not applied): recovering sites with known histories, best "
            "portions of the project site, pooled regional data."
        )
    if mode != "spanning":
        base = cands.index[no_mandatory_zero.reindex(cands.index, fill_value=False)]
        if len(base) == 0:
            base = cands.index
        pick_from = list(base)
    else:
        pick_from = list(elig)

    cz_all, fz, preds = _zspace(cands, focal_samples, pool, predictors)
    pos = {cid: i for i, cid in enumerate(cands.index)}
    chosen: list[Any] = []

    def covered_units() -> set[str]:
        got: set[str] = set()
        for cid in chosen:
            got |= set(unit_keys.loc[cid]) & set(units)
        return got

    while len(chosen) < n:
        remaining = [c for c in pick_from if c not in chosen]
        if not remaining:
            break
        missing_units = set(units) - covered_units()
        slots = n - len(chosen)
        if missing_units and slots <= len(missing_units):
            fit = [c for c in remaining if set(unit_keys.loc[c]) & missing_units]
            if fit:
                remaining = fit
        if not chosen or mode != "spanning":
            pick = remaining[0]  # candidates are in rank order
        else:
            cur = _mean_nearest(fz, cz_all[[pos[c] for c in chosen]])
            gains = []
            for c in remaining:
                sel = cz_all[[pos[x] for x in chosen + [c]]]
                gains.append(cur - _mean_nearest(fz, sel))
            gain = pd.Series(gains, index=remaining)
            rs = cands.loc[remaining, score_col]
            obj = span_weight * percentile_normalise(gain) + (
                1 - span_weight
            ) * percentile_normalise(rs)
            pick = sort_by_score(obj.to_frame("o"), "o").index[0]
        chosen.append(pick)

    sel = cands.loc[chosen].copy()
    roles = ["primary"] + ["secondary"] * (len(sel) - 1)
    sel["role"] = roles[: len(sel)] if mode == "spanning" else ["rank"] * len(sel)
    sel["covers_unit"] = [
        ",".join(sorted(set(unit_keys.loc[c]) & set(units))) or None for c in chosen
    ]
    sel["span_note"] = [_span_note(cz_all[pos[c]], fz, preds) for c in chosen]
    if len(fz) and len(chosen):
        d = np.linalg.norm(
            fz[:, None, :] - cz_all[[pos[c] for c in chosen]][None, :, :], axis=2
        ) / np.sqrt(max(len(preds), 1))
        coverage = float((d.min(axis=1) <= SPAN_ENV_COVERAGE_SD).mean())
    else:
        coverage = float("nan")
    unit_status = {u: u in covered_units() for u in units}
    for u, ok in unit_status.items():
        if not ok:
            warnings.append(f"Management unit '{u}' has no matching reference.")
    return SpanResult(sel, mode, coverage, warnings, unit_status)


# --------------------------------------------------------------------------- #
# SOP selection matrix                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class MatrixContext:
    """Focal values and settings the SOP selection matrix scores against.

    Attributes:
        site_type: 'reference' or 'degraded'.
        elevation: Focal median elevation (m).
        elevation_p5: Focal 5th percentile elevation.
        elevation_p95: Focal 95th percentile elevation.
        rainfall: Focal mean annual rainfall (mm/yr).
        rainfall_p5: Focal 5th percentile rainfall.
        rainfall_p95: Focal 95th percentile rainfall.
        elevation_tol_m: SOP elevation tolerance.
        rainfall_tol_frac: SOP rainfall tolerance.
        focal_area_ha: Focal site area (ha).
        is_forest: Forest integrity stratum (see forest_stratum).
        class_fallback: PNV + ecoregion used instead of VECEA.
        class_mosaic: Focal class is a mosaic class.
        substrate_fallback: iSDA texture used instead of WRB.
        substrate_off: Substrate gate disabled by the user.
        relaxed_criteria: Mandatory criteria relaxed for this rung.
        focal_class_label: Focal class description for evidence.
        focal_substrate_label: Focal soil group description for evidence.
        baseline_end_year: Last year of the degraded baseline window (evidence).
    """

    site_type: str
    elevation: float
    elevation_p5: float
    elevation_p95: float
    rainfall: float
    rainfall_p5: float
    rainfall_p95: float
    elevation_tol_m: float
    rainfall_tol_frac: float
    focal_area_ha: float
    is_forest: bool
    class_fallback: bool
    class_mosaic: bool
    substrate_fallback: bool
    substrate_off: bool
    relaxed_criteria: set[str]
    focal_class_label: str
    focal_substrate_label: str
    baseline_end_year: int | None = None


REFERENCE_MANDATORY = (
    "national_class",
    "substrate",
    "elevation",
    "rainfall",
    "integrity",
)
DEGRADED_MANDATORY = (
    "national_class",
    "substrate",
    "elevation",
    "rainfall",
    "below_baseline",
)
HUMAN_CRITERIA = ("data_availability", "field_access", "successional_phase")


def _band_score(value: float, lo2: float, hi2: float, lo1: float, hi1: float) -> int:
    if pd.isna(value):
        return 0
    if lo2 <= value <= hi2:
        return 2
    if lo1 <= value <= hi1:
        return 1
    return 0


def _fmt(v: Any, nd: int = 3) -> str:
    try:
        return "NA" if pd.isna(v) else f"{float(v):.{nd}g}"
    except (TypeError, ValueError):
        return str(v)


def mandatory_criteria(site_type: str, substrate_off: bool = False) -> list[str]:
    """Mandatory SOP criteria for a site type.

    Args:
        site_type: 'reference' or 'degraded'.
        substrate_off: Drop substrate when the user switched its gate off.

    Returns:
        List of criterion names.
    """
    base = REFERENCE_MANDATORY if site_type == "reference" else DEGRADED_MANDATORY
    return [c for c in base if not (substrate_off and c == "substrate")]


def build_selection_matrix(
    candidates: pd.DataFrame, ctx: MatrixContext
) -> pd.DataFrame:
    """CER-K SOP 0/1/2 selection matrix with evidence and H/M/L confidence.

    One row per candidate; for each criterion ``<c>_score``, ``<c>_evidence`` and
    ``<c>_confidence``. Human criteria (data availability, field access, successional
    phase) are left blank for the team. Adds mandatory_met (all mandatory = 2),
    mandatory_met_relaxed (relaxed mandatory >= 1, others = 2), fallback and
    matrix_total (automated criteria only).

    Args:
        candidates: Candidates with the gate, condition and label columns.
        ctx: MatrixContext.

    Returns:
        DataFrame indexed like ``candidates``.
    """
    t = SOP_MATRIX_THRESHOLDS
    rows: dict[str, dict[str, Any]] = {}
    e_lo2 = min(ctx.elevation_p5, ctx.elevation - t["elevation_band_m"])
    e_hi2 = max(ctx.elevation_p95, ctx.elevation + t["elevation_band_m"])
    r_lo2 = min(ctx.rainfall_p5, ctx.rainfall * (1 - t["rainfall_band_frac"]))
    r_hi2 = max(ctx.rainfall_p95, ctx.rainfall * (1 + t["rainfall_band_frac"]))

    for cid, c in candidates.iterrows():
        r: dict[str, Any] = {}
        # National class
        if ctx.class_mosaic:
            r["national_class"] = (
                2,
                f"focal class {ctx.focal_class_label} is a mosaic class; candidate "
                f"{c.get('national_class')}",
                "L",
            )
        elif ctx.class_fallback:
            share = c.get("pnv_eco_same_mean", np.nan)
            r["national_class"] = (
                1 if share >= 0.5 else 0,
                f"outside VECEA: PNV biome + RESOLVE ECO_ID match share {_fmt(share)}",
                "M",
            )
        else:
            same = c.get("in_class_set_mean", np.nan)
            grp = c.get("in_group_set_mean", np.nan)
            score = 2 if same >= 0.5 else (1 if grp >= 0.5 else 0)
            r["national_class"] = (
                score,
                f"VECEA {c.get('national_class')} (focal {ctx.focal_class_label}); "
                f"same-class share {_fmt(same)}, group share {_fmt(grp)}",
                "M",
            )
        # Substrate
        if ctx.substrate_fallback:
            share = c.get("texture_same_mean", np.nan)
            r["substrate"] = (
                2 if share >= 0.5 else 0,
                f"WRB unavailable; iSDA texture-class match share {_fmt(share)}",
                "L",
            )
        else:
            same = c.get("wrb_same_mean", np.nan)
            rel = c.get("wrb_related_mean", np.nan)
            score = 2 if same >= 0.5 else (1 if rel >= 0.5 else 0)
            r["substrate"] = (
                score,
                f"WRB {c.get('wrb_group')} (focal {ctx.focal_substrate_label}); "
                f"same {_fmt(same)}, related {_fmt(rel)}; parent material unverified",
                "M",
            )
        # Elevation / rainfall
        elev = c.get("elevation_p50", np.nan)
        r["elevation"] = (
            _band_score(
                elev,
                e_lo2,
                e_hi2,
                ctx.elevation - ctx.elevation_tol_m,
                ctx.elevation + ctx.elevation_tol_m,
            ),
            f"median {_fmt(elev, 4)} m vs focal {_fmt(ctx.elevation, 4)} m "
            f"(2: {_fmt(e_lo2, 4)}-{_fmt(e_hi2, 4)}; 1: +/-{_fmt(ctx.elevation_tol_m)})",
            "H",
        )
        rain = c.get("rain_mean_annual", np.nan)
        r["rainfall"] = (
            _band_score(
                rain,
                r_lo2,
                r_hi2,
                ctx.rainfall * (1 - ctx.rainfall_tol_frac),
                ctx.rainfall * (1 + ctx.rainfall_tol_frac),
            ),
            f"CHIRPS {_fmt(rain, 4)} mm/yr vs focal {_fmt(ctx.rainfall, 4)} "
            f"(1: +/-{int(ctx.rainfall_tol_frac * 100)}%)",
            "H",
        )
        loss = c.get("loss_since_2000_frac", np.nan)
        if ctx.site_type == "reference":
            r["integrity"] = _integrity(c, ctx)
            eff = c.get("effective_area_ha", np.nan)
            target = min(ctx.focal_area_ha, eff) if pd.notna(eff) else ctx.focal_area_ha
            nat_area = c.get("natural_frac", np.nan) * eff
            size = (
                2
                if nat_area >= target
                else (1 if nat_area >= t["size_partial"] * target else 0)
            )
            r["size_core"] = (
                size,
                f"natural area {_fmt(nat_area)} ha vs target {_fmt(target)} ha "
                f"(250 m fraction image)",
                "M",
            )
            p14 = c.get("pa_i_iv_share", np.nan)
            pany = c.get("pa_share", np.nan)
            prot = 2 if p14 >= t["protection_share"] else (1 if pany > 0 else 0)
            r["protection"] = (
                prot,
                f"designated WDPA IUCN I-IV share {_fmt(p14)}, any designated "
                f"{_fmt(pany)}; tenure not assessable",
                "M",
            )
        else:
            dl = (
                2
                if loss >= t["degraded_loss_2"]
                else (1 if loss >= t["degraded_loss_1"] else 0)
            )
            r["documented_loss"] = (
                dl,
                f"Hansen loss since 2000 {_fmt(loss)} of cell"
                + ("" if ctx.is_forest else " (non-forest pathway: low loss expected)"),
                "H",
            )
            zs = c.get("zero_star_likelihood")
            r["zero_star"] = (
                {"high": 2, "moderate": 1}.get(str(zs), 0),
                f"zero_star_likelihood {zs} (degradation pct "
                f"{_fmt(c.get('degradation_pct'))}, ndvi_performance "
                f"{_fmt(c.get('ndvi_performance'))})",
                "M",
            )
            bb = bool(c.get("below_baseline", False))
            failing = [
                x
                for x in ANCHOR_INDICES
                if not bool(c.get(f"below_baseline_{x}", False))
            ]
            r["below_baseline"] = (
                2 if bb else 0,
                (
                    "at least as degraded as the focal site (baseline and current "
                    "windows)"
                    if bb
                    else f"not below the focal site on {','.join(failing) or 'n/a'}"
                )
                + "; focal_baseline_pct "
                + ", ".join(
                    f"{x} {_fmt(c.get(f'focal_baseline_pct_{x}'))}"
                    for x in ANCHOR_INDICES
                )
                + f"; anchor_check {c.get('anchor_check')}",
                "M",
            )
            sp = c.get("signature_pct", np.nan)
            cy = c.get("dw_conversion_year", np.nan)
            land = (
                f"; land use {c.get('landuse_class')}, modified by baseline "
                f"{_fmt(c.get('modified_by_baseline'))}, converted "
                f"{_fmt(c.get('converted_share'))}"
                + (f" (mean year {int(round(cy))})" if pd.notna(cy) else "")
            )
            if bool(c.get("pathway_example", False)):
                land += (
                    f"; converted after the baseline window ({ctx.baseline_end_year}): "
                    "pathway example rather than anchor"
                )
            r["pathway"] = (
                2 if sp >= t["pathway_2"] else (1 if sp >= t["pathway_1"] else 0),
                f"signature percentile {_fmt(sp)} (gate: {c.get('pathway_gate')}; "
                "informational)" + land,
                "M",
            )
        row: dict[str, Any] = {}
        for crit, (score, ev, conf) in r.items():
            row[f"{crit}_score"] = int(score)
            row[f"{crit}_evidence"] = ev
            row[f"{crit}_confidence"] = conf
        for crit in HUMAN_CRITERIA:
            row[f"{crit}_score"] = np.nan
            row[f"{crit}_evidence"] = "for the team"
            row[f"{crit}_confidence"] = None
        rows[cid] = row
    mat = pd.DataFrame.from_dict(rows, orient="index")
    mat.index.name = candidates.index.name or "cell_id"
    if mat.empty:
        # No candidates (e.g. every cell gated out): keep the summary columns so
        # callers can join and export an empty matrix.
        empty = pd.DataFrame(
            {
                "mandatory_met": pd.Series(dtype=bool),
                "mandatory_met_relaxed": pd.Series(dtype=bool),
                "no_mandatory_zero": pd.Series(dtype=bool),
                "fallback": pd.Series(dtype=bool),
                "matrix_total": pd.Series(dtype=float),
            }
        )
        empty.index.name = mat.index.name
        return empty
    mand = mandatory_criteria(ctx.site_type, ctx.substrate_off)
    scores = mat[[f"{m}_score" for m in mand]]
    mat["mandatory_met"] = (scores == 2).all(axis=1)
    relaxed_ok = pd.Series(True, index=mat.index)
    fallback = pd.Series(False, index=mat.index)
    for m in mand:
        s = mat[f"{m}_score"]
        if m in ctx.relaxed_criteria:
            relaxed_ok &= s >= 1
            fallback |= s == 1
        else:
            relaxed_ok &= s == 2
    mat["mandatory_met_relaxed"] = relaxed_ok
    mat["no_mandatory_zero"] = (scores > 0).all(axis=1)
    mat["fallback"] = fallback
    auto = [
        c
        for c in mat.columns
        if c.endswith("_score") and not c.startswith(HUMAN_CRITERIA)
    ]
    mat["matrix_total"] = mat[auto].sum(axis=1)
    return mat


def sop_pool_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    """SOP status columns for the exported pool: ``mandatory_met`` + criterion scores.

    Keeps ``mandatory_met`` and every automated ``<criterion>_score`` (0/1/2) of a
    selection matrix; the human criteria (left blank for the team), the evidence and
    the confidence columns stay in the selection-matrix CSV.

    Args:
        matrix: Output of ``build_selection_matrix``.

    Returns:
        DataFrame indexed like ``matrix`` with ``mandatory_met`` first.
    """
    scores = [
        c
        for c in matrix.columns
        if c.endswith("_score") and not c.startswith(HUMAN_CRITERIA)
    ]
    cols = (["mandatory_met"] if "mandatory_met" in matrix else []) + scores
    return matrix[cols].copy()


# Absolute condition flags carried to the top-candidates file (flag, never filter).
TOP_CANDIDATE_FLAGS = {
    "reference": ("intactness_flag",),
    "degraded": ("zero_star_likelihood", "anchor_check"),
}


def _integrity(c: pd.Series, ctx: MatrixContext) -> tuple:
    """SOP integrity row with absolute thresholds (never pool-relative).

    0: loss since 2000 > integrity_loss_1 or recent loss > recent_loss_frac.
    2, forest: loss < integrity_loss_2 and canopy >= FOREST_MIN_CANOPY_M.
    2, non-forest: loss < integrity_loss_2, the reference land-use gate passed
    (persistent natural) and gHM < integrity_max_ghm. Otherwise 1. An unchanged
    candidate scores the same whatever the rest of the pool looks like.
    """
    t = SOP_MATRIX_THRESHOLDS
    loss = c.get("loss_since_2000_frac", np.nan)
    recent = c.get("recent_loss_frac", np.nan)
    loss = 0.0 if pd.isna(loss) else loss
    recent = 0.0 if pd.isna(recent) else recent
    ghm = c.get("ghm", np.nan)
    gate = c.get("landuse_class") == "persistent_natural"
    if loss > t["integrity_loss_1"] or recent > t["recent_loss_frac"]:
        score = 0
    else:
        cond2 = loss < t["integrity_loss_2"]
        if ctx.is_forest:
            ch = c.get("canopy_height_p50", np.nan)
            cond2 = cond2 and (pd.isna(ch) or ch >= FOREST_MIN_CANOPY_M)
        else:
            cond2 = cond2 and gate and pd.notna(ghm) and ghm < t["integrity_max_ghm"]
        score = 2 if cond2 else 1
    stratum = (
        f"forest (canopy >= {FOREST_MIN_CANOPY_M:g} m)"
        if ctx.is_forest
        else f"non-forest (needs land-use gate and gHM < {t['integrity_max_ghm']})"
    )
    ev = (
        f"loss since 2000 {_fmt(loss)}, recent loss {_fmt(recent)}, gHM {_fmt(ghm)}, "
        f"land use {c.get('landuse_class')}; stratum {stratum}"
    )
    return score, ev, "H"


# --------------------------------------------------------------------------- #
# Layer register                                                               #
# --------------------------------------------------------------------------- #
def build_layer_register(
    provenance: Mapping[str, str],
    params: Mapping[str, Any],
    access_date: str,
    extra_rows: Iterable[Mapping[str, Any]] = (),
) -> pd.DataFrame:
    """CER-K SOP §7 layer and query register.

    Columns: layer_or_query, version_or_date, extent_or_parameters, access_date,
    result, confidence. Datasets come from ``provenance`` ([H] unless ``extra_rows``
    say otherwise); gate thresholds and MESS floors from ``params`` at [L].

    Args:
        provenance: Layer name -> 'id | version | licence' description.
        params: Run parameters (thresholds, tolerances, rung reached, ...).
        access_date: ISO date of the run.
        extra_rows: Additional rows (fallbacks at [M], reused tables, ...).

    Returns:
        DataFrame, one row per layer, query or threshold.
    """
    rows: list[dict[str, Any]] = []
    for layer, desc in provenance.items():
        rows.append(
            {
                "layer_or_query": layer,
                "version_or_date": desc,
                "extent_or_parameters": params.get("search_extent", ""),
                "access_date": access_date,
                "result": "used",
                "confidence": "H",
            }
        )
    tol = params.get("tolerances", {})
    for key in (
        "elevation_tol_m",
        "rainfall_tol_frac",
        "focal_class_min_share",
        "mess_floor",
    ):
        if key in tol:
            rows.append(
                {
                    "layer_or_query": f"threshold: {key}",
                    "version_or_date": (
                        "SOP default" if "tol" in key else "spec default"
                    ),
                    "extent_or_parameters": tol[key],
                    "access_date": access_date,
                    "result": "default threshold, to be field-checked",
                    "confidence": "L",
                }
            )
    for key, value in params.get("mess_range_floors", {}).items():
        rows.append(
            {
                "layer_or_query": f"MESS range floor: {key}",
                "version_or_date": "spec §5.2",
                "extent_or_parameters": value,
                "access_date": access_date,
                "result": (
                    "widened"
                    if key in params.get("mess_range_widened", [])
                    else "not needed"
                ),
                "confidence": "L",
            }
        )
    for key, value in SOP_MATRIX_THRESHOLDS.items():
        rows.append(
            {
                "layer_or_query": f"matrix threshold: {key}",
                "version_or_date": "spec §5.5",
                "extent_or_parameters": value,
                "access_date": access_date,
                "result": "applied",
                "confidence": "L",
            }
        )
    rows.extend(dict(r) for r in extra_rows)
    return pd.DataFrame(rows)
