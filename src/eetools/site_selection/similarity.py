"""Environmental similarity: MESS, collinearity pruning, envelope gate, embeddings.

Everything here is pure Python (numpy/pandas) and costs no EECU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from eetools.constants import (
    FINE_MIN_SHARE,
    MESS_COLLINEARITY_R,
    MESS_RANGE_FLOORS,
    MIN_LAND_SHARE,
    SATELLITE_EMBEDDING_BANDS,
)


# --------------------------------------------------------------------------- #
# MESS (Elith et al. 2010; modEvA::MESS)                                       #
# --------------------------------------------------------------------------- #
def _floor_width(predictor: str, ref_median: float, floors: Mapping) -> float | None:
    spec = floors.get(predictor)
    if spec is None:
        return None
    kind, value = spec
    if kind == "abs":
        return float(value)
    return float(value) * abs(ref_median)


def mess_similarity(
    reference: np.ndarray, p: np.ndarray, lo: float, hi: float
) -> np.ndarray:
    """Per-predictor MESS similarity of candidate values against a reference sample.

    ``f`` = % of reference values < p. Similarity is ``(p - lo)/(hi - lo) * 100`` if
    f = 0; ``2f`` if 0 < f <= 50; ``2(100 - f)`` if 50 < f < 100;
    ``(hi - p)/(hi - lo) * 100`` if f = 100.

    Args:
        reference: 1-D array of reference values (NaNs ignored).
        p: 1-D array of candidate values.
        lo: Lower end of the reference range (observed min, possibly widened).
        hi: Upper end of the reference range (observed max, possibly widened).

    Returns:
        Array of similarities (NaN where p is NaN).
    """
    ref = np.sort(np.asarray(reference, float)[~np.isnan(reference)])
    p = np.asarray(p, float)
    n = len(ref)
    f = np.searchsorted(ref, p, side="left") / n * 100.0
    width = hi - lo
    out = np.where(
        f == 0,
        (p - lo) / width * 100.0,
        np.where(
            f <= 50,
            2 * f,
            np.where(f < 100, 2 * (100 - f), (hi - p) / width * 100.0),
        ),
    )
    out[np.isnan(p)] = np.nan
    return out


@dataclass(frozen=True)
class MessResult:
    """MESS output.

    Attributes:
        scores: DataFrame (candidate index) with 'mess', 'mess_mod' and one
            'mess_<predictor>' column per used predictor.
        widened: Predictors whose reference range was widened to its floor.
        dropped: Predictors dropped because their range was still zero (or they had
            no reference values).
    """

    scores: pd.DataFrame
    widened: list[str]
    dropped: list[str]


def compute_mess(
    reference: Mapping[str, Sequence[float] | np.ndarray],
    candidates: pd.DataFrame,
    predictors: Sequence[str],
    range_floors: Mapping[str, tuple[str, float]] | None = MESS_RANGE_FLOORS,
) -> MessResult:
    """Multivariate environmental similarity surface (MESS) for candidate cells.

    A pure-numpy port of ``modEvA::MESS(V, P)``. The reference range of each
    predictor is widened symmetrically about the reference median to at least its
    floor (``range_floors``), so a small focal site with near-constant climate does
    not divide by zero. Predictors whose range is still zero are dropped (never NaN).
    With ``range_floors=None`` the output matches modEvA exactly.

    Args:
        reference: Predictor name -> reference sample values (``V``); each predictor
            may have its own sample (native-scale sampling).
        candidates: DataFrame with one column per predictor (``P``, per-cell medians).
        predictors: Predictors to use.
        range_floors: Predictor -> ('abs', width) or ('rel', fraction of |median|);
            None disables widening.

    Returns:
        MessResult with per-candidate 'mess' (minimum over predictors) and 'mess_mod'
        (the limiting predictor).
    """
    floors = range_floors or {}
    cols: dict[str, np.ndarray] = {}
    widened: list[str] = []
    dropped: list[str] = []
    for name in predictors:
        ref = np.asarray(reference.get(name, []), float)
        ref = ref[~np.isnan(ref)]
        if name not in candidates.columns or len(ref) == 0:
            dropped.append(name)
            continue
        lo, hi = float(ref.min()), float(ref.max())
        width = _floor_width(name, float(np.median(ref)), floors)
        if width is not None and hi - lo < width:
            med = float(np.median(ref))
            lo, hi = min(lo, med - width / 2), max(hi, med + width / 2)
            widened.append(name)
        if not hi > lo:
            dropped.append(name)
            continue
        cols[name] = mess_similarity(ref, candidates[name].to_numpy(float), lo, hi)
    per = pd.DataFrame(
        {f"mess_{k}": v for k, v in cols.items()}, index=candidates.index
    )
    if per.shape[1] == 0:
        scores = pd.DataFrame(
            {"mess": np.nan, "mess_mod": None}, index=candidates.index
        )
    else:
        values = per.to_numpy()
        all_nan = np.all(np.isnan(values), axis=1)
        filled = np.where(np.isnan(values), np.inf, values)
        idx = np.argmin(filled, axis=1)
        names = np.array(list(cols))
        mess = filled[np.arange(len(per)), idx]
        mess[all_nan] = np.nan
        mod = np.where(all_nan, None, names[idx])
        scores = pd.DataFrame({"mess": mess, "mess_mod": mod}, index=candidates.index)
        scores = pd.concat([scores, per], axis=1)
    return MessResult(scores=scores, widened=widened, dropped=dropped)


def drop_collinear(
    pool: pd.DataFrame,
    predictors: Sequence[str],
    threshold: float = MESS_COLLINEARITY_R,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Drop collinear predictors (|r| > threshold), keeping the more interpretable one.

    Predictors are visited in priority order (the order given, see
    ``ENV_PREDICTORS``); one is kept if it is not correlated above ``threshold`` with
    any already-kept predictor. Correlations are computed over the candidate pool
    (per-cell medians), never the focal profile; constant columns (r undefined) are
    kept and never yield NaN.

    Args:
        pool: Candidate pool with one column per predictor.
        predictors: Predictors in priority order.
        threshold: Absolute Pearson r above which a pair is collinear.

    Returns:
        (kept predictors, list of {'dropped', 'kept', 'r'} records).
    """
    present = [p for p in predictors if p in pool.columns]
    corr = pool[present].astype(float).corr()
    kept: list[str] = []
    log: list[dict[str, Any]] = []
    for name in present:
        clash = None
        for other in kept:
            r = corr.loc[name, other]
            if pd.notna(r) and abs(r) > threshold:
                clash = (other, float(r))
                break
        if clash is None:
            kept.append(name)
        else:
            log.append({"dropped": name, "kept": clash[0], "r": round(clash[1], 4)})
    return kept, log


# --------------------------------------------------------------------------- #
# Envelope gate                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FocalEnvelope:
    """Focal values the envelope gate compares candidates against.

    Attributes:
        elevation: Focal median elevation (m).
        rainfall: Focal CHIRPS mean annual rainfall (mm/yr).
        class_mosaic: Focal class is a mosaic class (e.g. CM): class gate passes.
        class_fallback: VECEA missing at the focal site: use PNV + ecoregion.
        substrate_fallback: WRB missing at the focal site: use iSDA texture class.
        coastal_limit_m: If set, candidates must lie within this distance of the coast.
    """

    elevation: float
    rainfall: float
    class_mosaic: bool = False
    class_fallback: bool = False
    substrate_fallback: bool = False
    coastal_limit_m: float | None = None


def class_share_column(level: str, fallback: bool) -> str:
    """Column holding the class-gate share for a class level ('same'/'group').

    Args:
        level: 'same' or 'group'.
        fallback: True when the PNV + ecoregion fallback is in use.

    Returns:
        Column name in the Stage 1 table.
    """
    if fallback:
        return "pnv_eco_same_mean"
    return "in_group_set_mean" if level == "group" else "in_class_set_mean"


def substrate_share_column(level: str, fallback: bool) -> str | None:
    """Column holding the substrate-gate share ('same'/'related'/'off').

    Args:
        level: 'same', 'related' or 'off'.
        fallback: True when the iSDA texture fallback is in use.

    Returns:
        Column name, or None when the gate is off.
    """
    if level == "off":
        return None
    if fallback:
        return "texture_same_mean"
    return "wrb_related_mean" if level == "related" else "wrb_same_mean"


def envelope_gate(
    table: pd.DataFrame,
    focal: FocalEnvelope,
    *,
    class_level: str = "same",
    substrate_level: str = "same",
    elevation_tol_m: float = 300.0,
    rainfall_tol_frac: float = 0.25,
    hydro_match: bool = True,
    coarse_rule: bool = True,
    coarse_min_share: float = 0.05,
) -> pd.DataFrame:
    """Apply the Stage 1 hard filters to a table of raw per-cell reductions.

    Coarse cells pass a categorical gate when the in-set share is at least
    ``coarse_min_share`` (one focal-sized patch fits) and the elevation gate when the
    cell's p10-p90 range overlaps the focal window. Fine cells (and k == 1 grids,
    ``coarse_rule=False``) need an in-set share >= 0.5 and a median within tolerance.

    Args:
        table: Stage 1 table (columns ``<indicator>_mean``, ``elevation_p10/50/90``,
            ``rain_mean_annual`` and optionally ``coast_dist``).
        focal: Focal envelope values.
        class_level: 'same' or 'group'.
        substrate_level: 'same', 'related' or 'off'.
        elevation_tol_m: Elevation tolerance (m).
        rainfall_tol_frac: Rainfall tolerance (fraction of focal).
        hydro_match: Apply the HAND riparian/upland gate.
        coarse_rule: Use the coarse share/overlap rules (else fine rules).
        coarse_min_share: Share threshold for coarse categorical gates.

    Returns:
        DataFrame (same index) with boolean columns gate_land, gate_class,
        gate_substrate, gate_elevation, gate_rainfall, gate_coastal, gate_hydro and
        'passes', plus evidence columns class_share, substrate_share, hand_share.
    """
    share_min = coarse_min_share if coarse_rule else FINE_MIN_SHARE
    out = pd.DataFrame(index=table.index)

    def col(name: str) -> pd.Series:
        if name in table.columns:
            return table[name].astype(float)
        return pd.Series(np.nan, index=table.index)

    out["gate_land"] = col("land_mean") >= MIN_LAND_SHARE
    class_share = col(class_share_column(class_level, focal.class_fallback))
    out["class_share"] = class_share
    out["gate_class"] = True if focal.class_mosaic else class_share >= share_min
    sub_col = substrate_share_column(substrate_level, focal.substrate_fallback)
    if sub_col is None:
        out["substrate_share"] = np.nan
        out["gate_substrate"] = True
    else:
        out["substrate_share"] = col(sub_col)
        out["gate_substrate"] = out["substrate_share"] >= share_min
    if coarse_rule:
        out["gate_elevation"] = (
            col("elevation_p10") <= focal.elevation + elevation_tol_m
        ) & (col("elevation_p90") >= focal.elevation - elevation_tol_m)
    else:
        out["gate_elevation"] = (
            col("elevation_p50") - focal.elevation
        ).abs() <= elevation_tol_m
    out["gate_rainfall"] = (
        col("rain_mean_annual") - focal.rainfall
    ).abs() <= rainfall_tol_frac * focal.rainfall
    if focal.coastal_limit_m is None:
        out["gate_coastal"] = True
    else:
        out["gate_coastal"] = col("coast_dist") <= focal.coastal_limit_m
    out["hand_share"] = col("hand_focal_mean")
    out["gate_hydro"] = (out["hand_share"] >= share_min) if hydro_match else True
    gates = [c for c in out.columns if c.startswith("gate_")]
    out[gates] = out[gates].fillna(False).astype(bool)
    out["passes"] = out[gates].all(axis=1)
    return out


# --------------------------------------------------------------------------- #
# Embeddings                                                                   #
# --------------------------------------------------------------------------- #
def normalise_rows(vectors: np.ndarray) -> np.ndarray:
    """Renormalise vectors to unit length (a mean of unit vectors is shorter than 1).

    Args:
        vectors: (n, d) array (or (d,) for one vector).

    Returns:
        Array of the same shape with unit-length rows (zero rows become NaN).
    """
    v = np.asarray(vectors, float)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 0, v / norm, np.nan)


def mean_embedding_vector(vectors: np.ndarray) -> np.ndarray:
    """Mean of several (already cell-averaged) embedding vectors, renormalised.

    Args:
        vectors: (n, 64) array; rows with NaN are ignored.

    Returns:
        (64,) unit-length query vector.

    Raises:
        ValueError: If no finite vector is supplied.
    """
    v = np.asarray(vectors, float)
    v = v[~np.isnan(v).any(axis=1)]
    if len(v) == 0:
        raise ValueError("No valid embedding vectors to average.")
    return normalise_rows(normalise_rows(v).mean(axis=0))


def embedding_query_similarity(
    embeddings: pd.DataFrame,
    query: np.ndarray,
    bands: Sequence[str] = tuple(SATELLITE_EMBEDDING_BANDS),
) -> pd.Series:
    """Cosine similarity of each cell's mean embedding to a query vector.

    Both sides are renormalised to unit length before the dot product.

    Args:
        embeddings: DataFrame with one row per cell and columns ``bands``.
        query: Query vector (renormalised here).
        bands: Embedding band columns.

    Returns:
        Series of cosine similarities (NaN for cells without a valid embedding).
    """
    vecs = normalise_rows(embeddings[list(bands)].to_numpy(float))
    q = normalise_rows(np.asarray(query, float))
    return pd.Series(vecs @ q, index=embeddings.index, name="embed_sim")


def resolve_seed_source(
    has_user_seeds: bool, auto_seeds: str, n_wdpa_strict_cells: int, min_cells: int = 2
) -> tuple[str, bool]:
    """Resolve the reference embedding query source (spec §5.4).

    Args:
        has_user_seeds: seed_reference_sites were supplied.
        auto_seeds: 'none' or 'wdpa_strict'.
        n_wdpa_strict_cells: Stage 1 survivors with >= 50% IUCN Ia-II overlap.
        min_cells: Minimum WDPA-strict cells for Scenario A'.

    Returns:
        (seed_source, agree_independent): ('user', True), ('wdpa_strict', True) or
        ('pseudo', False).
    """
    if has_user_seeds:
        return "user", True
    if auto_seeds == "wdpa_strict" and n_wdpa_strict_cells >= min_cells:
        return "wdpa_strict", True
    return "pseudo", False
