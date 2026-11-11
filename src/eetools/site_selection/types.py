"""Typed configuration and result containers for the site-selection workflows."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import pandas as pd

DegradationType = Literal[
    "generic",
    "mining",
    "bare_soil",
    "cropland",
    "deforestation",
    "overgrazing",
    "rangeland",
    "settlement",
]
SiteType = Literal["reference", "degraded"]

_WEIGHT_TOL = 1e-6

# Provenance-key fields a run may *complete*: tables whose key has None here (built by
# a run that does not use the field, e.g. a reference run without
# project_start_year) are reused, and the reusing run records its value. A field only
# mismatches when both keys set it to different values.
COMPLETABLE_KEY_FIELDS = ("project_start_year",)


class ScreeningTablesMismatch(ValueError):
    """Raised when supplied ScreeningTables were built with a different provenance key."""


@dataclass(frozen=True)
class EnvelopeTolerances:
    """Stage 1 envelope-gate tolerances and search-ladder settings (spec §5.1).

    Every tolerance actually applied is recorded in ``params`` and in the layer
    register at [L] confidence ("default threshold, to be field-checked").

    Attributes:
        class_match: National vegetation class gate: 'same' class set, or 'group'
            (adjacent classes in the same VECEA_CLASS_GROUPS group).
        focal_class_min_share: A class enters the focal class set when it covers at
            least this share of the focal site's pixels.
        min_focal_class_pixels: Below this many native pixels, a categorical layer's
            focal shares are computed over focal site + reference_buffer_m.
        coarse_min_share: Share threshold for coarse-cell categorical gates; None ->
            max(focal area / cell area, COARSE_MIN_SHARE_FLOOR).
        substrate_match: WRB soil group gate: 'same', 'related' or 'off'.
        elevation_tol_m: Elevation envelope (SOP default 300 m).
        rainfall_tol_frac: Mean-annual rainfall envelope (SOP default 0.25).
        hydro_match: Require the focal-majority HAND class (riparian/upland).
        coastal_mode: 'auto' (on when focal site is near the ocean), 'on' or 'off'.
        mess_floor: Stage 2 MESS floor (cells below are dropped).
        relaxation: Search-ladder variant: 'before_tier3', 'after_tier3' or 'off'.
        min_screen_pool: Stage 1 survivors needed for a rung to stop the ladder;
            None -> 3 x n_candidates.
        exclude_max_share: A cell is dropped if more than this share lies inside the
            focal site + exclusion buffer (or exclude_fc).
        min_natural_persist: Reference land-use gate: a final cell needs at least this
            share of pixels that are Dynamic World natural in the required number of
            years (``natural_persist_years``). Applied inside each rung before
            scoring, thinning and the stop rule ([L]).
        natural_persist_years: 'all_but_one' (default; tolerates one off-year label,
            e.g. DW grass/shrub/crop swapping, or edge pixels in large cells) or 'all'
            (natural in every year of the land-use history). Both shares are reduced,
            so this never re-reduces. Default chosen on the v0.6.0 validation: at
            TGBS_Kwale's 2.5 km cells the known Buda reference failed 'all' (median
            0.77) and passed 'all_but_one' (0.92); Ecosia kept all 18 strict-PA cells
            either way.
        unit_coverage_stop: The search ladder stops only when the rung has enough
            eligible thinned references *and* each management unit (focal class) has
            at least one; False stops on the count alone (saves Tier 3 compute).
    """

    class_match: Literal["same", "group"] = "same"
    focal_class_min_share: float = 0.2
    min_focal_class_pixels: int = 25
    coarse_min_share: float | None = None
    substrate_match: Literal["same", "related", "off"] = "same"
    elevation_tol_m: float = 300.0
    rainfall_tol_frac: float = 0.25
    hydro_match: bool = True
    coastal_mode: Literal["auto", "on", "off"] = "auto"
    mess_floor: float = -25.0
    relaxation: Literal["before_tier3", "after_tier3", "off"] = "before_tier3"
    min_screen_pool: int | None = None
    exclude_max_share: float = 0.5
    min_natural_persist: float = 0.8
    natural_persist_years: Literal["all", "all_but_one"] = "all_but_one"
    unit_coverage_stop: bool = True

    def __post_init__(self) -> None:
        _check_choice(
            "natural_persist_years", self.natural_persist_years, ("all", "all_but_one")
        )
        _check_unit("min_natural_persist", self.min_natural_persist)
        _check_choice("class_match", self.class_match, ("same", "group"))
        _check_choice(
            "substrate_match", self.substrate_match, ("same", "related", "off")
        )
        _check_choice("coastal_mode", self.coastal_mode, ("auto", "on", "off"))
        _check_choice(
            "relaxation", self.relaxation, ("before_tier3", "after_tier3", "off")
        )
        _check_unit("focal_class_min_share", self.focal_class_min_share, open_low=True)
        _check_unit("exclude_max_share", self.exclude_max_share)
        if self.coarse_min_share is not None:
            _check_unit("coarse_min_share", self.coarse_min_share, open_low=True)
        if self.min_focal_class_pixels < 1:
            raise ValueError("min_focal_class_pixels must be >= 1.")
        if self.elevation_tol_m <= 0 or self.rainfall_tol_frac <= 0:
            raise ValueError("elevation_tol_m and rainfall_tol_frac must be > 0.")
        if self.min_screen_pool is not None and self.min_screen_pool < 1:
            raise ValueError("min_screen_pool must be >= 1.")


@dataclass(frozen=True)
class ReferenceWeights:
    """Intactness-score weights (spec §5.3). Must sum to 1.

    For the forest stratum (forest, riverine, bamboo), ``forest_loss_shift`` moves from ``core`` to
    pct(1 - forest_loss_frac) and ``canopy_shift`` moves from ``performance`` to
    pct(canopy_height_p50). ``natural_persist`` scores the Dynamic World persistent-
    natural share (it replaced BII in v0.6.0: BII did not separate intact from
    degraded cells at the TGBS sites).
    """

    naturalness: float = 0.25
    natural_cover: float = 0.15
    core: float = 0.15
    performance: float = 0.10
    moisture: float = 0.05
    persistence: float = 0.05
    burn_persist: float = 0.05
    natural_persist: float = 0.10
    protection: float = 0.10
    forest_loss_shift: float = 0.05
    canopy_shift: float = 0.05

    def __post_init__(self) -> None:
        _check_weights(self, exclude=("forest_loss_shift", "canopy_shift"))
        if self.forest_loss_shift > self.core or self.canopy_shift > self.performance:
            raise ValueError(
                "forest_loss_shift must be <= core and canopy_shift <= performance."
            )


@dataclass(frozen=True)
class DegradedWeights:
    """Degradation-score weights (spec §5.3, amended §0a D3). Must sum to 1.

    STATE + TRAJECTORY only. The embedding/rules signature is a pathway gate, not a
    ranking weight (a weight penalised sites further along the same pathway). The
    defaults are the original weights without the 0.35 signature term, renormalised
    (thirteenths). ``loss`` applies to forest strata only; otherwise it is
    redistributed proportionally to the four STATE terms.
    """

    perf_deficit: float = 3 / 13
    moist_deficit: float = 2 / 13
    unnatural: float = 3 / 13
    pressure: float = 2 / 13
    browning: float = 1 / 13
    nbr_decline: float = 1 / 13
    loss: float = 1 / 13

    def __post_init__(self) -> None:
        _check_weights(self)


def default_degraded_weights(degradation_type: str) -> DegradedWeights:
    """Default DegradedWeights for a degradation pathway.

    The gHM ``pressure`` term ranks peri-urban cells as more degraded, which pulls
    picks toward built-up sprawl (2026-11-08 / 11-09 reviews). It is kept only for
    ``degradation_type='settlement'``; for every other pathway its weight is 0 and the
    remaining terms keep their proportions.

    Args:
        degradation_type: Degradation pathway.

    Returns:
        DegradedWeights.
    """
    w = DegradedWeights()
    if degradation_type == "settlement":
        return w
    rest = 1.0 - w.pressure
    return DegradedWeights(
        perf_deficit=w.perf_deficit / rest,
        moist_deficit=w.moist_deficit / rest,
        unnatural=w.unnatural / rest,
        pressure=0.0,
        browning=w.browning / rest,
        nbr_decline=w.nbr_decline / rest,
        loss=w.loss / rest,
    )


def _check_choice(name: str, value: Any, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}.")


def _check_unit(name: str, value: float, open_low: bool = False) -> None:
    low_ok = value > 0 if open_low else value >= 0
    if not (low_ok and value <= 1):
        raise ValueError(f"{name} must be in {'(0' if open_low else '[0'}, 1].")


def _check_weights(obj: Any, exclude: tuple[str, ...] = ()) -> None:
    values = {
        f.name: getattr(obj, f.name) for f in fields(obj) if f.name not in exclude
    }
    negative = [k for k, v in values.items() if v < 0]
    if negative:
        raise ValueError(f"Weights must be non-negative: {negative}.")
    total = sum(values.values())
    if not math.isclose(total, 1.0, abs_tol=_WEIGHT_TOL):
        raise ValueError(
            f"{type(obj).__name__} weights must sum to 1 (got {total:.6f})."
        )


def _jsonable(value: Any) -> Any:
    """Normalise a value so provenance keys compare equal after a JSON round-trip."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


@dataclass(frozen=True)
class ScreeningTables:
    """Raw per-cell Earth Engine reductions, shared between runs (spec §5.3).

    Holds raw values only (gate-indicator shares, elevation, climate, MESS predictors,
    condition mean/p10/p90, labels, embeddings), each family flagged by a
    ``_done_<family>`` column. It never holds pools, MESS scores, percentiles, ranks
    or the spanning set; every run recomputes those for its own pool.

    Attributes:
        coarse: Per-coarse-cell reductions, indexed by cell_id.
        fine: Per-fine-cell reductions, indexed by cell_id.
        embeddings: Mean embedding per (cell_id, year): columns cell_id, year, A00..A63.
        labels: Per-cell VECEA / WRB frequency histograms (JSON strings), indexed by
            cell_id.
        key: Provenance key; a run whose key differs raises ScreeningTablesMismatch.
        created: ISO-8601 creation timestamp (UTC).
    """

    coarse: pd.DataFrame
    fine: pd.DataFrame
    embeddings: pd.DataFrame
    labels: pd.DataFrame
    key: dict[str, Any]
    created: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def check_key(self, key: dict[str, Any]) -> None:
        """Raise ScreeningTablesMismatch if ``key`` differs from the stored key.

        Args:
            key: Provenance key of the run that wants to reuse these tables.

        Returns:
            None.

        Raises:
            ScreeningTablesMismatch: Naming every differing field.
        """
        mine = _jsonable(self.key)
        theirs = _jsonable(key)
        diff = sorted(
            k
            for k in set(mine) | set(theirs)
            if mine.get(k) != theirs.get(k)
            and not (
                k in COMPLETABLE_KEY_FIELDS
                and (mine.get(k) is None or theirs.get(k) is None)
            )
        )
        if diff:
            raise ScreeningTablesMismatch(
                "ScreeningTables were built with a different provenance key; "
                f"differing fields: {diff}. Re-run without screening= (or with "
                "matching inputs)."
            )

    def to_parquet(self, path: str | Path) -> None:
        """Write the tables to a local directory of parquet files plus meta.json.

        Args:
            path: Output directory (created if missing). Writes coarse.parquet,
                fine.parquet, embeddings.parquet, labels.parquet and meta.json
                (provenance key and creation time). Full float64 precision is kept.

        Returns:
            None.
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("coarse", "fine", "labels"):
            df = getattr(self, name)
            df.to_parquet(out / f"{name}.parquet", index=True)
        self.embeddings.to_parquet(out / "embeddings.parquet", index=False)
        meta = {"key": _jsonable(self.key), "created": self.created}
        (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    @classmethod
    def from_parquet(cls, path: str | Path) -> ScreeningTables:
        """Read tables written by :meth:`to_parquet`.

        Args:
            path: Directory written by to_parquet.

        Returns:
            ScreeningTables with the stored key and creation time.
        """
        src = Path(path)
        meta = json.loads((src / "meta.json").read_text())
        return cls(
            coarse=pd.read_parquet(src / "coarse.parquet"),
            fine=pd.read_parquet(src / "fine.parquet"),
            embeddings=pd.read_parquet(src / "embeddings.parquet"),
            labels=pd.read_parquet(src / "labels.parquet"),
            key=meta["key"],
            created=meta["created"],
        )

    @classmethod
    def empty(cls, key: dict[str, Any]) -> ScreeningTables:
        """Return empty tables for a provenance key.

        Args:
            key: Provenance key.

        Returns:
            ScreeningTables with no rows.
        """

        def _frame() -> pd.DataFrame:
            return pd.DataFrame(index=pd.Index([], name="cell_id", dtype=object))

        return cls(
            coarse=_frame(),
            fine=_frame(),
            embeddings=pd.DataFrame(columns=["cell_id", "year"]),
            labels=_frame(),
            key=_jsonable(key),
        )


@dataclass(frozen=True)
class SiteSelectionResult:
    """Output of identify_reference_sites / identify_degraded_sites (spec §5.5).

    Attributes:
        site_type: 'reference' or 'degraded'.
        ranked: Top-N proposed sites, rank 1..N (EPSG:4326 cell polygons).
        pool: Every final cell of the rung reached that passed the fine envelope
            gate, with ``pool_status`` ('eligible' / 'gated_out'), ``gate_reason``
            and ``landuse_class``. Only eligible cells carry scores.
        focal_profile: Focal samples of condition-independent predictors; column
            ``sample_set`` is 'mess' (native-scale MESS reference sample) or 'span'
            (joint sample used for environmental-range spanning).
        params: Every input, default, weight, tolerance and derived setting used.
        provenance: Dataset IDs, versions and licence notes.
        selection_matrix: CER-K SOP 0/1/2 matrix with evidence and H/M/L confidence.
        layer_register: CER-K SOP §7 layer and query register.
        screening: Raw per-cell reductions; pass as ``screening=`` to the other
            workflow on the same focal site.
        warnings: Human-readable warnings (ladder expanded, gate relaxed, ...).
        top_candidates: The best distinct eligible sites by ``top_score`` (condition
            percentile and ``env_sim``), with ``proposed`` flagging the shortlist; see
            ``scoring.select_top_candidates``. None for results built before 0.6.1.
    """

    site_type: SiteType
    ranked: gpd.GeoDataFrame
    pool: gpd.GeoDataFrame
    focal_profile: pd.DataFrame
    params: dict[str, Any]
    provenance: dict[str, str]
    selection_matrix: pd.DataFrame
    layer_register: pd.DataFrame
    screening: ScreeningTables
    warnings: list[str]
    top_candidates: gpd.GeoDataFrame | None = None


def dataclass_params(obj: Any) -> dict[str, Any]:
    """Return a dataclass instance's fields as a plain dict (for ``params``).

    Args:
        obj: A dataclass instance.

    Returns:
        Dict of field name to value.
    """
    return asdict(obj)
