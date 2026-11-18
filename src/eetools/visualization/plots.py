from collections.abc import Collection, Mapping, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def add_year_shading(
    ax,
    dry_start: str = "2023-05-01",
    dry_end: str = "2023-09-30",
    wet_start: str = "2025-01-01",
    wet_end: str = "2025-06-30",
    dry_label: str = "Dry year",
    wet_label: str = "Wet year",
    alpha: float = 0.12,
):
    """Add dry-year and wet-year shaded date windows to a matplotlib axis.

    Args:
        ax: matplotlib Axes object to annotate.
        dry_start: Start date of the dry shading window as an ISO string (default '2023-05-01').
        dry_end: End date of the dry shading window as an ISO string (default '2023-09-30').
        wet_start: Start date of the wet shading window as an ISO string (default '2025-01-01').
        wet_end: End date of the wet shading window as an ISO string (default '2025-06-30').
        dry_label: Legend label for the dry shading band (default 'Dry year').
        wet_label: Legend label for the wet shading band (default 'Wet year').
        alpha: Opacity of the shading bands (default 0.12).

    Returns:
        matplotlib Axes object with dry and wet axvspan patches added.
    """
    dry_start = pd.to_datetime(dry_start)
    dry_end = pd.to_datetime(dry_end)
    wet_start = pd.to_datetime(wet_start)
    wet_end = pd.to_datetime(wet_end)

    ax.axvspan(dry_start, dry_end, alpha=alpha, label=dry_label)
    ax.axvspan(wet_start, wet_end, alpha=alpha, label=wet_label)

    return ax


def add_year_markers(
    ax,
    dry_date: str = "2023-01-01",
    wet_date: str = "2025-01-01",
    dry_label: str = "2023 dry year",
    wet_label: str = "2025 wet year",
):
    """Add vertical dashed markers for dry and wet reference years to a matplotlib axis.

    Args:
        ax: matplotlib Axes object to annotate.
        dry_date: Date string for the dry-year marker line (default '2023-01-01').
        wet_date: Date string for the wet-year marker line (default '2025-01-01').
        dry_label: Text label placed at the dry-year marker (default '2023 dry year').
        wet_label: Text label placed at the wet-year marker (default '2025 wet year').

    Returns:
        matplotlib Axes object with dry and wet vertical dashed lines and text annotations added.
    """
    dry_date = pd.to_datetime(dry_date)
    wet_date = pd.to_datetime(wet_date)

    ax.axvline(dry_date, linestyle="--", linewidth=1)
    ax.axvline(wet_date, linestyle="--", linewidth=1)

    ax.text(dry_date, ax.get_ylim()[1], dry_label, rotation=90, va="top", ha="right")
    ax.text(wet_date, ax.get_ylim()[1], wet_label, rotation=90, va="top", ha="right")

    return ax


def plot_single_metric_annually(
    df: pd.DataFrame,
    title: str,
    date_col: str = "date",
    metric_col: str = "metric",
    figsize: tuple = (10, 5),
):
    """Plot a single annual timeseries metric as a clean lineplot.

    Args:
        df: pd.DataFrame containing date_col and metric_col.
        title: Title string for the plot.
        date_col: Column name for dates; parsed to datetime before plotting (default 'date').
        metric_col: Column name for the metric values to plot (default 'metric').
        figsize: Figure size as a (width, height) tuple in inches (default (10, 5)).

    Returns:
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col)

    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(data=d, x=date_col, y=metric_col, linewidth=2, ax=ax)

    ax.set_title(title, pad=12)
    ax.set_xlabel("Date", labelpad=10)
    ax.set_ylabel(metric_col, labelpad=10)
    ax.grid(True)

    plt.tight_layout()

    return fig, ax


def plot_single_metric_monthly(
    df: pd.DataFrame,
    title: str,
    date_col: str = "date",
    metric_col: str = "metric",
    figsize: tuple = (12, 5),
    month_interval: int = 6,
):
    """Plot a single monthly timeseries metric as a lineplot with a configurable date
    tick interval.

    Args:
        df: pd.DataFrame containing date_col and metric_col.
        title: Title string for the plot.
        date_col: Column name for dates; parsed to datetime before plotting (default 'date').
        metric_col: Column name for the metric values to plot (default 'metric').
        figsize: Figure size as a (width, height) tuple in inches (default (12, 5)).
        month_interval: Interval in months between x-axis tick marks (default 6).

    Returns:
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col)

    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(data=d, x=date_col, y=metric_col, linewidth=1.8, ax=ax)

    ax.set_title(title, pad=12)
    ax.set_xlabel("Date", labelpad=10)
    ax.set_ylabel(metric_col, labelpad=10)
    ax.grid(True)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)

    plt.tight_layout()

    return fig, ax


def plot_multi_metric_annually(
    df: pd.DataFrame,
    title: str,
    date_col: str = "date",
    metric_col: list[str] | tuple[str, ...] | None = None,
    figsize: tuple = (12, 6),
):
    """Plot multiple annual timeseries metrics overlaid on the same axis.

    Args:
        df: pd.DataFrame containing date_col and all columns listed in metric_col.
        title: Title string for the plot.
        date_col: Column name for dates; parsed to datetime before plotting (default 'date').
        metric_col: List or tuple of metric column names to plot; raises ValueError if None.
        figsize: Figure size as a (width, height) tuple in inches (default (12, 6)).

    Returns:
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
    """
    if metric_col is None:
        raise ValueError("metric_col must be a list or tuple of metric column names.")

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col)

    fig, ax = plt.subplots(figsize=figsize)

    for col in metric_col:
        sns.lineplot(data=d, x=date_col, y=col, label=col, linewidth=2, ax=ax)

    ax.set_title(title, pad=12)
    ax.set_xlabel("Date", labelpad=10)
    ax.set_ylabel("Metric Value", labelpad=10)
    ax.grid(True)
    ax.legend(title="Metric")

    plt.tight_layout()

    return fig, ax


def plot_multi_metric_monthly(
    df: pd.DataFrame,
    title: str,
    date_col: str = "date",
    metric_col: list[str] | tuple[str, ...] | None = None,
    figsize: tuple = (14, 6),
    month_interval: int = 6,
):
    """Plot multiple monthly timeseries metrics overlaid on the same axis with a
    configurable date tick interval.

    Args:
        df: pd.DataFrame containing date_col and all columns listed in metric_col.
        title: Title string for the plot.
        date_col: Column name for dates; parsed to datetime before plotting (default 'date').
        metric_col: List or tuple of metric column names to plot; raises ValueError if None.
        figsize: Figure size as a (width, height) tuple in inches (default (14, 6)).
        month_interval: Interval in months between x-axis tick marks (default 6).

    Returns:
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
    """
    if metric_col is None:
        raise ValueError("metric_col must be a list or tuple of metric column names.")

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col)

    fig, ax = plt.subplots(figsize=figsize)

    for col in metric_col:
        sns.lineplot(data=d, x=date_col, y=col, label=col, linewidth=1.8, ax=ax)

    ax.set_title(title, pad=12)
    ax.set_xlabel("Date", labelpad=10)
    ax.set_ylabel("Metric Value", labelpad=10)
    ax.grid(True)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)

    ax.legend(title="Metric")
    plt.tight_layout()

    return fig, ax


# --------------------------------------------------------------------------- #
# Site-selection condition diagnostics                                        #
# --------------------------------------------------------------------------- #
# Per-cell condition signals compared across site-selection pools: higher = more intact
# (+1) or less intact (-1). Datasets the workflows no longer use (SBTN, BII, dry-season
# greenness) are deliberately absent.
CONDITION_SIGNALS: dict[str, int] = {
    "natural_frac": 1,  # ESA WorldCover natural cover
    "dw_natural_recent": 1,  # Dynamic World natural-label share, recent years
    "dw_natural_persist": 1,  # Dynamic World natural label in every year
    "dw_crop_recent": -1,  # Dynamic World crop-label share, recent years
    "dw_crop_any_year": -1,  # Dynamic World crop label in any year
    "ghm": -1,  # global Human Modification
    "ndvi_state": 1,
    "ndmi_state": 1,
    "core_frac": 1,
}

CONDITION_GROUP_COLUMNS: dict[str, str] = {
    "Strict PA": "strict_pa",
    "Ref top quartile": "ref_topq",
    "Ref pool": "in_ref_pool",
    "Deg pool": "in_deg_pool",
}

# Site-selection pool columns (dw_landuse family, 0.6.0+) that stand in for condition
# signals of the same meaning. dw_crop_share is the crop share of the most frequent
# label over the state window, i.e. recent crop.
POOL_CONDITION_COLUMNS: dict[str, str] = {
    "dw_nat_share_all": "dw_natural_persist",
    "dw_crop_share": "dw_crop_recent",
}

# Condition signal -> the site_selection.ReferenceWeights field that scores it. When
# that weight is non-zero, the reference top quartile is chosen partly on the signal,
# so its AUC against the degraded pool is circular.
CONDITION_SIGNAL_WEIGHTS: dict[str, str] = {
    "natural_frac": "natural_cover",
    "dw_natural_persist": "natural_persist",
    "core_frac": "core",
    "ghm": "naturalness",
    "ndvi_state": "performance",
    "ndmi_state": "moisture",
}

# Signals the degraded pool is selected on, so they are circular on the degraded side:
# the D1 anchor constraint and the degradation score use NDVI / NDMI state for every
# pathway; the conversion test (Dynamic World modified share, which also corrects
# natural_frac) applies to conversion pathways; the cropland gate (max_dw_crop) to
# natural-cover pathways.
DEGRADED_SELECTION_SIGNALS: tuple[str, ...] = ("ndvi_state", "ndmi_state")
CONVERSION_SELECTION_SIGNALS: tuple[str, ...] = (
    "natural_frac",
    "dw_natural_persist",
    "dw_crop_recent",
)
NATURAL_COVER_SELECTION_SIGNALS: tuple[str, ...] = ("dw_crop_recent",)


def separation_auc(pos: pd.Series, neg: pd.Series) -> float:
    """Rank-based AUC (Mann-Whitney U / (n_pos * n_neg)) of ``pos`` over ``neg``.

    0.5 means no separation, 1.0 means every ``pos`` value exceeds every ``neg`` value.
    Ties count half. NaNs are dropped.

    Args:
        pos: Values of the group expected to score higher.
        neg: Values of the comparison group.

    Returns:
        AUC in [0, 1], or NaN if either group is empty.
    """
    p = pd.Series(pos, dtype=float).dropna().to_numpy()
    n = pd.Series(neg, dtype=float).dropna().to_numpy()
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    ranks = pd.Series(np.r_[p, n]).rank().to_numpy()
    u = ranks[: len(p)].sum() - len(p) * (len(p) + 1) / 2
    return float(u / (len(p) * len(n)))


def condition_groups(
    cells: pd.DataFrame, group_columns: Mapping[str, str] | None = None
) -> dict[str, pd.DataFrame]:
    """Split a per-cell condition table into the diagnostic comparison groups.

    Args:
        cells: One row per cell with boolean group columns (default: ``strict_pa``,
            ``ref_topq``, ``in_ref_pool``, ``in_deg_pool``).
        group_columns: Group label -> boolean column (default CONDITION_GROUP_COLUMNS).
            Missing columns give empty groups.

    Returns:
        dict of group label -> subset of ``cells``.
    """
    cols = dict(group_columns or CONDITION_GROUP_COLUMNS)
    out = {}
    for label, col in cols.items():
        mask = (
            cells[col].fillna(False).astype(bool)
            if col in cells
            else pd.Series(False, index=cells.index)
        )
        out[label] = cells.loc[mask]
    return out


def condition_cells_from_pools(
    reference_pool: pd.DataFrame,
    degraded_pool: pd.DataFrame,
    *,
    strict_pa_min: float = 0.5,
    top_quantile: float = 0.75,
    score_column: str = "intactness",
) -> pd.DataFrame:
    """Per-cell condition table from site-selection pools (no Earth Engine).

    Takes the ``result.pool`` frames (or the exported ``<kind>_pool.gpkg``) of a
    reference and a degraded run and builds the input of ``condition_groups`` /
    ``condition_separation_table`` / ``plot_condition_signals``. Only eligible cells
    (``pool_status == "eligible"``; every row when the column is absent) enter. Pool
    columns are renamed to the signal names with ``POOL_CONDITION_COLUMNS``; a cell in
    both pools takes its values from the reference pool (the raw reductions are the
    same).

    Group columns: ``in_ref_pool`` / ``in_deg_pool`` (eligible cells), ``ref_topq``
    (reference cells with ``score_column`` at or above its ``top_quantile`` quantile)
    and ``strict_pa`` (reference cells with ``pa_strict_share >= strict_pa_min``).

    Args:
        reference_pool: Reference pool with ``cell_id`` and the signal columns.
        degraded_pool: Degraded pool with ``cell_id`` and the signal columns.
        strict_pa_min: Minimum strict (IUCN I-IV) protected share for ``strict_pa``
            (default 0.5).
        top_quantile: Quantile of ``score_column`` that starts the reference top
            group (default 0.75, the top quartile).
        score_column: Reference score that defines the top group (default
            ``intactness``).

    Returns:
        DataFrame indexed by ``cell_id`` (sorted) with the available signals,
        ``pa_strict_share`` and the four boolean group columns.

    Raises:
        ValueError: If a pool has no ``cell_id`` column, or the reference pool lacks
            ``score_column``.
    """
    for name, pool in (
        ("reference_pool", reference_pool),
        ("degraded_pool", degraded_pool),
    ):
        if "cell_id" not in pool:
            raise ValueError(f"{name} has no cell_id column.")
    if score_column not in reference_pool:
        raise ValueError(f"reference_pool has no {score_column!r} column.")

    def eligible(pool: pd.DataFrame) -> pd.DataFrame:
        if "pool_status" not in pool:
            return pool
        return pool[pool["pool_status"] == "eligible"]

    def values(pool: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(pool).rename(columns=POOL_CONDITION_COLUMNS)
        keep = [c for c in (*CONDITION_SIGNALS, "pa_strict_share") if c in df]
        return df[["cell_id", *keep]]

    ref, deg = eligible(reference_pool), eligible(degraded_pool)
    cells = (
        pd.concat([values(ref), values(deg)], ignore_index=True)
        .drop_duplicates("cell_id")
        .set_index("cell_id")
        .sort_index()
    )
    cells["in_ref_pool"] = cells.index.isin(ref["cell_id"])
    cells["in_deg_pool"] = cells.index.isin(deg["cell_id"])
    score = pd.to_numeric(ref[score_column], errors="coerce")
    top = ref.loc[score >= score.quantile(top_quantile), "cell_id"]
    cells["ref_topq"] = cells.index.isin(top)
    pa = pd.to_numeric(
        cells.get("pa_strict_share", pd.Series(np.nan, index=cells.index)),
        errors="coerce",
    )
    cells["strict_pa"] = cells["in_ref_pool"] & (pa >= strict_pa_min).to_numpy()
    return cells


def circular_condition_signals(
    reference_weights: object | None = None,
    degradation_type: str = "generic",
) -> set[str]:
    """Signals whose reference top quartile vs degraded pool AUC is circular.

    That AUC compares two groups the workflows selected, so it is circular for a
    signal either group was chosen on:

    - reference side: signals scored by a non-zero ``ReferenceWeights`` field
      (``CONDITION_SIGNAL_WEIGHTS``), because the top quartile is a cut of the
      intactness score;
    - degraded side: ``DEGRADED_SELECTION_SIGNALS`` for every pathway, plus
      ``CONVERSION_SELECTION_SIGNALS`` (conversion pathways) or
      ``NATURAL_COVER_SELECTION_SIGNALS`` (natural-cover pathways).

    The strict-PA vs rest AUC is not circular: neither the reference gate nor the
    signals use the protected-area label.

    Args:
        reference_weights: The ``site_selection.ReferenceWeights`` of the reference
            run (default: ``ReferenceWeights()``).
        degradation_type: The degraded run's ``degradation_type`` (default
            ``generic``).

    Returns:
        Set of condition-signal names.
    """
    from eetools.constants import NATURAL_COVER_DEGRADATION_TYPES
    from eetools.site_selection.types import ReferenceWeights

    w = reference_weights if reference_weights is not None else ReferenceWeights()
    out = {
        s
        for s, field in CONDITION_SIGNAL_WEIGHTS.items()
        if float(getattr(w, field, 0.0)) > 0
    }
    out |= set(DEGRADED_SELECTION_SIGNALS)
    if degradation_type in NATURAL_COVER_DEGRADATION_TYPES:
        out |= set(NATURAL_COVER_SELECTION_SIGNALS)
    else:
        out |= set(CONVERSION_SELECTION_SIGNALS)
    return out


def condition_separation_table(
    cells: pd.DataFrame,
    signals: Mapping[str, int] | None = None,
    circular: Collection[str] | None = None,
) -> pd.DataFrame:
    """Medians per group and separation AUCs for each condition signal.

    AUCs are oriented so that higher always means "more intact" (signals with
    direction -1 are negated):

    - ``auc_topq_vs_deg``: reference top quartile vs degraded pool. Circular for signals
      either group was selected on (see ``circular_condition_signals``).
    - ``auc_pa_vs_rest``: strict protected-area cells vs the rest of the reference pool,
      an independent label where strict PAs exist.

    Args:
        cells: Per-cell table with the signal and group columns (see condition_groups).
        signals: Signal -> direction (default CONDITION_SIGNALS); absent columns skipped.
        circular: Signals whose ``auc_topq_vs_deg`` is circular. When given, that AUC
            is set to NaN for them, ``topq_vs_deg_circular`` flags them, and rows are
            sorted by the highest remaining AUC.

    Returns:
        DataFrame with one row per signal, sorted by ``auc_topq_vs_deg`` (descending),
        or by the highest non-circular AUC when ``circular`` is given.
    """
    sig = {s: d for s, d in (signals or CONDITION_SIGNALS).items() if s in cells}
    g = condition_groups(cells)
    ref_pool = g["Ref pool"]
    pa_ids = g["Strict PA"].index
    rest = ref_pool.loc[~ref_pool.index.isin(pa_ids)]
    rows = []
    for s, d in sig.items():
        row = {"signal": s, "direction": d}
        for label, sub in g.items():
            key = "median_" + label.lower().replace(" ", "_")
            row[key] = sub[s].median() if len(sub) else np.nan
        row["auc_topq_vs_deg"] = separation_auc(
            g["Ref top quartile"][s] * d, g["Deg pool"][s] * d
        )
        row["auc_pa_vs_rest"] = separation_auc(g["Strict PA"][s] * d, rest[s] * d)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if circular is None:
        return out.sort_values("auc_topq_vs_deg", ascending=False).reset_index(
            drop=True
        )
    flag = out["signal"].isin(set(circular))
    out["topq_vs_deg_circular"] = flag
    out.loc[flag, "auc_topq_vs_deg"] = np.nan
    best = out[["auc_topq_vs_deg", "auc_pa_vs_rest"]].max(axis=1)
    return (
        out.assign(_best=best)
        .sort_values(["_best", "signal"], ascending=[False, True], na_position="last")
        .drop(columns="_best")
        .reset_index(drop=True)
    )


def gate_trial_table(
    cells: pd.DataFrame, gates: Mapping[str, pd.Series]
) -> pd.DataFrame:
    """How many cells of each group a candidate reference-eligibility gate keeps.

    Args:
        cells: Per-cell table with the group columns plus ``ref_ranked`` (optional).
        gates: Gate label -> boolean Series aligned to ``cells`` (True = eligible).

    Returns:
        DataFrame with one row per gate and "kept/total" strings for the reference
        pool, its top quartile, the ranked references, strict-PA cells and the degraded
        pool (degraded cells that would be reference-eligible), plus the overlap left.
    """

    def flag(col: str) -> pd.Series:
        if col not in cells:
            return pd.Series(False, index=cells.index)
        return cells[col].fillna(False).astype(bool)

    groups = {
        "ref_pool_kept": flag("in_ref_pool"),
        "ref_topq_kept": flag("ref_topq"),
        "ref_ranked_kept": flag("ref_ranked"),
        "strict_pa_kept": flag("strict_pa"),
        "deg_pool_ref_eligible": flag("in_deg_pool"),
    }
    rows = []
    for name, gate in gates.items():
        g = pd.Series(gate, index=cells.index).fillna(False).astype(bool)
        row: dict[str, object] = {"gate": name}
        for key, m in groups.items():
            row[key] = f"{int((g & m).sum())}/{int(m.sum())}"
        row["overlap_left"] = int(
            (g & groups["ref_pool_kept"] & groups["deg_pool_ref_eligible"]).sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_condition_signals(
    cells: pd.DataFrame,
    title: str,
    signals: Sequence[str] | None = None,
    ncols: int = 4,
    figsize: tuple | None = None,
    seed: int = 0,
    circular: Collection[str] | None = None,
    sort_by_auc: bool = False,
):
    """Box-and-strip plots of condition signals by site-selection group.

    One panel per signal, with boxes (no fliers) and jittered points for the strict-PA
    cells, the reference top quartile, the reference pool and the degraded pool. Each
    panel title gives the oriented AUC of the reference top quartile vs the degraded
    pool (and vs-rest AUC for strict PAs when present). Circular AUCs (``circular``)
    are left out of the titles.

    Args:
        cells: Per-cell table with the signal and group columns (see condition_groups).
        title: Figure title (group sizes are appended).
        signals: Signal columns to plot (default: CONDITION_SIGNALS present in ``cells``).
        ncols: Panels per row (default 4).
        figsize: Figure size; default scales with the panel grid.
        seed: Seed for the point jitter, so figures are reproducible.
        circular: Signals whose top-quartile vs degraded AUC is circular (see
            ``circular_condition_signals``); that AUC is not shown for them.
        sort_by_auc: Order panels by the highest AUC shown (default False: the
            ``signals`` order).

    Returns:
        tuple of (matplotlib.figure.Figure, numpy array of Axes).

    Raises:
        ValueError: If none of the requested signals are columns of ``cells``.
    """
    names = [s for s in (signals or CONDITION_SIGNALS) if s in cells]
    if not names:
        raise ValueError("None of the requested signals are columns of cells.")
    groups = condition_groups(cells)
    sep_table = condition_separation_table(
        cells, {s: CONDITION_SIGNALS.get(s, 1) for s in names}, circular=circular
    )
    if sort_by_auc:
        order = list(sep_table["signal"])
        names = sorted(names, key=order.index)
    sep = sep_table.set_index("signal")
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize or (4 * ncols, 3.3 * nrows), squeeze=False
    )
    rng = np.random.default_rng(seed)
    labels = list(groups)
    for ax, s in zip(axes.flat, names):
        data = [groups[g][s].dropna() for g in labels]
        ax.boxplot(data, showfliers=False)
        for i, d in enumerate(data, 1):
            ax.scatter(rng.normal(i, 0.06, len(d)), d, s=3, alpha=0.3, color="k")
        ax.set_xticks(
            range(1, len(labels) + 1),
            [lab.replace(" ", "\n") for lab in labels],
            fontsize=7,
        )
        a = sep.loc[s, "auc_topq_vs_deg"]
        pa = sep.loc[s, "auc_pa_vs_rest"]
        if circular is None:
            sub = f"AUC top-q vs deg {a:.2f}" + (
                "" if np.isnan(pa) else f" | PA vs rest {pa:.2f}"
            )
        else:
            parts = []
            if not np.isnan(a):
                parts.append(f"top-q vs deg {a:.2f}")
            if not np.isnan(pa):
                parts.append(f"PA vs rest {pa:.2f}")
            sub = "AUC " + " | ".join(parts) if parts else "no independent AUC"
        ax.set_title(f"{s}\n{sub}", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    for ax in list(axes.flat)[len(names) :]:
        ax.set_visible(False)
    sizes = ", ".join(f"{k} {len(v)}" for k, v in groups.items())
    fig.suptitle(f"{title} (n: {sizes})")
    fig.tight_layout()
    return fig, axes


def plot_pool_condition(
    reference_pool: pd.DataFrame,
    degraded_pool: pd.DataFrame,
    title: str,
    *,
    degradation_type: str = "generic",
    reference_weights: object | None = None,
    strict_pa_min: float = 0.5,
    top_quantile: float = 0.75,
    signals: Sequence[str] | None = None,
    ncols: int = 4,
    figsize: tuple | None = None,
    seed: int = 0,
):
    """Condition diagnostic of a site-selection run, from its two pools.

    Builds the per-cell table (``condition_cells_from_pools``), drops the circular
    top-quartile vs degraded AUCs (``circular_condition_signals``), and plots every
    signal with panels ordered by the highest AUC left. No Earth Engine; works for
    any project's ``result.pool`` frames or exported ``<kind>_pool.gpkg`` files.

    Args:
        reference_pool: Reference run pool (``cell_id``, signals, ``pool_status``,
            ``intactness``, ``pa_strict_share``).
        degraded_pool: Degraded run pool.
        title: Figure title (group sizes are appended).
        degradation_type: The degraded run's ``degradation_type`` (default
            ``generic``); sets the degraded-side circular signals.
        reference_weights: The reference run's ``ReferenceWeights`` (default:
            ``ReferenceWeights()``); sets the reference-side circular signals.
        strict_pa_min: Minimum strict protected share for the strict-PA group
            (default 0.5).
        top_quantile: Intactness quantile that starts the reference top group
            (default 0.75).
        signals: Signals to plot (default: CONDITION_SIGNALS present in the pools).
        ncols: Panels per row (default 4).
        figsize: Figure size; default scales with the panel grid.
        seed: Seed for the point jitter.

    Returns:
        tuple of (matplotlib.figure.Figure, numpy array of Axes, separation
        DataFrame from ``condition_separation_table`` with circular AUCs as NaN).
    """
    cells = condition_cells_from_pools(
        reference_pool,
        degraded_pool,
        strict_pa_min=strict_pa_min,
        top_quantile=top_quantile,
    )
    circular = circular_condition_signals(reference_weights, degradation_type)
    names = [s for s in (signals or CONDITION_SIGNALS) if s in cells]
    fig, axes = plot_condition_signals(
        cells,
        title,
        signals=names,
        ncols=ncols,
        figsize=figsize,
        seed=seed,
        circular=circular,
        sort_by_auc=True,
    )
    table = condition_separation_table(
        cells, {s: CONDITION_SIGNALS.get(s, 1) for s in names}, circular=circular
    )
    shown = sorted(circular & set(names))
    if shown:
        fig.text(
            0.01,
            0.005,
            "Top-quartile vs degraded AUC omitted as circular (a selection criterion of "
            f"either group) for: {', '.join(shown)}. Strict PA: pa_strict_share >= "
            f"{strict_pa_min:g}.",
            fontsize=7,
        )
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.08))
    return fig, axes, table
