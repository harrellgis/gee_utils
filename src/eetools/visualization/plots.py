from collections.abc import Mapping, Sequence

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


def condition_separation_table(
    cells: pd.DataFrame, signals: Mapping[str, int] | None = None
) -> pd.DataFrame:
    """Medians per group and separation AUCs for each condition signal.

    AUCs are oriented so that higher always means "more intact" (signals with
    direction -1 are negated):

    - ``auc_topq_vs_deg``: reference top quartile vs degraded pool. Circular for signals
      already in the intactness score.
    - ``auc_pa_vs_rest``: strict protected-area cells vs the rest of the reference pool,
      an independent label where strict PAs exist.

    Args:
        cells: Per-cell table with the signal and group columns (see condition_groups).
        signals: Signal -> direction (default CONDITION_SIGNALS); absent columns skipped.

    Returns:
        DataFrame with one row per signal, sorted by ``auc_topq_vs_deg`` (descending).
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
    return out.sort_values("auc_topq_vs_deg", ascending=False).reset_index(drop=True)


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
):
    """Box-and-strip plots of condition signals by site-selection group.

    One panel per signal, with boxes (no fliers) and jittered points for the strict-PA
    cells, the reference top quartile, the reference pool and the degraded pool. Each
    panel title gives the oriented AUC of the reference top quartile vs the degraded
    pool (and vs-rest AUC for strict PAs when present).

    Args:
        cells: Per-cell table with the signal and group columns (see condition_groups).
        title: Figure title (group sizes are appended).
        signals: Signal columns to plot (default: CONDITION_SIGNALS present in ``cells``).
        ncols: Panels per row (default 4).
        figsize: Figure size; default scales with the panel grid.
        seed: Seed for the point jitter, so figures are reproducible.

    Returns:
        tuple of (matplotlib.figure.Figure, numpy array of Axes).
    """
    names = [s for s in (signals or CONDITION_SIGNALS) if s in cells]
    if not names:
        raise ValueError("None of the requested signals are columns of cells.")
    groups = condition_groups(cells)
    sep = condition_separation_table(
        cells, {s: CONDITION_SIGNALS.get(s, 1) for s in names}
    ).set_index("signal")
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
        sub = f"AUC top-q vs deg {a:.2f}" + (
            "" if np.isnan(pa) else f" | PA vs rest {pa:.2f}"
        )
        ax.set_title(f"{s}\n{sub}", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    for ax in list(axes.flat)[len(names) :]:
        ax.set_visible(False)
    sizes = ", ".join(f"{k} {len(v)}" for k, v in groups.items())
    fig.suptitle(f"{title} (n: {sizes})")
    fig.tight_layout()
    return fig, axes
