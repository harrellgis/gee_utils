import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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
    """Plot a single monthly timeseries metric as a lineplot with a configurable date tick interval.

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
    metric_col: list[str] | tuple[str, ...] = None,
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
    metric_col: list[str] | tuple[str, ...] = None,
    figsize: tuple = (14, 6),
    month_interval: int = 6,
):
    """Plot multiple monthly timeseries metrics overlaid on the same axis with a configurable date tick interval.

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
