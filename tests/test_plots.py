"""Tests for eetools.visualization.plots (matplotlib on the Agg backend)."""

import matplotlib.pyplot as plt
import pytest

from eetools.visualization.plots import (
    add_year_markers,
    add_year_shading,
    plot_multi_metric_annually,
    plot_multi_metric_monthly,
    plot_single_metric_annually,
    plot_single_metric_monthly,
)


def test_plot_single_metric_annually_returns_fig_ax(monthly_timeseries_df):
    fig, ax = plot_single_metric_annually(monthly_timeseries_df, title="NDVI")

    assert ax.get_title() == "NDVI"
    assert ax.get_ylabel() == "metric"
    assert len(ax.lines) == 1


def test_plot_single_metric_does_not_mutate_input(monthly_timeseries_df):
    before = monthly_timeseries_df["date"].tolist()
    plot_single_metric_annually(monthly_timeseries_df, title="NDVI")
    # The function copies internally; the caller's date column stays as strings.
    assert monthly_timeseries_df["date"].tolist() == before


def test_plot_single_metric_monthly_custom_columns(monthly_timeseries_df):
    df = monthly_timeseries_df.rename(columns={"metric": "ndvi"})
    fig, ax = plot_single_metric_monthly(
        df, title="Monthly", metric_col="ndvi", month_interval=3
    )
    assert ax.get_ylabel() == "ndvi"
    assert len(ax.lines) == 1


def test_plot_multi_metric_annually_one_line_per_metric(multi_metric_df):
    metrics = ["NDVI", "EVI", "SAVI"]
    fig, ax = plot_multi_metric_annually(
        multi_metric_df, title="Indices", metric_col=metrics
    )
    assert len(ax.lines) == len(metrics)
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels == metrics


def test_plot_multi_metric_monthly_one_line_per_metric(multi_metric_df):
    metrics = ["NDVI", "EVI"]
    fig, ax = plot_multi_metric_monthly(
        multi_metric_df, title="Indices", metric_col=metrics
    )
    assert len(ax.lines) == len(metrics)


@pytest.mark.parametrize(
    "func", [plot_multi_metric_annually, plot_multi_metric_monthly]
)
def test_multi_metric_requires_metric_col(func, multi_metric_df):
    with pytest.raises(ValueError, match="metric_col"):
        func(multi_metric_df, title="x")


def test_add_year_shading_adds_two_spans():
    fig, ax = plt.subplots()
    n_patches_before = len(ax.patches)

    returned = add_year_shading(ax)

    assert returned is ax
    assert len(ax.patches) == n_patches_before + 2


def test_add_year_markers_adds_lines_and_text():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])  # establish y-limits for the text placement
    n_lines_before = len(ax.lines)

    returned = add_year_markers(ax)

    assert returned is ax
    assert len(ax.lines) == n_lines_before + 2
    assert len(ax.texts) == 2
