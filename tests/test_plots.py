"""Tests for eetools.visualization.plots (matplotlib on the Agg backend)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from eetools.visualization.plots import (
    CONDITION_SIGNALS,
    add_year_markers,
    add_year_shading,
    condition_groups,
    condition_separation_table,
    gate_trial_table,
    plot_condition_signals,
    plot_multi_metric_annually,
    plot_multi_metric_monthly,
    plot_single_metric_annually,
    plot_single_metric_monthly,
    separation_auc,
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


@pytest.fixture
def condition_cells():
    """12 cells: 4 strict-PA reference cells, 4 other reference cells, 4 degraded cells."""
    n = 12
    idx = pd.Index([f"c{i}" for i in range(n)], name="cell_id")
    intact = np.r_[np.full(4, 0.95), np.full(4, 0.7), np.full(4, 0.2)]
    return pd.DataFrame(
        {
            "natural_frac": intact,
            "dw_natural_persist": intact,
            "dw_crop_recent": 1 - intact,
            "ghm": 1 - intact,
            "strict_pa": [True] * 4 + [False] * 8,
            "ref_topq": [True] * 4 + [False] * 8,
            "in_ref_pool": [True] * 8 + [False] * 4,
            "in_deg_pool": [False] * 8 + [True] * 4,
            "ref_ranked": [True, True] + [False] * 10,
        },
        index=idx,
    )


def test_separation_auc_bounds_ties_and_empty():
    assert separation_auc(pd.Series([3, 4]), pd.Series([1, 2])) == 1.0
    assert separation_auc(pd.Series([1, 2]), pd.Series([3, 4])) == 0.0
    assert separation_auc(pd.Series([1, 1]), pd.Series([1, 1])) == 0.5
    assert np.isnan(separation_auc(pd.Series([], dtype=float), pd.Series([1.0])))
    assert separation_auc(pd.Series([2, np.nan]), pd.Series([1])) == 1.0


def test_condition_signals_exclude_retired_datasets():
    for retired in (
        "sbtn_natural",
        "bii",
        "dry_ndvi",
        "dry_wet_ratio",
        "dw_label_stable",
    ):
        assert retired not in CONDITION_SIGNALS


def test_condition_groups_and_missing_column(condition_cells):
    g = condition_groups(condition_cells)
    assert {k: len(v) for k, v in g.items()} == {
        "Strict PA": 4,
        "Ref top quartile": 4,
        "Ref pool": 8,
        "Deg pool": 4,
    }
    g2 = condition_groups(condition_cells.drop(columns="strict_pa"))
    assert len(g2["Strict PA"]) == 0


def test_condition_separation_table_orients_negative_signals(condition_cells):
    t = condition_separation_table(condition_cells).set_index("signal")
    # Absent signals are skipped; present ones all separate perfectly once oriented.
    assert set(t.index) == {
        "natural_frac",
        "dw_natural_persist",
        "dw_crop_recent",
        "ghm",
    }
    assert (t["auc_topq_vs_deg"] == 1.0).all()
    assert (t["auc_pa_vs_rest"] == 1.0).all()
    assert t.loc["ghm", "median_deg_pool"] == pytest.approx(0.8)


def test_gate_trial_table_counts(condition_cells):
    gate = condition_cells["dw_natural_persist"] >= 0.8
    t = gate_trial_table(condition_cells, {"persist": gate}).iloc[0]
    assert t["ref_pool_kept"] == "4/8"
    assert t["strict_pa_kept"] == "4/4"
    assert t["ref_ranked_kept"] == "2/2"
    assert t["deg_pool_ref_eligible"] == "0/4"
    assert t["overlap_left"] == 0


def test_plot_condition_signals_panels_and_hidden_axes(condition_cells):
    fig, axes = plot_condition_signals(condition_cells, title="Test", ncols=3)
    visible = [ax for ax in axes.flat if ax.get_visible()]
    assert len(visible) == 4
    assert axes.shape == (2, 3)
    assert "AUC top-q vs deg 1.00" in visible[0].get_title()
    assert fig._suptitle.get_text().startswith("Test (n: Strict PA 4")
    plt.close(fig)


def test_plot_condition_signals_requires_known_signal(condition_cells):
    with pytest.raises(ValueError):
        plot_condition_signals(condition_cells, title="x", signals=["nope"])
