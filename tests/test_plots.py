"""Tests for eetools.visualization.plots (matplotlib on the Agg backend)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from eetools.site_selection.types import ReferenceWeights
from eetools.visualization.plots import (
    CONDITION_SIGNALS,
    add_year_markers,
    add_year_shading,
    circular_condition_signals,
    condition_cells_from_pools,
    condition_groups,
    condition_separation_table,
    gate_trial_table,
    plot_condition_signals,
    plot_multi_metric_annually,
    plot_multi_metric_monthly,
    plot_pool_condition,
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


@pytest.fixture
def pools():
    """Reference pool (6 eligible + 1 gated) and degraded pool (3 eligible, one shared)."""
    ref = pd.DataFrame(
        {
            "cell_id": [f"r{i}" for i in range(6)] + ["shared", "gated"],
            "pool_status": ["eligible"] * 7 + ["gated_out"],
            "intactness": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.99],
            "pa_strict_share": [1.0, 0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 1.0],
            "natural_frac": [1.0, 0.95, 0.9, 0.9, 0.85, 0.8, 0.7, 1.0],
            "dw_nat_share_all": [1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.6, 1.0],
            "dw_crop_share": [0.0] * 8,
            "ndvi_state": [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.9],
            "ghm": [0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.1],
        }
    )
    deg = pd.DataFrame(
        {
            "cell_id": ["shared", "d1", "d2", "d3"],
            "pool_status": ["eligible", "eligible", "eligible", "gated_out"],
            "pa_strict_share": [0.0] * 4,
            "natural_frac": [0.1, 0.5, 0.4, 0.0],
            "dw_nat_share_all": [0.2, 0.5, 0.3, 0.0],
            "dw_crop_share": [0.0, 0.4, 0.5, 1.0],
            "ndvi_state": [0.1, 0.4, 0.3, 0.2],
            "ghm": [0.6, 0.5, 0.7, 0.9],
        }
    )
    return ref, deg


def test_condition_cells_from_pools_groups_and_renames(pools):
    ref, deg = pools
    cells = condition_cells_from_pools(ref, deg)
    assert list(cells.index) == sorted(cells.index)
    assert "gated" not in cells.index and "d3" not in cells.index
    assert {"dw_natural_persist", "dw_crop_recent"} <= set(cells.columns)
    assert "dw_nat_share_all" not in cells.columns
    assert int(cells["in_ref_pool"].sum()) == 7
    assert int(cells["in_deg_pool"].sum()) == 3
    # A cell in both pools keeps the reference values.
    assert cells.loc["shared", "natural_frac"] == pytest.approx(0.7)
    assert cells.loc["shared", "in_ref_pool"] and cells.loc["shared", "in_deg_pool"]
    # Top quartile of the 7 eligible intactness values (q75 = 0.75): r0, r1.
    assert set(cells.index[cells["ref_topq"]]) == {"r0", "r1"}
    assert set(cells.index[cells["strict_pa"]]) == {"r0", "r1"}
    assert set(
        cells.index[
            condition_cells_from_pools(ref, deg, strict_pa_min=0.3)["strict_pa"]
        ]
    ) == {"r0", "r1", "r2"}


def test_condition_cells_from_pools_without_pool_status_and_errors(pools):
    ref, deg = pools
    cells = condition_cells_from_pools(ref.drop(columns="pool_status"), deg)
    assert "gated" in cells.index
    with pytest.raises(ValueError, match="cell_id"):
        condition_cells_from_pools(ref.drop(columns="cell_id"), deg)
    with pytest.raises(ValueError, match="intactness"):
        condition_cells_from_pools(ref.drop(columns="intactness"), deg)


def test_circular_condition_signals_by_weights_and_pathway():
    default = circular_condition_signals()
    # Reference side (default weights): natural cover, persistence, core, gHM.
    assert {"natural_frac", "dw_natural_persist", "core_frac", "ghm"} <= default
    # Degraded side, every pathway: NDVI / NDMI state.
    assert {"ndvi_state", "ndmi_state"} <= default
    # Conversion pathway: crop share via the conversion test.
    assert "dw_crop_recent" in default
    w = ReferenceWeights(
        naturalness=0.0,
        natural_cover=0.25,
        core=0.15,
        persistence=0.05,
        burn_persist=0.05,
        natural_persist=0.25,
        low_loss=0.15,
        protection=0.10,
    )
    natural = circular_condition_signals(w, "natural_degradation")
    assert "ghm" not in natural
    assert {"dw_crop_recent", "ndvi_state"} <= natural


def test_separation_table_drops_circular_aucs(condition_cells):
    t = condition_separation_table(condition_cells, circular={"ghm"}).set_index(
        "signal"
    )
    assert np.isnan(t.loc["ghm", "auc_topq_vs_deg"])
    assert bool(t.loc["ghm", "topq_vs_deg_circular"])
    assert not bool(t.loc["natural_frac", "topq_vs_deg_circular"])
    assert t.loc["natural_frac", "auc_topq_vs_deg"] == 1.0
    assert t.loc["ghm", "auc_pa_vs_rest"] == 1.0
    # Without circular the table is unchanged (no flag column).
    assert "topq_vs_deg_circular" not in condition_separation_table(condition_cells)


def test_plot_condition_signals_hides_circular_aucs(condition_cells):
    fig, axes = plot_condition_signals(
        condition_cells, "test", circular={"ghm"}, sort_by_auc=True, ncols=2
    )
    titles = {ax.get_title().split("\n")[0]: ax.get_title() for ax in axes.flat}
    assert "top-q" not in titles["ghm"] and "PA vs rest" in titles["ghm"]
    assert "top-q vs deg 1.00" in titles["natural_frac"]
    plt.close(fig)


def test_plot_pool_condition(pools):
    ref, deg = pools
    fig, axes, table = plot_pool_condition(ref, deg, "pools", ncols=3)
    assert set(table["signal"]) == {
        "natural_frac",
        "dw_natural_persist",
        "dw_crop_recent",
        "ghm",
        "ndvi_state",
    }
    # Default weights + generic pathway: every top-q vs degraded AUC is circular.
    assert table["auc_topq_vs_deg"].isna().all()
    assert table["topq_vs_deg_circular"].all()
    visible = [ax for ax in axes.flat if ax.get_visible()]
    assert len(visible) == 5
    assert all("top-q" not in ax.get_title() for ax in visible)
    # Panels follow the table order (highest strict-PA AUC first).
    assert [ax.get_title().split("\n")[0] for ax in visible] == list(table["signal"])
    assert any("circular" in t.get_text() for t in fig.texts)
    plt.close(fig)
