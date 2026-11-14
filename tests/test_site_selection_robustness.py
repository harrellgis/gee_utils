"""Regression tests for the v0.6.0 robustness changes (2026-11-09 design review).

Land-use history classes and the k-years tolerance, the conversion-only pathway test,
gate-then-thin order, absolute stop rule with management units, type-aware degraded
weights, split-and-retry batching, the reference gate fallback and the separate
candidate / pool exports. Pure pandas, no Earth Engine.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from eetools.site_selection.scoring import (
    conversion_pathway,
    gate_then_thin,
    landuse_history,
    near_references,
    rank_candidates,
    stop_rule_met,
)
from eetools.site_selection.types import (
    DegradedWeights,
    EnvelopeTolerances,
    ReferenceWeights,
    default_degraded_weights,
)

END = 2025


def _landuse(**cols):
    """One row per cell with dw_landuse columns (2019-2025)."""
    n = len(next(iter(cols.values())))
    base = {
        "dw_valid_share": [1.0] * n,
        "dw_nat_share_all": [0.0] * n,
        "dw_nat_share_all_but_one": [0.0] * n,
        "dw_mod_share_persist": [0.0] * n,
        "dw_built_share": [0.0] * n,
        "dw_conversion_year": [np.nan] * n,
        **{f"dw_converted_by_{y}": [0.0] * n for y in range(2020, END + 1)},
    }
    base.update(cols)
    return pd.DataFrame(base, index=[f"c{i}" for i in range(n)])


# --------------------------------------------------------------------------- #
# Land-use history                                                             #
# --------------------------------------------------------------------------- #
def test_landuse_history_classes_and_priority():
    df = _landuse(
        dw_valid_share=[1, 1, 1, 1, 1, 0.2],
        dw_nat_share_all=[0.9, 0.0, 0.0, 0.3, 0.9, 0.9],
        dw_mod_share_persist=[0.0, 0.7, 0.1, 0.1, 0.0, 0.0],
        dw_built_share=[0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
        **{
            f"dw_converted_by_{y}": [0, 0, 0.6, 0.1, 0, 0] for y in range(2020, END + 1)
        },
    )
    h = landuse_history(df, end_year=END, min_natural_persist=0.8, max_built_share=0.05)
    assert list(h["landuse_class"]) == [
        "persistent_natural",
        "persistent_modified",
        "converted",
        "mixed",
        "persistent_natural",  # natural beats built (few % homesteads in a big cell)
        "no_dw",  # too few observations beats everything
    ]


def test_landuse_history_built_class_for_non_natural_cells():
    df = _landuse(dw_nat_share_all=[0.3, 0.3], dw_built_share=[0.2, 0.04])
    h = landuse_history(df, end_year=END, max_built_share=0.05)
    assert list(h["landuse_class"]) == ["built", "mixed"]


def test_landuse_history_k_years_tolerance():
    """'all' vs 'all_but_one' is a pandas choice on stored shares, never a re-reduction."""
    df = _landuse(dw_nat_share_all=[0.6], dw_nat_share_all_but_one=[0.9])
    strict = landuse_history(df, end_year=END, natural_persist_years="all")
    loose = landuse_history(df, end_year=END, natural_persist_years="all_but_one")
    assert strict["landuse_class"].iloc[0] == "mixed"
    assert loose["landuse_class"].iloc[0] == "persistent_natural"
    with pytest.raises(ValueError):
        landuse_history(df, end_year=END, natural_persist_years="most")


def test_landuse_history_modified_by_baseline_uses_that_year():
    conv = {
        f"dw_converted_by_{y}": [0.0 if y < 2023 else 0.6] for y in range(2020, END + 1)
    }
    df = _landuse(dw_mod_share_persist=[0.1], **conv)
    early = landuse_history(df, end_year=END, baseline_end_year=2021)
    late = landuse_history(df, end_year=END, baseline_end_year=2024)
    assert early["modified_by_baseline"].iloc[0] == pytest.approx(0.1)
    assert late["modified_by_baseline"].iloc[0] == pytest.approx(0.7)
    # baseline before the history starts: only the persistent share counts
    pre = landuse_history(df, end_year=END, baseline_end_year=2017)
    assert pre["modified_by_baseline"].iloc[0] == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# Pathway test: conversion pathways only                                       #
# --------------------------------------------------------------------------- #
def test_conversion_pathway_only_for_conversion_types():
    h = landuse_history(_landuse(dw_nat_share_all=[1.0, 1.0]), end_year=END)
    # rangeland / overgrazing stay inside natural cover: no land-use test, so a
    # degraded pool of persistently natural grass/shrub cells is not emptied
    assert conversion_pathway(h, "rangeland") is None
    assert conversion_pathway(h, "overgrazing") is None
    assert not conversion_pathway(h, "generic").any()


def test_conversion_pathway_thresholds_and_deforestation_loss():
    conv = {f"dw_converted_by_{y}": [0.6, 0.6, 0.0] for y in range(2020, END + 1)}
    df = _landuse(dw_mod_share_persist=[0.0, 0.0, 0.0], **conv)
    df.loc["c1", [f"dw_converted_by_{y}" for y in range(2020, 2024)]] = 0.0
    h = landuse_history(df, end_year=END, baseline_end_year=2022)
    # c0 converted by 2020, c1 only from 2024 (after the baseline window)
    assert list(conversion_pathway(h, "cropland")) == [True, False, False]
    loss = pd.Series([0.0, 0.0, 0.2], index=h.index)
    assert list(conversion_pathway(h, "deforestation", forest_loss_frac=loss)) == [
        True,
        False,
        True,
    ]


def test_near_references_drops_only_close_cells():
    pool = pd.DataFrame({"cx": [0.0, 400.0, 5000.0], "cy": [0.0, 0.0, 0.0]})
    refs = pd.DataFrame({"cx": [0.0], "cy": [0.0]})
    assert list(near_references(pool, refs, 500.0)) == [True, True, False]
    assert not near_references(pool, refs.iloc[0:0], 500.0).any()


# --------------------------------------------------------------------------- #
# Gate, then thin                                                              #
# --------------------------------------------------------------------------- #
def test_gate_then_thin_keeps_neighbour_of_dropped_cell():
    pool = pd.DataFrame(
        {"rank_score": [0.9, 0.8, 0.1], "cx": [0.0, 100.0, 5000.0], "cy": [0.0] * 3},
        index=["built", "neighbour", "far"],
    )
    eligible = pd.Series([False, True, True], index=pool.index)
    out = gate_then_thin(pool, eligible, "rank_score", 500.0)
    assert list(out.index) == ["neighbour", "far"]
    # the v0.5.0 order (thin, then filter) loses the neighbour for good
    thin_first = rank_candidates(pool, "rank_score", 500.0)
    assert "neighbour" not in thin_first.loc[eligible.reindex(thin_first.index)].index
    empty = gate_then_thin(pool, pd.Series(False, index=pool.index), "rank_score", 1.0)
    assert empty.empty and "rank" in empty.columns


# --------------------------------------------------------------------------- #
# Stop rule                                                                    #
# --------------------------------------------------------------------------- #
def test_stop_rule_continues_when_a_unit_is_missing():
    units = {"Fb": {"Fb"}, "B": {"B"}}
    assert not stop_rule_met(5, 5, units, {"Fb"})
    assert stop_rule_met(5, 5, units, {"Fb", "B"})
    assert not stop_rule_met(2, 5, units, {"Fb", "B"})
    # count-only mode (saves Tier 3 compute)
    assert stop_rule_met(5, 5, units, {"Fb"}, require_units=False)
    assert stop_rule_met(3, 3)


def test_tolerances_new_fields_validate():
    t = EnvelopeTolerances()
    assert t.min_natural_persist == 0.9 and t.natural_persist_years == "all"
    assert t.unit_coverage_stop
    with pytest.raises(ValueError):
        EnvelopeTolerances(natural_persist_years="most")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        EnvelopeTolerances(min_natural_persist=1.5)


# --------------------------------------------------------------------------- #
# Weights                                                                      #
# --------------------------------------------------------------------------- #
def test_default_degraded_weights_drop_pressure_except_settlement():
    assert default_degraded_weights("settlement") == DegradedWeights()
    w = default_degraded_weights("generic")
    assert w.pressure == 0.0
    total = sum(getattr(w, f) for f in w.__dataclass_fields__)
    assert total == pytest.approx(1.0)
    # remaining terms keep their proportions
    assert w.perf_deficit / w.moist_deficit == pytest.approx(1.5)


def test_reference_weights_replace_bii_with_natural_persist():
    w = ReferenceWeights()
    assert w.natural_persist == pytest.approx(0.25)
    assert not hasattr(w, "bii")


# --------------------------------------------------------------------------- #
# Batching: spatial order and split-and-retry                                  #
# --------------------------------------------------------------------------- #
def test_reduce_cells_sorts_spatially_and_splits_on_object_too_large(monkeypatch):
    from eetools.site_selection import predictors as P

    cells = pd.DataFrame(
        {
            "ix": [1000, 190, 191, 5, 6, 7],
            "iy": [9000, 9000, 9000, 1, 1, 1],
            "size": [1000.0] * 6,
        },
        index=pd.Index(
            ["1000_1000_9000", "1000_190_9000", "1000_191_9000", "a", "b", "c"],
            name="cell_id",
        ),
    )
    requests: list[list[str]] = []

    class FakeImage:
        def reduceRegions(self, collection, **kw):  # noqa: N802 - EE API name
            return collection

    def fake_fc(chunk, crs):
        return chunk

    def fake_pull(chunk):
        requests.append(list(chunk.index))
        if len(chunk) > 2:
            raise RuntimeError("Object too large (156323440 bytes).")
        return pd.DataFrame({"cell_id": chunk.index, "mean": chunk["ix"].astype(float)})

    monkeypatch.setattr(P, "cells_feature_collection", fake_fc)
    monkeypatch.setattr(P, "_pull", fake_pull)
    df = P.reduce_cells(
        FakeImage(),
        cells,
        bands=["v"],
        stats=None,
        reducer=None,
        scale=30,
        crs="EPSG:32737",
        batch_size=6,
    )
    # first attempt is the whole batch in numeric (ix, iy) order, not cell_id order
    assert requests[0] == [
        "a",
        "b",
        "c",
        "1000_190_9000",
        "1000_191_9000",
        "1000_1000_9000",
    ]
    # 6 fails -> 3 + 3, each 3 fails -> 1 + 2 (served)
    assert sorted(len(r) for r in requests) == [1, 1, 2, 2, 3, 3, 6]
    assert list(df.index) == list(cells.index)  # caller's order restored
    assert df.loc["1000_1000_9000", "v"] == 1000.0

    def other_error(chunk):
        raise RuntimeError("Computation timed out.")

    monkeypatch.setattr(P, "_pull", other_error)
    with pytest.raises(RuntimeError, match="timed out"):
        P.reduce_cells(
            FakeImage(),
            cells,
            bands=["v"],
            stats=None,
            reducer=None,
            scale=30,
            crs="EPSG:32737",
        )


# --------------------------------------------------------------------------- #
# Reference gate fallback                                                      #
# --------------------------------------------------------------------------- #
def _ref_cells(ids, xs, natural, reason=None):
    n = len(ids)
    return pd.DataFrame(
        {
            "cx": xs,
            "cy": [0.0] * n,
            "ghm": [0.2] * n,
            "natural_frac": natural,
            "core_frac": natural,
            "ndvi_state": [0.6] * n,
            "ndmi_state": [0.1] * n,
            "ndvi_trend": [0.0] * n,
            "nbr_trend": [0.0] * n,
            "natural_persist": natural,
            "pa_overlap": [0.0] * n,
            "mess": [10.0] * n,
            "gate_reason": [reason] * n,
        },
        index=ids,
    )


def test_reference_gate_fallback_fills_with_flagged_gated_out_cells():
    from eetools.site_selection.workflows import _reference_gate_fallback

    pool = _ref_cells(["e1"], [0.0], [0.95])
    pool["rank_score"] = 0.9
    thinned = pool.assign(rank=1, gate_fallback=False)
    gated = pd.concat(
        [
            _ref_cells(["g_near"], [100.0], [0.7], "natural_persist"),
            _ref_cells(["g1", "g2"], [5000.0, 9000.0], [0.6, 0.5], "natural_persist"),
            _ref_cells(["g_mess"], [20000.0], [0.9], "mess"),
        ]
    )
    pool_all = pd.concat([pool.assign(gate_reason=None), gated])
    rw: list[str] = []
    out = _reference_gate_fallback(
        thinned, pool, pool_all, ReferenceWeights(), False, 500.0, 3, rw
    )
    assert list(out.index) == [
        "e1",
        "g1",
        "g2",
    ]  # g_near thinned, g_mess not a gate fail
    assert list(out["gate_fallback"]) == [False, True, True]
    assert list(out["rank"]) == [1, 2, 3]
    assert (out.loc[["g1", "g2"], "gate_reason"] == "natural_persist").all()
    assert rw and "fallback=True" in rw[0]


# --------------------------------------------------------------------------- #
# Exports                                                                      #
# --------------------------------------------------------------------------- #
def _export_result(site_type="reference"):
    from eetools.site_selection.types import ScreeningTables, SiteSelectionResult

    geom = [box(36.0, -1.0, 36.01, -0.99), box(36.02, -1.0, 36.03, -0.99)]
    idx = pd.Index(["c1", "c2"], name="cell_id")
    pool = gpd.GeoDataFrame(
        {
            "ix": [1, 2],
            "iy": [1, 1],
            "size": [250, 250],
            "cx": [1.0, 2.0],
            "cy": [1.0, 1.0],
            "area_ha": [6.25, 6.25],
            "dist_to_focal_m": [1000.0, 2000.0],
            "dist_to_focal_km": [1.0, 2.0],
            "rung": ["T1 strict"] * 2,
            "gate_land": [True, True],
            "gate_class": [True, True],
            "class_code": [21, 21],
            "national_class": ["Fb", "Fb"],
            "vecea_class": ['{"21": 1.0}'] * 2,
            "class_source": ["vecea"] * 2,
            "wrb_code": [3, 3],
            "wrb_group": ["Andosols"] * 2,
            "requires_confirmation": [True, True],
            "elevation": [2400.0, 2390.0],
            "pred_elevation": [2400.0, 2390.0],
            "elevation_p50": [2401.0, 2391.0],
            "ghm_mean": [0.2, 0.5],
            "ghm": [0.2, 0.5],
            "dw_converted_by_2025": [0.0, 0.3],
            "converted_share": [0.0, 0.3],
            "ndvi_base": [np.nan, np.nan],
            "modified_by_baseline": [0.0, 0.4],
            "pathway_test": ["landuse"] * 2,
            "pool_status": ["eligible", "gated_out"],
            "gate_reason": [None, "natural_persist"],
            "landuse_class": ["persistent_natural", "converted"],
            "mandatory_met": [True, None],
        },
        geometry=geom,
        crs=4326,
        index=idx,
    )
    ranked = pool.iloc[:1].assign(
        rank=1,
        role="primary",
        covers_unit="Fb",
        span_note="lower sand content",
        fallback=False,
        gate_fallback=False,
        mandatory_met_relaxed=True,
        matrix_total=12,
        bio12=950.0,
        dw_bare=0.1,
        dw_bare_focal=0.05,
        intactness=0.8,
        degradation=0.8,
    )
    top = gpd.GeoDataFrame(
        {
            "top_rank": [2, 1],
            "top_score": [0.7, 0.9],
            "intactness_pct": [0.5, 1.0],
            "proposed": [True, False],
            "proposed_rank": [1, np.nan],
            "proposed_role": ["primary", None],
        },
        geometry=geom,
        crs=4326,
        index=idx,
    )
    return SiteSelectionResult(
        site_type=site_type,
        ranked=ranked,
        pool=pool,
        focal_profile=pd.DataFrame(),
        params={"top_candidates_n": 10},
        provenance={},
        selection_matrix=pd.DataFrame({"cell_id": ["c1"], "rank": [1]}),
        layer_register=pd.DataFrame({"layer_or_query": ["x"]}),
        screening=ScreeningTables.empty({}),
        warnings=[],
        top_candidates=top,
    )


def test_pool_export_drops_audit_d_columns_and_folds_shortlist():
    from eetools.site_selection.workflows import pool_export_frame

    res = _export_result()
    out = pool_export_frame(res)
    dropped = {
        "ix", "iy", "size", "cx", "cy", "area_ha", "dist_to_focal_m", "rung",
        "gate_land", "gate_class", "class_code", "vecea_class", "class_source",
        "wrb_code", "requires_confirmation", "pred_elevation", "elevation_p50",
        "ghm_mean", "converted_share", "pathway_test", "ndvi_base",
        "modified_by_baseline", "proposed_rank", "proposed_role",
    }  # fmt: skip
    assert not dropped & set(out.columns)
    for kept in ("national_class", "wrb_group", "elevation", "ghm",
                 "dw_converted_by_2025", "dist_to_focal_km", "pool_status",
                 "gate_reason", "landuse_class", "mandatory_met"):  # fmt: skip
        assert kept in out.columns
    assert out.loc["c1", "rank"] == 1 and pd.isna(out.loc["c2", "rank"])
    assert out.loc["c1", "role"] == "primary"
    assert list(out["top_rank"]) == [2, 1]
    assert list(out["proposed"]) == [True, False]
    assert set(out["site_role"]) == {"reference"}
    # the in-memory result is untouched
    assert "ghm_mean" in res.pool.columns
    # degraded pools keep the baseline columns
    deg = pool_export_frame(_export_result("degraded"))
    assert {"ndvi_base", "modified_by_baseline"} <= set(deg.columns)


def test_no_top_file_without_top_candidates(tmp_path):
    from dataclasses import replace

    from eetools.site_selection.workflows import (
        export_site_selection,
        top_candidates_export_frame,
    )

    res = replace(_export_result(), top_candidates=None)
    assert top_candidates_export_frame(res).empty
    paths = export_site_selection(res, tmp_path, project="TGBS", site="T")
    assert not any("top" in k for k in paths)


def test_shortlist_table_is_slim_and_keyed_by_cell_id():
    from eetools.site_selection.workflows import shortlist_table

    t = shortlist_table(_export_result())
    assert list(t.columns[:4]) == ["cell_id", "rank", "role", "covers_unit"]
    assert t.loc[0, "cell_id"] == "c1" and t.loc[0, "top_rank"] == 2
    assert not {"bio12", "dw_bare_focal", "geometry", "elevation_p50"} & set(t.columns)
    d = shortlist_table(_export_result("degraded"))
    assert "dw_bare" in d.columns and "intactness" not in d.columns


def test_export_writes_slim_files_without_index(tmp_path):
    from eetools.site_selection.workflows import export_site_selection

    res = _export_result()
    vec = tmp_path / "vectors"
    paths = export_site_selection(
        res,
        tmp_path / "tables",
        project="TGBS",
        site="T",
        run_date="20261114",
        vector_folder=vec,
    )
    assert set(paths) == {
        "reference_pool",
        "reference_top10",
        "reference_ranked",
        "reference_selection_matrix",
        "reference_layer_register",
    }
    assert paths["reference_pool"] == vec / "TGBS_T_reference_pool_20261114_v1.gpkg"
    assert sorted(q.name for q in vec.iterdir()) == sorted(
        [paths["reference_pool"].name, "TGBS_T_reference_top10_20261114_v1.gpkg"]
    )
    top = gpd.read_file(paths["reference_top10"], layer="top_candidates")
    assert list(top.columns[:4]) == ["cell_id", "top_rank", "top_score", "proposed"]
    assert list(top["cell_id"]) == ["c2", "c1"]  # top_rank order
    assert "ghm_mean" not in top.columns and "pred_elevation" not in top.columns
    assert set(top["site_role"]) == {"reference"}
    back = gpd.read_file(paths["reference_pool"], layer="pool")
    assert set(back["pool_status"]) == {"eligible", "gated_out"}
    assert "ghm_mean" not in back.columns and "top_rank" in back.columns
    for product in ("ranked", "selection_matrix", "layer_register"):
        df = pd.read_csv(paths[f"reference_{product}"])
        assert not any(c.startswith("Unnamed") for c in df.columns)
    assert pd.read_csv(paths["reference_ranked"])["cell_id"].tolist() == ["c1"]
    # re-export overwrites instead of appending
    export_site_selection(
        res, tmp_path / "tables", project="TGBS", site="T", run_date="20261114",
        vector_folder=vec,
    )  # fmt: skip
    assert len(gpd.read_file(paths["reference_pool"], layer="pool")) == 2


def test_export_proposed_sites_links_rows_to_cell_ids(tmp_path):
    from eetools.site_selection.workflows import export_proposed_sites

    ref = _export_result()
    deg = _export_result("degraded")
    focal = gpd.GeoDataFrame(geometry=[box(36.1, -1.0, 36.11, -0.99)], crs=4326)
    p = export_proposed_sites(
        focal, [ref, deg], tmp_path, project="TGBS", site="T", run_date="20261114"
    )
    assert p.name == "TGBS_T_proposed_sites_20261114_v1.gpkg"
    back = gpd.read_file(p, layer="sites")
    assert list(back.columns) == [
        "site_role", "rank", "cell_id", "role", "covers_unit", "geometry",
    ]  # fmt: skip
    assert list(back["site_role"]) == ["focal", "reference", "degraded"]
    assert back["cell_id"].isna().iloc[0] and list(back["cell_id"][1:]) == ["c1"] * 2
    with pytest.raises(ValueError, match="CRS"):
        export_proposed_sites(
            gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)]), [], tmp_path
        )


def test_selection_matrix_empty_keeps_summary_columns():
    from eetools.site_selection.scoring import MatrixContext, build_selection_matrix

    ctx = MatrixContext(
        site_type="degraded",
        elevation=1000.0,
        elevation_p5=1000.0,
        elevation_p95=1000.0,
        rainfall=1000.0,
        rainfall_p5=1000.0,
        rainfall_p95=1000.0,
        elevation_tol_m=300.0,
        rainfall_tol_frac=0.25,
        focal_area_ha=50.0,
        is_forest=False,
        class_fallback=False,
        class_mosaic=False,
        substrate_fallback=False,
        substrate_off=False,
        relaxed_criteria=set(),
        focal_class_label="Be",
        focal_substrate_label="Ferralsols",
    )
    m = build_selection_matrix(pd.DataFrame(index=pd.Index([], name="cell_id")), ctx)
    assert m.empty
    assert {
        "mandatory_met",
        "mandatory_met_relaxed",
        "fallback",
        "matrix_total",
    } <= set(m.columns)


def test_pathway_example_only_when_most_modification_is_recent():
    from eetools.site_selection.scoring import pathway_example

    h = pd.DataFrame(
        {
            # mostly persistent cropland + small recent conversion; mostly recent; old
            "modified_share": [1.0, 0.9, 0.8],
            "modified_by_baseline": [0.95, 0.2, 0.8],
        }
    )
    assert list(pathway_example(h)) == [False, True, False]


def test_reference_gate_fallback_accepts_sop_failures_as_last_resort():
    from eetools.site_selection.workflows import _reference_gate_fallback

    pool = _ref_cells(["e1"], [0.0], [0.95])
    pool["rank_score"] = 0.9
    thinned = pool.assign(rank=1, gate_fallback=False)
    gated = _ref_cells(["logged"], [5000.0], [0.9], "sop_integrity")
    pool_all = pd.concat([pool.assign(gate_reason=None), gated])
    out = _reference_gate_fallback(
        thinned, pool, pool_all, ReferenceWeights(), False, 500.0, 2, []
    )
    assert list(out.index) == ["e1", "logged"]
    assert out.loc["logged", "gate_reason"] == "sop_integrity"


# --------------------------------------------------------------------------- #
# Top candidates (0.6.1)                                                       #
# --------------------------------------------------------------------------- #
def test_sort_by_score_accepts_a_score_column_named_key():
    from eetools.site_selection.scoring import sort_by_score

    df = pd.DataFrame({"_key": [0.1, 0.9, 0.5]}, index=["c", "a", "b"])
    out = sort_by_score(df, "_key")
    assert list(out.index) == ["a", "b", "c"]  # by score, not by index
    assert list(out.columns) == ["_key"]


def _top_pool():
    # five eligible cells on a line 1000 m apart, plus one gated-out cell
    return pd.DataFrame(
        {
            "pool_status": ["eligible"] * 5 + ["gated_out"],
            "intactness": [0.9, 0.8, 0.7, 0.6, 0.5, 0.99],
            "env_sim": [0.1, 0.9, 0.8, 0.95, 0.2, 1.0],
            "rank_score": [0.5, 0.75, 0.6, 0.6, 0.3, np.nan],
            "cx": [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            "cy": [0.0] * 6,
        },
        index=pd.Index(["a", "b", "c", "d", "e", "g"], name="cell_id"),
    )


def test_select_top_candidates_weights_condition_and_similarity_equally():
    from eetools.site_selection.scoring import select_top_candidates

    shortlist = pd.DataFrame(
        {"rank": [1], "role": ["primary"]}, index=pd.Index(["a"], name="cell_id")
    )
    top = select_top_candidates(
        _top_pool(), "reference", 500.0, n=10, shortlist=shortlist
    )
    # intactness pct: a 1.0, b 0.8, c 0.6, d 0.4, e 0.2
    # top_score = 0.5 * pct + 0.5 * env_sim: b 0.85, d 0.675, c 0.7, a 0.55, e 0.2
    assert list(top.index) == ["b", "c", "d", "a", "e"]
    assert list(top["top_rank"]) == [1, 2, 3, 4, 5]
    assert "g" not in top.index  # gated-out cells never enter
    assert top["top_score"].is_monotonic_decreasing
    assert list(top.columns[:6]) == [
        "top_rank",
        "top_score",
        "intactness_pct",
        "proposed",
        "proposed_rank",
        "proposed_role",
    ]
    assert bool(top.loc["a", "proposed"]) and top.loc["a", "proposed_rank"] == 1
    assert top.loc["a", "proposed_role"] == "primary"
    assert not top.loc["b", "proposed"]


def test_select_top_candidates_spaces_sites_and_caps_n():
    from eetools.site_selection.scoring import select_top_candidates

    top = select_top_candidates(_top_pool(), "reference", 1500.0, n=2)
    # b leads; c (1000 m away) and a (1000 m) are suppressed; d is next
    assert list(top.index) == ["b", "d"]
    assert not top["proposed"].any()


def test_select_top_candidates_degraded_uses_degradation():
    from eetools.site_selection.scoring import select_top_candidates

    pool = _top_pool().rename(columns={"intactness": "degradation"})
    top = select_top_candidates(pool, "degraded", 500.0, condition_weight=1.0)
    assert list(top.index) == ["a", "b", "c", "d", "e"]
    assert "degradation_pct" in top.columns


def test_select_top_candidates_ties_broken_by_rank_score():
    from eetools.site_selection.scoring import select_top_candidates

    pool = _top_pool().iloc[:5].assign(intactness=0.5, env_sim=0.5)
    top = select_top_candidates(pool, "reference", 500.0)
    assert list(top.index[:2]) == ["b", "c"]  # rank_score 0.75, then 0.6 (c before d)


def test_select_top_candidates_empty_pool():
    from eetools.site_selection.scoring import select_top_candidates

    pool = _top_pool().assign(pool_status="gated_out")
    top = select_top_candidates(pool, "reference", 500.0)
    assert len(top) == 0 and "top_rank" in top.columns


def test_sop_pool_columns_keep_mandatory_met_and_automated_scores():
    from eetools.site_selection.scoring import sop_pool_columns

    matrix = pd.DataFrame(
        {
            "elevation_score": [2, 1],
            "elevation_evidence": ["x", "y"],
            "elevation_confidence": ["H", "H"],
            "integrity_score": [2, 0],
            "field_access_score": [np.nan, np.nan],
            "field_access_evidence": ["for the team"] * 2,
            "mandatory_met": [True, False],
            "mandatory_met_relaxed": [True, False],
            "matrix_total": [4, 1],
        },
        index=pd.Index(["a", "b"], name="cell_id"),
    )
    out = sop_pool_columns(matrix)
    assert list(out.columns) == ["mandatory_met", "elevation_score", "integrity_score"]
    assert list(out["integrity_score"]) == [2, 0]


def test_select_top_candidates_shows_flags_and_sop_status_without_filtering():
    from eetools.site_selection.scoring import select_top_candidates

    pool = _top_pool().assign(
        intactness_flag=["ok", "weak", "ok", "weak", "ok", "ok"],
        mandatory_met=[True, False, True, False, True, np.nan],
        elevation_score=[2, 1, 2, 1, 2, np.nan],
    )
    top = select_top_candidates(pool, "reference", 500.0)
    assert list(top.index) == ["b", "c", "d", "a", "e"]  # ranking unchanged
    assert list(top.columns[6:8]) == ["intactness_flag", "mandatory_met"]
    assert top.loc["b", "intactness_flag"] == "weak"
    assert "elevation_score" in top.columns

    deg = pool.drop(columns="intactness_flag").rename(
        columns={"intactness": "degradation"}
    )
    deg = deg.assign(zero_star_likelihood="high", anchor_check="ok")
    top = select_top_candidates(deg, "degraded", 500.0)
    assert list(top.columns[6:9]) == [
        "zero_star_likelihood",
        "anchor_check",
        "mandatory_met",
    ]


@pytest.mark.parametrize(
    "kw",
    [{"top_candidates_n": 0}, {"top_candidates_condition_weight": 1.5}],
)
def test_workflows_validate_top_candidate_parameters(kw):
    from eetools.site_selection.workflows import (
        identify_degraded_sites,
        identify_reference_sites,
    )

    with pytest.raises(ValueError, match="top_candidates"):
        identify_reference_sites(box(0, 0, 1, 1), **kw)
    with pytest.raises(ValueError, match="top_candidates"):
        identify_degraded_sites(box(0, 0, 1, 1), project_start_year=2022, **kw)
