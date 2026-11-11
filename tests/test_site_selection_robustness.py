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
    assert t.min_natural_persist == 0.8 and t.unit_coverage_stop
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
    assert w.natural_persist == pytest.approx(0.10)
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
def test_export_writes_separate_candidate_and_pool_files(tmp_path):
    from eetools.site_selection.types import ScreeningTables, SiteSelectionResult
    from eetools.site_selection.workflows import export_site_selection

    geom = [box(36.0, -1.0, 36.01, -0.99), box(36.02, -1.0, 36.03, -0.99)]
    ranked = gpd.GeoDataFrame(
        {"rank": [1], "landuse_class": ["persistent_natural"]},
        geometry=geom[:1],
        crs=4326,
        index=pd.Index(["c1"], name="cell_id"),
    )
    pool = gpd.GeoDataFrame(
        {
            "pool_status": ["eligible", "gated_out"],
            "gate_reason": [None, "natural_persist"],
            "landuse_class": ["persistent_natural", "converted"],
        },
        geometry=geom,
        crs=4326,
        index=pd.Index(["c1", "c2"], name="cell_id"),
    )
    res = SiteSelectionResult(
        site_type="reference",
        ranked=ranked,
        pool=pool,
        focal_profile=pd.DataFrame(),
        params={},
        provenance={},
        selection_matrix=pd.DataFrame({"cell_id": ["c1"]}),
        layer_register=pd.DataFrame({"layer_or_query": ["x"]}),
        screening=ScreeningTables.empty({}),
        warnings=[],
    )
    paths = export_site_selection(
        res, tmp_path, project="TGBS", site="T", run_date="20261110"
    )
    cand = paths["reference_candidates"]
    pl = paths["reference_pool"]
    assert cand.name == "TGBS_T_reference_candidates_20261110_v1.gpkg"
    assert pl.name == "TGBS_T_reference_pool_20261110_v1.gpkg"
    back = gpd.read_file(pl, layer="pool")
    assert set(back["pool_status"]) == {"eligible", "gated_out"}
    assert "landuse_class" in back.columns
    assert len(gpd.read_file(cand, layer="candidates")) == 1
    # re-export overwrites instead of appending
    export_site_selection(res, tmp_path, project="TGBS", site="T", run_date="20261110")
    assert len(gpd.read_file(pl, layer="pool")) == 2


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


def test_export_writes_top_candidates_and_site_role(tmp_path):
    from eetools.site_selection.types import ScreeningTables, SiteSelectionResult
    from eetools.site_selection.workflows import export_site_selection

    geom = [box(36.0, -1.0, 36.01, -0.99), box(36.02, -1.0, 36.03, -0.99)]
    idx = pd.Index(["c1", "c2"], name="cell_id")
    pool = gpd.GeoDataFrame(
        {"pool_status": ["eligible", "gated_out"]}, geometry=geom, crs=4326, index=idx
    )
    top = gpd.GeoDataFrame(
        {"top_rank": [1], "top_score": [0.9], "proposed": [True]},
        geometry=geom[:1],
        crs=4326,
        index=idx[:1],
    )
    res = SiteSelectionResult(
        site_type="degraded",
        ranked=pool.iloc[:1].assign(rank=1),
        pool=pool,
        focal_profile=pd.DataFrame(),
        params={"top_candidates_n": 10},
        provenance={},
        selection_matrix=pd.DataFrame({"cell_id": ["c1"]}),
        layer_register=pd.DataFrame({"layer_or_query": ["x"]}),
        screening=ScreeningTables.empty({}),
        warnings=[],
        top_candidates=top,
    )
    vec = tmp_path / "vectors"
    paths = export_site_selection(
        res,
        tmp_path / "tables",
        project="TGBS",
        site="T",
        run_date="20261111",
        vector_folder=vec,
    )
    t = paths["degraded_top10"]
    assert t == vec / "TGBS_T_degraded_top10_20261111_v1.gpkg"
    assert paths["degraded_pool"].parent == vec
    assert paths["degraded_ranked"].parent == tmp_path / "tables"
    back = gpd.read_file(t, layer="top_candidates")
    assert list(back["top_rank"]) == [1]
    for product in ("degraded_candidates", "degraded_pool", "degraded_top10"):
        assert set(gpd.read_file(paths[product])["site_role"]) == {"degraded"}
