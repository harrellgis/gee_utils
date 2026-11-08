"""Pure tests for scoring, thinning, spanning and the SOP selection matrix."""

import itertools

import numpy as np
import pandas as pd
import pytest

from eetools.site_selection.scoring import (
    MatrixContext,
    anchor_check,
    anchor_constraint,
    baseline_window,
    build_layer_register,
    build_selection_matrix,
    degradation_score,
    focal_baseline_pct,
    intactness_score,
    pathway_gate,
    percentile_normalise,
    select_spanning_set,
    signature_from_rules,
    thin_spatially,
    zero_star_likelihood,
)
from eetools.site_selection.types import DegradedWeights, ReferenceWeights


def test_percentile_normalise_ties_and_span():
    s = pd.Series([3.0, 1.0, 1.0, np.nan, 5.0])
    p = percentile_normalise(s)
    assert p.iloc[1] == p.iloc[2]
    assert p.min() >= 0 and p.max() == 1  # tied minima share an averaged rank
    assert np.isnan(p.iloc[3])
    assert percentile_normalise(pd.Series([7.0])).iloc[0] == 1.0
    # a narrow-range column (e.g. cosine similarities) still spans 0-1
    narrow = percentile_normalise(pd.Series([0.951, 0.952, 0.953]))
    assert list(narrow) == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("c", [0.3, 1.0, 17.0])
def test_c3_invariance(c):
    rng = np.random.default_rng(1)
    s = pd.Series(rng.random(40))
    pd.testing.assert_series_equal(percentile_normalise(s), percentile_normalise(s / c))
    t = pd.Series([-0.02, 0.0, 0.01, 0.03, -0.01])
    capped = percentile_normalise(t.clip(upper=0))
    assert capped.iloc[1] == capped.iloc[2] == capped.iloc[3]  # non-negative trends tie
    assert capped.iloc[0] < capped.iloc[4] < capped.iloc[1]


def test_weights_must_sum_to_one():
    ReferenceWeights()
    DegradedWeights()
    with pytest.raises(ValueError):
        ReferenceWeights(naturalness=0.5)
    with pytest.raises(ValueError):
        DegradedWeights(perf_deficit=0.5)
    # The signature is a pathway gate since 2026-11-07, never a ranking weight.
    with pytest.raises(TypeError):
        DegradedWeights(signature=0.1)  # type: ignore[call-arg]
    w = DegradedWeights()
    assert w.perf_deficit == pytest.approx(3 / 13)
    assert w.loss == pytest.approx(1 / 13)


def _pool(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ghm": rng.random(n),
            "natural_frac": rng.random(n),
            "core_frac": rng.random(n),
            "ndvi_state": rng.random(n),
            "ndmi_state": rng.random(n),
            "ndvi_trend": rng.normal(0, 0.01, n),
            "nbr_trend": rng.normal(0, 0.01, n),
            "bii": rng.random(n),
            "pa_overlap": rng.choice([0, 0, 0.5, 1], n),
            "forest_loss_frac": rng.random(n) * 0.1,
            "signature_sim": rng.uniform(0.9, 0.97, n),
            "canopy_height_p50": rng.random(n) * 20,
        },
        index=[f"c{i}" for i in range(n)],
    )


def test_intactness_weights_and_forest_shift():
    res = intactness_score(_pool())
    assert sum(res.weights.values()) == pytest.approx(1)
    assert "canopy" not in res.weights
    res_f = intactness_score(_pool(), is_forest=True)
    assert res_f.weights["forest_persistence"] == pytest.approx(0.05)
    assert res_f.weights["canopy"] == pytest.approx(0.05)
    assert res_f.weights["core"] == pytest.approx(0.10)
    assert res.score.between(0, 1).all()


def test_degradation_redistributes_loss_and_ignores_signature():
    res = degradation_score(_pool(), is_forest=False)
    assert "loss" not in res.weights
    assert "signature" not in res.weights
    assert sum(res.weights.values()) == pytest.approx(1)
    forest = degradation_score(_pool(), is_forest=True)
    assert forest.weights["loss"] == pytest.approx(1 / 13)
    # signature_sim in the pool does not change the score (it gates, never ranks)
    other = _pool().assign(signature_sim=np.linspace(0, 1, 20))
    pd.testing.assert_series_equal(res.score, degradation_score(other).score)


def test_baseline_window():
    assert baseline_window(2022) == (2019, 2021)
    assert baseline_window(2003) == (2001, 2002)
    assert baseline_window(2002) == (2001, 2001)
    assert baseline_window(2001) is None


_FOCAL = {"ndvi_base": 0.40, "ndvi_state": 0.55, "ndmi_base": 0.10, "ndmi_state": 0.20}


def _anchor_df(**cols):
    base = {"ndvi_base": 0.3, "ndvi_state": 0.3, "ndmi_base": 0.0, "ndmi_state": 0.0}
    base.update(cols)
    return pd.DataFrame([base], index=["c"])


def test_anchor_constraint_both_windows_and_both_indices():
    ok = anchor_constraint(_anchor_df(), _FOCAL)
    assert ok.loc["c", "below_baseline"]
    # healthier than the focal site at baseline, degraded since -> must fail
    late = anchor_constraint(_anchor_df(ndvi_base=0.45, ndvi_state=0.2), _FOCAL)
    assert not late.loc["c", "below_baseline_ndvi"]
    assert not late.loc["c", "below_baseline"]
    # current state above the focal baseline (focal improved): ceiling is min(base, now)
    above = anchor_constraint(_anchor_df(ndvi_state=0.45), _FOCAL)
    assert not above.loc["c", "below_baseline"]
    # NDVI fine but NDMI fails -> overall fails, and the column says which
    ndmi = anchor_constraint(_anchor_df(ndmi_base=0.15), _FOCAL)
    assert ndmi.loc["c", "below_baseline_ndvi"]
    assert not ndmi.loc["c", "below_baseline_ndmi"]
    assert not ndmi.loc["c", "below_baseline"]
    # focal declined: current ceiling is the focal current value
    declined = {**_FOCAL, "ndvi_state": 0.25}
    assert not anchor_constraint(_anchor_df(), declined).loc["c", "below_baseline"]


def test_anchor_constraint_missing_values():
    # a NaN candidate value cannot be verified -> fails
    nan_cand = anchor_constraint(_anchor_df(ndvi_base=np.nan), _FOCAL)
    assert not nan_cand.loc["c", "below_baseline"]
    # no focal baseline -> only the current-window test applies
    no_base = {"ndvi_state": 0.55, "ndmi_state": 0.20}
    res = anchor_constraint(_anchor_df(ndvi_base=0.9, ndmi_base=0.9), no_base)
    assert res.loc["c", "below_baseline"]


def test_pathway_gate():
    g = pathway_gate(pd.Series([0.0, 0.49, 0.5, 1.0, np.nan]), 0.5)
    assert list(g) == ["failed", "failed", "pass", "pass", "failed"]


def test_worked_example_zero_star_beats_baseline_look_alike():
    # A: near zero-star but a weak baseline match; B: a moderately degraded
    # look-alike. With the signature as a gate (both pass), A must rank above B.
    pool = _pool(n=10, seed=3).drop(columns="signature_sim")
    pool.loc["c0", ["ndvi_state", "ndmi_state", "natural_frac", "ghm"]] = [
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    pool.loc["c0", ["ndvi_trend", "nbr_trend"]] = [-0.05, -0.05]
    pool.loc["c1", ["ndvi_state", "ndmi_state", "natural_frac", "ghm"]] = [
        0.5,
        0.5,
        0.5,
        0.5,
    ]
    score = degradation_score(pool).score
    assert score["c0"] > score["c1"]


def test_focal_baseline_pct_and_anchor_check():
    df = pd.DataFrame(
        {"ndvi_state": [0.1, 0.39, 0.9], "ndmi_state": [0.0, 0.0, 0.0]},
        index=["deep", "close", "above_ref"],
    )
    pcts = focal_baseline_pct(df, _FOCAL, {"ndvi": 0.9, "ndmi": 0.3})
    assert pcts.loc["deep", "focal_baseline_pct_ndvi"] == pytest.approx(0.375)
    assert pcts.loc["close", "focal_baseline_pct_ndvi"] == pytest.approx(
        (0.40 - 0.39) / (0.9 - 0.39)
    )
    assert np.isnan(pcts.loc["above_ref", "focal_baseline_pct_ndvi"])
    chk = anchor_check(pcts, 0.3)
    assert chk["deep"] == "review"  # NDVI 0.375 and NDMI 0.333 exceed 0.3
    assert chk["close"] == "review"  # NDMI 0.10 / 0.30 = 0.333 > 0.3
    assert list(anchor_check(pcts, 0.5)) == ["ok", "ok", "ok"]


def test_signature_from_rules_orients_terms():
    df = pd.DataFrame({"bare_frac": [0.0, 0.5, 1.0], "natural_frac": [1.0, 0.5, 0.0]})
    sig = signature_from_rules(
        df, [("bare_frac", 1), ("natural_frac", -1), ("missing", 1)]
    )
    assert list(sig) == [0.0, 0.5, 1.0]


def test_zero_star_likelihood():
    z = zero_star_likelihood(
        pd.Series([0.95, 0.95, 0.8, 0.1]), pd.Series([0.4, 0.6, 0.2, 0.1])
    )
    assert list(z) == ["high", "moderate", "moderate", "low"]


def test_thin_spatially_enforces_separation():
    df = pd.DataFrame(
        {"cx": [0, 500, 1000, 3000, 3500], "cy": [0, 0, 0, 0, 0], "s": [5, 4, 3, 2, 1]},
        index=list("abcde"),
    )
    kept = thin_spatially(df, "s", 1000)
    assert list(kept) == ["a", "c", "d"]
    for i, j in itertools.combinations(kept, 2):
        assert abs(df.loc[i, "cx"] - df.loc[j, "cx"]) >= 1000


def _span_inputs():
    idx = [f"r{i}" for i in range(8)]
    cand = pd.DataFrame(
        {
            "elevation": [100, 102, 150, 60, 101, 99, 103, 140],
            "bio12": [800, 805, 900, 700, 801, 799, 806, 880],
            "rank_score": np.linspace(1, 0.3, 8),
        },
        index=idx,
    )
    focal = pd.DataFrame({"elevation": [60, 100, 150], "bio12": [700, 800, 900]})
    return cand, focal


def test_select_spanning_set_primary_and_coverage():
    cand, focal = _span_inputs()
    elig = pd.Series(True, index=cand.index)
    res = select_spanning_set(
        cand,
        focal,
        cand,
        ["elevation", "bio12"],
        3,
        eligible=elig,
        no_mandatory_zero=elig,
    )
    assert res.span_mode == "spanning"
    assert res.selected.index[0] == "r0"
    assert list(res.selected["role"]) == ["primary", "secondary", "secondary"]
    # coverage pulls in the high-elevation r2 (or r7) and/or the low r3
    assert set(res.selected.index[1:]) & {"r2", "r3", "r7"}
    pure = select_spanning_set(
        cand,
        focal,
        cand,
        ["elevation", "bio12"],
        3,
        eligible=elig,
        no_mandatory_zero=elig,
        span_environment=False,
    )
    assert list(pure.selected.index) == ["r0", "r1", "r2"]
    assert res.env_coverage >= pure.env_coverage


def test_select_spanning_set_rank_fallback_and_units():
    cand, focal = _span_inputs()
    elig = pd.Series(False, index=cand.index)
    elig.iloc[:2] = True
    nz = pd.Series(True, index=cand.index)
    res = select_spanning_set(
        cand,
        focal,
        cand,
        ["elevation", "bio12"],
        3,
        eligible=elig,
        no_mandatory_zero=nz,
    )
    assert res.span_mode == "rank_fallback"
    assert any("fall back" in w for w in res.warnings)
    cand["unit_keys"] = [{"Fb"}] * 7 + [{"B"}]
    elig = pd.Series(True, index=cand.index)
    res2 = select_spanning_set(
        cand,
        focal,
        cand,
        ["elevation", "bio12"],
        2,
        eligible=elig,
        no_mandatory_zero=elig,
        units={"Fb": {"Fb"}, "B": {"B"}},
        span_score_floor=0.0,
    )
    assert res2.units == {"Fb": True, "B": True}
    assert "r7" in res2.selected.index


def _ctx(**over):
    pool = pd.DataFrame({"bii": np.linspace(0, 1, 21), "ghm": np.linspace(0, 1, 21)})
    base = dict(
        site_type="reference",
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
        focal_class_label="Fb",
        focal_substrate_label="Ferralsols",
        pool=pool,
    )
    base.update(over)
    return MatrixContext(**base)


def _cand(**over):
    row = dict(
        in_class_set_mean=1.0,
        in_group_set_mean=1.0,
        wrb_same_mean=1.0,
        wrb_related_mean=1.0,
        elevation_p50=1000.0,
        rain_mean_annual=1000.0,
        loss_since_2000_frac=0.0,
        recent_loss_frac=0.0,
        bii=0.9,
        ghm=0.05,
        natural_frac=0.95,
        effective_area_ha=100.0,
        pa_i_iv_share=0.0,
        pa_share=0.0,
    )
    row.update(over)
    return row


def test_matrix_elevation_and_rainfall_bands_never_narrower_than_half_tolerance():
    cands = pd.DataFrame(
        [
            _cand(elevation_p50=1149, rain_mean_annual=1124),
            _cand(elevation_p50=1151, rain_mean_annual=1126),
            _cand(elevation_p50=1301, rain_mean_annual=1260),
        ],
        index=["a", "b", "c"],
    )
    m = build_selection_matrix(cands, _ctx())
    assert list(m["elevation_score"]) == [2, 1, 0]
    assert list(m["rainfall_score"]) == [2, 1, 0]


def test_matrix_integrity_exclusive_and_exhaustive():
    combos = itertools.product(
        [0.0, 0.005, 0.03, 0.08],
        [0.0, 0.003, 0.01],
        [0.1, 0.9],
        [0.05, 0.9],
        [0.5, 0.9],
    )
    rows = [
        _cand(loss_since_2000_frac=a, recent_loss_frac=b, bii=c, ghm=d, natural_frac=e)
        for a, b, c, d, e in combos
    ]
    m = build_selection_matrix(pd.DataFrame(rows), _ctx())
    assert set(m["integrity_score"]) <= {0, 1, 2}
    df = pd.DataFrame(rows)
    zero = (df["loss_since_2000_frac"] > 0.05) | (df["recent_loss_frac"] > 0.005)
    assert (m.loc[zero, "integrity_score"] == 0).all()
    assert (m.loc[~zero, "integrity_score"] >= 1).all()
    two = (
        (~zero)
        & (df["loss_since_2000_frac"] < 0.01)
        & (df["bii"] >= 0.75)
        & (1 - df["ghm"] >= 0.75)
        & (df["natural_frac"] >= 0.8)
    )
    assert (m.loc[two, "integrity_score"] == 2).all()
    assert (m.loc[~zero & ~two, "integrity_score"] == 1).all()


def test_matrix_relaxation_eligibility_and_size():
    cands = pd.DataFrame(
        [
            _cand(wrb_same_mean=0.1, wrb_related_mean=0.9),
            _cand(wrb_same_mean=0.1, wrb_related_mean=0.1),
            _cand(),
            _cand(natural_frac=0.4, effective_area_ha=100.0),
        ],
        index=["related", "different", "strict", "small"],
    )
    m = build_selection_matrix(cands, _ctx(relaxed_criteria={"substrate"}))
    assert m.loc["related", "substrate_score"] == 1
    assert (
        m.loc["related", "mandatory_met_relaxed"]
        and not m.loc["related", "mandatory_met"]
    )
    assert m.loc["related", "fallback"]
    assert not m.loc["different", "mandatory_met_relaxed"]
    assert m.loc["strict", "mandatory_met"]
    # size compares against min(focal_area 50 ha, effective area)
    assert m.loc["strict", "size_core_score"] == 2
    # 40 ha natural vs a 50 ha target: >= 50% of the target -> 1
    assert m.loc["small", "size_core_score"] == 1


def test_degraded_matrix_below_baseline_is_mandatory():
    cands = pd.DataFrame(
        [
            _cand(
                below_baseline=True, below_baseline_ndvi=True, below_baseline_ndmi=True
            ),
            _cand(
                below_baseline=False,
                below_baseline_ndvi=True,
                below_baseline_ndmi=False,
            ),
        ],
        index=["anchor", "too_green"],
    )
    m = build_selection_matrix(cands, _ctx(site_type="degraded"))
    assert m.loc["anchor", "below_baseline_score"] == 2
    assert m.loc["too_green", "below_baseline_score"] == 0
    assert "ndmi" in m.loc["too_green", "below_baseline_evidence"]
    assert m.loc["anchor", "mandatory_met"]
    assert not m.loc["too_green", "mandatory_met"]


def test_layer_register_marks_thresholds_low_confidence():
    reg = build_layer_register(
        {"WDPA": "WCMC/WDPA/current/polygons"},
        {
            "tolerances": {"elevation_tol_m": 300},
            "mess_range_floors": {"slope": ["abs", 2.0]},
        },
        "2026-11-05",
    )
    assert set(reg.columns) == {
        "layer_or_query",
        "version_or_date",
        "extent_or_parameters",
        "access_date",
        "result",
        "confidence",
    }
    assert (
        reg.loc[reg["layer_or_query"].str.startswith("threshold"), "confidence"] == "L"
    ).all()
    assert reg.loc[reg["layer_or_query"] == "WDPA", "confidence"].item() == "H"


# --------------------------------------------------------------------------- #
# Integrity stratum and degraded built-up filters (Ecosia first-run review)    #
# --------------------------------------------------------------------------- #
def test_forest_stratum_excludes_woodland_and_wooded_grassland():
    from eetools.constants import VECEA_CLASS_GROUPS
    from eetools.site_selection.scoring import forest_stratum

    def groups(*map_ids):
        return [VECEA_CLASS_GROUPS[m] for m in map_ids]

    kw = {"class_fallback": False, "pnv_code": None}
    # Ecosia: Be (bushland) + Wcd (dry combretum wooded grassland) -> non-forest
    assert not forest_stratum(groups(10, 46), **kw)
    assert not forest_stratum(["woodland"], **kw)
    # ITF: Fb (forest) + B (bamboo) -> forest; riverine stays forest
    assert forest_stratum(groups(21, 3), **kw)
    assert forest_stratum(["riverine"], **kw)
    # a mixed set with any forest group is forest
    assert forest_stratum(["woodland", "forest"], **kw)
    # tree cover / mangrove target classes force the forest stratum
    assert forest_stratum(["woodland"], **kw, target_cover_classes=[10, 30])
    assert forest_stratum(["woodland"], **kw, target_cover_classes=[95])
    assert not forest_stratum(["woodland"], **kw, target_cover_classes=[20, 30])


def test_forest_stratum_pnv_fallback():
    from eetools.site_selection.scoring import forest_stratum

    def fb(code):
        return forest_stratum([], class_fallback=True, pnv_code=code)

    assert fb(1) and fb(3)  # tropical evergreen / deciduous forest
    assert not fb(16) and not fb(17)  # open woodland / shrubland
    assert not fb(18) and not fb(20)  # tropical savanna, xerophytic woods/scrub
    # the PNV code is ignored when VECEA covers the focal site
    assert not forest_stratum([], class_fallback=False, pnv_code=1)


def test_degraded_exclusions_thresholds():
    from eetools.site_selection.scoring import degraded_exclusions

    pool = pd.DataFrame(
        {
            "water_frac": [0.0, 0.0, 0.0, 0.0, 0.6, 0.0],
            "built_frac": [0.0, 0.01, 0.0101, 0.4, 0.0, np.nan],
            "pa_share": [0.0, 0.0, 0.0, 0.0, 0.0, 0.7],
        },
        index=list("abcdef"),
    )
    final = degraded_exclusions(pool, final=True, max_built_frac=0.01)
    # 0.01 kept (inclusive max), 0.0101 dropped, 0.4 dropped, water dropped, NaN built kept
    assert list(final) == [False, False, True, True, True, False]
    # coarse screen keeps the old >= 0.5 built rule
    coarse = degraded_exclusions(pool, final=False, max_built_frac=0.01)
    assert list(coarse) == [False, False, False, False, True, False]
    # settlement keeps built-up land; water still goes
    settle = degraded_exclusions(pool, final=True, max_built_frac=0.01, keep_built=True)
    assert list(settle) == [False, False, False, False, True, False]
    # max 0 drops any built-up pixel
    strict = degraded_exclusions(pool, final=True, max_built_frac=0.0)
    assert list(strict) == [False, True, True, True, True, False]
    prot = degraded_exclusions(
        pool, final=True, max_built_frac=0.01, exclude_protected=True
    )
    assert prot["f"]
    # missing columns are treated as 0
    bare = degraded_exclusions(
        pd.DataFrame(index=["x"]), final=True, max_built_frac=0.01
    )
    assert list(bare) == [False]


def test_dw_built_filter_keeps_order_and_renumbers():
    from eetools.site_selection.scoring import dw_built_filter

    short = pd.DataFrame(
        {"rank": [1, 2, 3, 4], "rank_score": [0.9, 0.8, 0.7, 0.6]},
        index=["c1", "c2", "c3", "c4"],
    )
    shares = pd.Series({"c1": 0.30, "c2": 0.01, "c3": np.nan, "c4": 0.0})
    kept, dropped = dw_built_filter(short, shares, 0.01)
    assert dropped == ["c1"]
    assert list(kept.index) == [
        "c2",
        "c3",
        "c4",
    ]  # NaN (no DW obs) kept; 0.01 inclusive
    assert list(kept["rank"]) == [1, 2, 3]
    assert kept.loc["c2", "dw_built_share"] == 0.01 and np.isnan(
        kept.loc["c3", "dw_built_share"]
    )
    empty, gone = dw_built_filter(short, pd.Series(1.0, index=short.index), 0.01)
    assert empty.empty and gone == list(short.index)
