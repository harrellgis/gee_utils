"""Pure tests for scoring, thinning, spanning and the SOP selection matrix."""

import itertools

import numpy as np
import pandas as pd
import pytest

from eetools.site_selection.scoring import (
    MatrixContext,
    build_layer_register,
    build_selection_matrix,
    degradation_score,
    intactness_score,
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
        DegradedWeights(signature=0.1)


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


def test_degradation_redistributes_loss_and_signature():
    res = degradation_score(_pool(), is_forest=False)
    assert "loss" not in res.weights
    assert sum(res.weights.values()) == pytest.approx(1)
    no_sig = degradation_score(_pool(), use_signature=False)
    assert "signature" not in no_sig.weights


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
