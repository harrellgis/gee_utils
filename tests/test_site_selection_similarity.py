"""Pure tests for MESS, collinearity, the envelope gate and embedding math."""

import numpy as np
import pandas as pd
import pytest

from eetools.site_selection.similarity import (
    FocalEnvelope,
    compute_mess,
    drop_collinear,
    embedding_query_similarity,
    envelope_gate,
    mean_embedding_vector,
    mess_similarity,
    normalise_rows,
    resolve_seed_source,
)


def test_mess_similarity_branches():
    ref = np.array([1.0, 2.0, 3.0, 4.0])  # min 1, max 4, range 3
    p = np.array([0.0, 1.5, 2.5, 3.5, 5.0])
    out = mess_similarity(ref, p, 1.0, 4.0)
    # p=0: f=0 -> (0-1)/3*100; p=1.5: f=25 -> 50; p=2.5: f=50 -> 100;
    # p=3.5: f=75 -> 50; p=5: f=100 -> (4-5)/3*100
    np.testing.assert_allclose(out, [-100 / 3, 50, 100, 50, -100 / 3])


def test_compute_mess_without_floors_matches_modeva_semantics():
    ref = {"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}
    cands = pd.DataFrame({"a": [2.5, 5.0], "b": [15.0, 25.0]}, index=["x", "y"])
    res = compute_mess(ref, cands, ["a", "b"], range_floors=None)
    # x: a -> 100, b (f=25) -> 50 => mess 50 limited by b
    # y: a -> -33.3, b (f=50) -> 100 => mess -33.3 limited by a
    assert res.scores.loc["x", "mess"] == pytest.approx(50)
    assert res.scores.loc["x", "mess_mod"] == "b"
    assert res.scores.loc["y", "mess"] == pytest.approx(-100 / 3)
    assert res.scores.loc["y", "mess_mod"] == "a"
    assert res.widened == [] and res.dropped == []


def test_compute_mess_zero_range_is_widened_and_finite():
    ref = {"bio01": [20.0] * 50, "elevation": [100, 101, 102]}
    cands = pd.DataFrame({"bio01": [20.1, 21.0], "elevation": [101.0, 101.0]})
    res = compute_mess(ref, cands, ["bio01", "elevation"])
    assert np.isfinite(res.scores["mess"]).all()
    assert "bio01" in res.widened and "elevation" in res.widened
    # 20.1 lies inside the widened 0.5 degC range -> positive similarity.
    assert res.scores.loc[0, "mess_bio01"] > 0
    # Without floors, a zero range is dropped rather than producing NaN/inf.
    res2 = compute_mess(ref, cands, ["bio01", "elevation"], range_floors=None)
    assert "bio01" in res2.dropped
    assert np.isfinite(res2.scores["mess"]).all()


def test_drop_collinear_priority_and_constant_column():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    pool = pd.DataFrame(
        {
            "elevation": x,
            "bio01": -x * 2 + 0.01 * rng.normal(size=50),
            "sand": 5.0,
            "clay": rng.normal(size=50),
        }
    )
    kept, log = drop_collinear(pool, ["elevation", "bio01", "sand", "clay"])
    assert kept == ["elevation", "sand", "clay"]
    assert log[0]["dropped"] == "bio01" and log[0]["kept"] == "elevation"


def _gate_table():
    return pd.DataFrame(
        {
            "land_mean": [1.0, 1.0, 1.0, 0.3],
            "in_class_set_mean": [0.06, 0.40, 0.55, 1.0],
            "in_group_set_mean": [0.06, 0.75, 0.55, 1.0],
            "wrb_same_mean": [1.0, 1.0, 0.2, 1.0],
            "wrb_related_mean": [1.0, 1.0, 0.9, 1.0],
            "hand_focal_mean": [1.0, 1.0, 1.0, 1.0],
            "elevation_p10": [900.0, 1000, 1000, 1000],
            "elevation_p50": [1400.0, 1000, 1000, 1000],
            "elevation_p90": [1900.0, 1000, 1000, 1000],
            "rain_mean_annual": [1000.0, 1000, 1300, 1000],
        },
        index=["remnant", "split", "soil", "ocean"],
    )


def test_envelope_gate_coarse_rules():
    focal = FocalEnvelope(elevation=1000, rainfall=1000)
    g = envelope_gate(_gate_table(), focal, coarse_rule=True, coarse_min_share=0.05)
    # median 400 m off but p10-p90 overlaps focal +/- 300 -> passes on coarse cells
    assert g.loc["remnant", "gate_elevation"]
    assert g.loc["remnant", "passes"]
    assert not g.loc["ocean", "gate_land"]
    assert not g.loc["soil", "gate_rainfall"]  # +30% rainfall


def test_envelope_gate_fine_rules_and_relaxation():
    focal = FocalEnvelope(elevation=1000, rainfall=1000)
    g = envelope_gate(_gate_table(), focal, coarse_rule=False)
    assert not g.loc["remnant", "gate_elevation"]  # fine cells use the median
    assert not g.loc["split", "gate_class"]  # 40/35/25 split: share 0.40 < 0.5
    g2 = envelope_gate(_gate_table(), focal, coarse_rule=False, class_level="group")
    assert g2.loc["split", "gate_class"]
    assert not g.loc["soil", "gate_substrate"]
    g3 = envelope_gate(
        _gate_table(), focal, coarse_rule=False, substrate_level="related"
    )
    assert g3.loc["soil", "gate_substrate"]
    g4 = envelope_gate(_gate_table(), focal, coarse_rule=False, substrate_level="off")
    assert g4["gate_substrate"].all()


def test_envelope_gate_mosaic_class_passes():
    focal = FocalEnvelope(elevation=1000, rainfall=1000, class_mosaic=True)
    g = envelope_gate(_gate_table(), focal, coarse_rule=False)
    assert g["gate_class"].all()


def test_embedding_renormalisation():
    a = normalise_rows(np.array([[3.0, 4.0], [0.0, 2.0]]))
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0)
    q = mean_embedding_vector(np.array([[1.0, 0.0], [0.0, 1.0]]))
    np.testing.assert_allclose(q, [2**-0.5, 2**-0.5])
    emb = pd.DataFrame([[0.5, 0.5], [0.0, 0.3]], columns=["A", "B"], index=["c1", "c2"])
    sim = embedding_query_similarity(emb, np.array([1.0, 1.0]), bands=["A", "B"])
    assert sim["c1"] == pytest.approx(1.0)
    assert sim["c2"] == pytest.approx(2**-0.5)


def test_resolve_seed_source():
    assert resolve_seed_source(True, "none", 0) == ("user", True)
    assert resolve_seed_source(False, "wdpa_strict", 3) == ("wdpa_strict", True)
    assert resolve_seed_source(False, "wdpa_strict", 1) == ("pseudo", False)
    assert resolve_seed_source(False, "none", 10) == ("pseudo", False)
