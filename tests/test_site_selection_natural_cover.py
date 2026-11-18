"""Natural-cover degraded pathway (2026-11-18 spec): cropland gate, S2 anchor view,
natural-cover coarse ordering, bare-soil ranking term and option resolution.

Pure pandas, no Earth Engine.
"""

import numpy as np
import pandas as pd
import pytest

from eetools.constants import (
    CONVERSION_DEGRADATION_TYPES,
    DEGRADATION_TYPE_PROFILES,
    DEGRADED_MAX_DW_CROP,
    NATURAL_COVER_DEGRADATION_TYPES,
    S2_ANCHOR_FIRST_YEAR,
)
from eetools.site_selection.scoring import (
    anchor_constraint,
    anchor_view,
    conversion_pathway,
    degradation_score,
    degraded_exclusion_reasons,
    degraded_exclusions,
    natural_cover_refine_score,
)
from eetools.site_selection.types import (
    DegradedWeights,
    ScreeningTables,
    default_degraded_weights,
)
from eetools.site_selection.workflows import resolve_degraded_options


def _r(reasons):
    """Reason Series as a list with None for kept cells (NaN or None)."""
    return [None if pd.isna(v) else v for v in reasons]


def _cells(**cols):
    n = len(next(iter(cols.values())))
    base = {"water_frac": [0.0] * n, "built_frac": [0.0] * n}
    base.update(cols)
    return pd.DataFrame(base, index=[f"c{i}" for i in range(n)])


# --------------------------------------------------------------------------- #
# F1: pathway type                                                            #
# --------------------------------------------------------------------------- #
def test_natural_degradation_is_a_natural_cover_type():
    assert "natural_degradation" in DEGRADATION_TYPE_PROFILES
    assert "natural_degradation" not in CONVERSION_DEGRADATION_TYPES
    assert NATURAL_COVER_DEGRADATION_TYPES <= set(DEGRADATION_TYPE_PROFILES)
    assert not NATURAL_COVER_DEGRADATION_TYPES & CONVERSION_DEGRADATION_TYPES
    hist = pd.DataFrame({"modified_by_baseline": [1.0, 0.0]})
    assert conversion_pathway(hist, "natural_degradation") is None


# --------------------------------------------------------------------------- #
# F2: cropland gate                                                           #
# --------------------------------------------------------------------------- #
def test_cropland_gate_thresholds_and_nan():
    df = _cells(
        dw_crop_share=[0.06, 0.05, np.nan, 0.0],
        landuse_class=["mixed", "mixed", "mixed", "persistent_natural"],
    )
    r = degraded_exclusion_reasons(
        df, final=True, max_built_frac=0.02, max_dw_crop=DEGRADED_MAX_DW_CROP
    )
    assert _r(r) == ["cropland", None, None, None]


def test_converted_land_use_classes_are_gated():
    df = _cells(
        dw_crop_share=[0.0, 0.0, 0.0, 0.0],
        landuse_class=["converted", "persistent_modified", "mixed", "built"],
    )
    r = degraded_exclusion_reasons(
        df, final=True, max_built_frac=0.02, max_dw_crop=0.05
    )
    assert _r(r) == ["converted", "converted", None, None]


def test_cropland_gate_off_keeps_todays_reasons():
    df = _cells(
        water_frac=[0.6, 0.0, 0.0, 0.0],
        built_frac=[0.0, 0.03, 0.0, 0.0],
        dw_built_share=[0.0, 0.0, 0.2, 0.0],
        dw_crop_share=[0.9, 0.9, 0.9, 0.9],
        landuse_class=["converted"] * 4,
    )
    kw = dict(final=True, max_built_frac=0.02, max_dw_built=0.05)
    before = degraded_exclusion_reasons(df, **kw)
    after = degraded_exclusion_reasons(df, **kw, max_dw_crop=None)
    pd.testing.assert_series_equal(before, after)
    assert _r(before) == ["water", "built_wc", "built_dw", None]


def test_cropland_gate_ignored_on_coarse_screen_and_after_built():
    df = _cells(dw_crop_share=[0.9], landuse_class=["converted"], built_frac=[0.03])
    coarse = degraded_exclusion_reasons(
        df, final=False, max_built_frac=0.02, max_dw_crop=0.05
    )
    assert _r(coarse) == [None]
    final = degraded_exclusion_reasons(
        df, final=True, max_built_frac=0.02, max_dw_crop=0.05
    )
    assert _r(final) == ["built_wc"]  # first matching reason wins
    assert degraded_exclusions(
        _cells(dw_crop_share=[0.5]), final=True, max_built_frac=0.02, max_dw_crop=0.05
    ).tolist() == [True]


# --------------------------------------------------------------------------- #
# F3: coarse ordering                                                         #
# --------------------------------------------------------------------------- #
def test_natural_cover_refine_score_gates_and_orders_by_env_only():
    cs = pd.DataFrame(
        {"coarse_landuse_gate": [True, False, True, True]}, index=list("abcd")
    )
    env = pd.Series([0.2, 1.0, 0.8, np.nan], index=list("abcd"))
    sc = natural_cover_refine_score(cs, env)
    assert np.isnan(sc["b"])
    assert sc["c"] > sc["a"]
    assert sc["d"] == pytest.approx(0.75)  # missing env_sim -> 0.5


# --------------------------------------------------------------------------- #
# F4: anchor view and bare term                                               #
# --------------------------------------------------------------------------- #
def test_anchor_view_selects_sensor_columns():
    df = pd.DataFrame(
        {
            "ndvi_base": [0.5],
            "ndvi_state": [0.5],
            "ndmi_base": [0.1],
            "ndmi_state": [0.1],
            "s2_ndvi_base": [0.3],
            "s2_ndvi_state": [0.31],
            "s2_ndmi_base": [-0.05],
        }
    )
    m = anchor_view(df, "modis")
    s2 = anchor_view(df, "s2")
    assert m["ndvi_base"].iloc[0] == 0.5
    assert s2["ndvi_base"].iloc[0] == 0.3
    assert np.isnan(s2["ndmi_state"].iloc[0])  # missing column -> NaN
    with pytest.raises(ValueError):
        anchor_view(df, "landsat")


def test_s2_anchor_constraint_matches_modis_on_same_values():
    rng = np.random.default_rng(0)
    vals = {c: rng.uniform(0, 0.7, 20) for c in ("ndvi", "ndmi")}
    df = pd.DataFrame(
        {
            **{
                f"{x}_{w}": vals[x] + d
                for x in vals
                for w, d in (("base", 0), ("state", 0.01))
            },
            **{
                f"s2_{x}_{w}": vals[x] + d
                for x in vals
                for w, d in (("base", 0), ("state", 0.01))
            },
        }
    )
    focal = {"ndvi_base": 0.4, "ndvi_state": 0.45, "ndmi_base": 0.3, "ndmi_state": 0.3}
    a = anchor_constraint(anchor_view(df, "modis"), focal)
    b = anchor_constraint(anchor_view(df, "s2"), focal)
    pd.testing.assert_frame_equal(a, b)


def test_s2_anchor_missing_value_fails_constraint():
    df = pd.DataFrame({"s2_ndvi_base": [np.nan], "s2_ndvi_state": [0.1]})
    out = anchor_constraint(
        anchor_view(df, "s2"), {"ndvi_base": 0.5, "ndvi_state": 0.5}
    )
    assert not out["below_baseline_ndvi"].iloc[0]


def test_default_natural_cover_weights():
    w = default_degraded_weights("natural_degradation")
    assert w.unnatural == 0 and w.pressure == 0
    assert w.bare == pytest.approx(3.75 / 13)
    assert w.perf_deficit == pytest.approx(3.75 / 13)
    assert w.moist_deficit == pytest.approx(2.5 / 13)
    for t in NATURAL_COVER_DEGRADATION_TYPES:
        assert default_degraded_weights(t) == w
    for t in CONVERSION_DEGRADATION_TYPES:
        assert default_degraded_weights(t).bare == 0
    assert DegradedWeights().bare == 0
    with pytest.raises(ValueError):
        DegradedWeights(bare=0.5)  # must still sum to 1


def _rank_pool():
    return pd.DataFrame(
        {
            "ndvi_state": [0.5, 0.5, 0.5],
            "ndmi_state": [0.0, 0.0, 0.0],
            "natural_frac": [1.0, 1.0, 1.0],
            "ghm": [0.3, 0.3, 0.3],
            "ndvi_trend": [0.0, 0.0, 0.0],
            "nbr_trend": [0.0, 0.0, 0.0],
            "forest_loss_frac": [0.0, 0.0, 0.0],
            "bare_frac": [0.0, 0.0, 0.0],
            "s2_bsi_state": [-0.1, 0.05, 0.2],
        },
        index=["ref_like", "mid", "bare"],
    )


def test_bare_term_ranks_barer_cells_higher_for_natural_cover_only():
    pool = _rank_pool()
    nat = degradation_score(pool, default_degraded_weights("natural_degradation"))
    assert "bare" in nat.weights
    assert sum(nat.weights.values()) == pytest.approx(1.0)
    assert nat.score["bare"] > nat.score["mid"] > nat.score["ref_like"]
    gen = degradation_score(pool, default_degraded_weights("generic"))
    assert "bare" not in gen.weights
    assert gen.score.nunique() == 1  # identical apart from BSI, which is unweighted


def test_bare_term_falls_back_to_worldcover_bare_frac():
    pool = _rank_pool().drop(columns="s2_bsi_state")
    pool["bare_frac"] = [0.0, 0.1, 0.3]
    sc = degradation_score(pool, default_degraded_weights("overgrazing"))
    assert sc.score["bare"] > sc.score["ref_like"]


# --------------------------------------------------------------------------- #
# Option resolution                                                           #
# --------------------------------------------------------------------------- #
def test_auto_options_follow_the_pathway_family():
    crop, sensor, months, msgs = resolve_degraded_options(
        "natural_degradation",
        anchor_months=(6, 9),
        baseline_start_year=2019,
        state_start_year=2023,
    )
    assert (crop, sensor, months, msgs) == (DEGRADED_MAX_DW_CROP, "s2", (6, 9), [])
    crop, sensor, months, msgs = resolve_degraded_options(
        "generic", anchor_months=(6, 9), baseline_start_year=2019, state_start_year=2023
    )
    assert (crop, sensor, months) == (None, "modis", None)


def test_natural_cover_without_months_uses_modis_with_warning():
    crop, sensor, months, msgs = resolve_degraded_options("rangeland")
    assert crop == DEGRADED_MAX_DW_CROP and sensor == "modis" and months is None
    assert msgs and "anchor_months" in msgs[0]


def test_explicit_options_and_overrides():
    crop, sensor, months, _ = resolve_degraded_options(
        "generic",
        max_dw_crop=0.1,
        anchor_sensor="s2",
        anchor_months=[1, 2],
        baseline_start_year=2020,
        state_start_year=2023,
    )
    assert (crop, sensor, months) == (0.1, "s2", (1, 2))
    crop, *_ = resolve_degraded_options("natural_degradation", max_dw_crop=None)
    assert crop is None


def test_early_s2_window_falls_back_to_modis():
    _, sensor, months, msgs = resolve_degraded_options(
        "natural_degradation",
        anchor_months=(6, 9),
        baseline_start_year=S2_ANCHOR_FIRST_YEAR - 2,
        state_start_year=2023,
    )
    assert sensor == "modis" and months is None
    assert any("MODIS" in m for m in msgs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"anchor_sensor": "s2"},  # no months
        {"anchor_months": (9, 6)},  # crosses the year
        {"anchor_months": (0, 3)},
        {"anchor_months": (6,)},
        {"anchor_sensor": "landsat"},
        {"max_dw_crop": 1.5},
        {"max_dw_crop": "always"},
    ],
)
def test_invalid_options_raise(kwargs):
    with pytest.raises(ValueError):
        resolve_degraded_options("natural_degradation", **kwargs)


def test_s2_anchor_key_field_is_completable():
    key = {"a": 1, "s2_anchor": None, "project_start_year": None}
    tables = ScreeningTables.empty(key)
    tables.check_key(
        {"a": 1, "s2_anchor": {"months": [6, 9]}, "project_start_year": 2022}
    )
    tables2 = ScreeningTables.empty({"a": 1, "s2_anchor": {"months": [6, 9]}})
    with pytest.raises(ValueError):
        tables2.check_key({"a": 1, "s2_anchor": {"months": [1, 2]}})
