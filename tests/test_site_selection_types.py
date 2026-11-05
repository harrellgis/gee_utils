"""Pure tests for site-selection types, constants tables and helpers."""

import numpy as np
import pandas as pd
import pytest

from eetools.constants import (
    DEGRADATION_TYPE_PROFILES,
    TERRACLIMATE_SCALES,
    VECEA_CLASS_GROUPS,
    VECEA_CLASS_TABLE,
    WRB_CLASS_TABLE,
    WRB_RELATED_GROUPS,
)
from eetools.sensors.assets import resolve_asset_id
from eetools.sensors.soilgrids.preprocessing import wrb_related_codes
from eetools.sensors.vecea.preprocessing import vecea_class_code, vecea_group
from eetools.site_selection.types import (
    EnvelopeTolerances,
    ScreeningTables,
    ScreeningTablesMismatch,
)
from eetools.site_selection.workflows import resolve_signature_mode


def test_tolerances_validation():
    EnvelopeTolerances()
    with pytest.raises(ValueError):
        EnvelopeTolerances(class_match="adjacent")
    with pytest.raises(ValueError):
        EnvelopeTolerances(relaxation="always")
    with pytest.raises(ValueError):
        EnvelopeTolerances(focal_class_min_share=0)


def _tables():
    coarse = pd.DataFrame(
        {"land_mean": [0.123456789012345, 1.0], "_done_stage1": [True, True]},
        index=pd.Index(["2000_1_2", "2000_1_3"], name="cell_id"),
    )
    fine = pd.DataFrame(
        {"ndvi_state_mean": [0.5], "_done_ndvi": [True]},
        index=pd.Index(["250_8_16"], name="cell_id"),
    )
    emb = pd.DataFrame({"cell_id": ["250_8_16"], "year": [2021], "A00": [0.1]})
    labels = pd.DataFrame(
        {"vecea_class": ['{"21": 3}']}, index=pd.Index(["250_8_16"], name="cell_id")
    )
    return ScreeningTables(coarse, fine, emb, labels, key={"end_year": 2025, "k": 8})


def test_screening_tables_parquet_roundtrip(tmp_path):
    t = _tables()
    t.to_parquet(tmp_path / "tables")
    back = ScreeningTables.from_parquet(tmp_path / "tables")
    pd.testing.assert_frame_equal(t.coarse, back.coarse)
    pd.testing.assert_frame_equal(t.fine, back.fine)
    pd.testing.assert_frame_equal(t.embeddings, back.embeddings)
    pd.testing.assert_frame_equal(t.labels, back.labels)
    assert back.key == t.key and back.created == t.created


def test_screening_tables_key_mismatch_names_fields():
    t = _tables()
    t.check_key({"end_year": 2025, "k": 8})
    with pytest.raises(ScreeningTablesMismatch, match="end_year"):
        t.check_key({"end_year": 2024, "k": 8})


def test_screening_tables_hold_raw_values_only():
    t = _tables()
    for df in (t.coarse, t.fine):
        assert not {"mess", "rank", "rank_score", "env_sim"} & set(df.columns)


def test_signature_mode_baselines():
    base, mode, msg = resolve_signature_mode(2016, "embedding")
    assert (base, mode) == (2015, "rules") and msg
    base, mode, msg = resolve_signature_mode(2018, "embedding")
    assert (base, mode, msg) == (2017, "embedding", None)
    with pytest.raises(ValueError):
        resolve_signature_mode(2030, "embedding")
    with pytest.raises(ValueError):
        resolve_signature_mode(2020, "blend")


def test_vecea_class_groups_table():
    assert len(VECEA_CLASS_TABLE) == 39
    expected = {
        "B": "bamboo",
        "E": "afroalpine_ericaceous",
        "R": "riverine",
        "P": "woodland",
        "D": "semi_desert",
        "G": "climatic_grassland",
        "g": "edaphic_grassland",
        "Be": "bushland",
        "Bd": "bushland",
        "wd": "woodland",
        "Fb/Be/wd": "forest",
        "g/X": "edaphic_grassland",
        "Bd/Wcd": "bushland",
        "CM": "coastal_mosaic",
    }
    by_code = {code: mid for mid, (code, _) in VECEA_CLASS_TABLE.items()}
    for code, group in expected.items():
        assert vecea_group(by_code[code]) == group, code
    assert vecea_group(by_code["s"]) is None and vecea_group(by_code["w"]) is None
    assert set(VECEA_CLASS_GROUPS) | {87, 88} == set(VECEA_CLASS_TABLE)
    assert vecea_class_code(21) == "Fb"


def test_wrb_related_groups_cover_all_codes_once():
    members = [c for s in WRB_RELATED_GROUPS.values() for c in s]
    assert sorted(members) == sorted(WRB_CLASS_TABLE)
    assert len(members) == len(set(members)) == 30
    arenosols, leptosols, andosols = 4, 16, 3
    assert leptosols not in wrb_related_codes(arenosols)
    assert wrb_related_codes(andosols) == [andosols]
    assert set(wrb_related_codes(10)) == {22, 10, 2, 19, 0, 17}  # Ferralsols set #6


def test_terraclimate_pr_is_unscaled():
    assert TERRACLIMATE_SCALES["pr"] == 1.0
    assert TERRACLIMATE_SCALES["pet"] == 0.1


def test_degradation_profiles_have_no_dynamic_world_terms():
    for profile in DEGRADATION_TYPE_PROFILES.values():
        cols = [c for c, _ in profile["terms"]]
        assert not any(c.startswith("dw_") for c in cols)
        assert set(profile["layers"]) <= {"canopy", "rue", "viirs"}


def test_resolve_asset_id_order(monkeypatch):
    monkeypatch.delenv("EETOOLS_VECEA_ASSET", raising=False)
    assert resolve_asset_id("vecea").endswith("pnv_vecea_v2_0_Kenya")
    monkeypatch.setenv("EETOOLS_VECEA_ASSET", "projects/x/assets/env")
    assert resolve_asset_id("vecea") == "projects/x/assets/env"
    assert resolve_asset_id("vecea", "projects/x/assets/arg") == "projects/x/assets/arg"
    with pytest.raises(KeyError):
        resolve_asset_id("glim")


def test_management_units_from_focal_sets():
    from eetools.site_selection.predictors import FocalProfile
    from eetools.site_selection.workflows import _focal_sets

    def prof(vecea, wrb):
        return FocalProfile(
            native_counts={},
            shares={"vecea_class": vecea, "wrb_group": wrb, "hand_riparian": {0: 1.0}},
            buffered={},
            elevation={},
            rainfall={},
            coast_dist=None,
            mess_reference=pd.DataFrame(),
            span_samples=pd.DataFrame(),
        )

    itf = _focal_sets(
        prof({21: 0.505, 3: 0.495}, {3: 0.96, 10: 0.04}), EnvelopeTolerances()
    )
    assert itf["class_set"] == [3, 21] and itf["wrb_code"] == 3
    assert itf["wrb_related"] == [3]
    ecosia = _focal_sets(
        prof({10: 0.87, 46: 0.13}, {10: 0.67, 19: 0.33}), EnvelopeTolerances()
    )
    assert ecosia["class_set"] == [10]
    assert set(ecosia["group_members"]) == {4, 5, 8, 9, 10}
    outside = _focal_sets(prof({}, {}), EnvelopeTolerances())
    assert outside["class_fallback"] and outside["substrate_fallback"]
    assert np.isnan(np.nan)
