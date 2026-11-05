"""Pure tests for eetools.site_selection.domain (no Earth Engine)."""

import math

import numpy as np
import pytest
import shapely
from shapely.geometry import box

from eetools.constants import ALLOWED_K, FRACTION_SCALE_M
from eetools.site_selection.domain import (
    apply_exclusions,
    assign_tiers,
    build_candidate_grid,
    cell_polygons,
    derive_cell_size,
    refine_cells,
    search_rungs,
    utm_projection_for,
)

HA = 1e4


@pytest.mark.parametrize(
    "area_ha, expected",
    [
        (441, (2500.0, 2500.0)),  # TGBS_Kwale
        (97.6, (2000.0, 1000.0)),  # ITF
        (5.8, (2000.0, 250.0)),  # Ecosia
        (1000, (3500.0, 3500.0)),
        (36, (2000.0, 500.0)),
        (16, (2000.0, 500.0)),
    ],
)
def test_derive_cell_size_examples(area_ha, expected):
    assert derive_cell_size(area_ha * HA) == expected


@pytest.mark.parametrize("area_ha", np.geomspace(0.5, 5000, 60))
def test_derive_cell_size_invariants(area_ha):
    screen, fine = derive_cell_size(area_ha * HA)
    side = math.sqrt(area_ha * HA)
    k = screen / fine
    assert screen % fine == 0
    assert fine == int(fine)
    assert fine % FRACTION_SCALE_M == 0 and screen % FRACTION_SCALE_M == 0
    assert int(k) in ALLOWED_K
    if side > 2000:
        assert fine >= side and k == 1


def test_derive_cell_size_rejects_bad_inputs():
    with pytest.raises(ValueError):
        derive_cell_size(0)
    with pytest.raises(ValueError):
        derive_cell_size(1e4, screen_cell_m=1100)


def test_derive_cell_size_respects_min_fine_for_small_screen():
    screen, fine = derive_cell_size(5.8 * HA, screen_cell_m=1000)
    assert screen == 1000 and fine == 250


def test_assign_tiers():
    tiers = assign_tiers([0, 24_999, 25_000, 25_001, 50_000, 149_000, 151_000])
    assert list(tiers) == [1, 1, 1, 2, 2, 3, 0]
    with pytest.raises(ValueError):
        assign_tiers([1], (50, 25, 150))


def test_utm_projection_for_kenya():
    assert utm_projection_for(shapely.Point(36.8, -1.3)) == "EPSG:32737"
    assert utm_projection_for(shapely.Point(34.8, 0.3)) == "EPSG:32636"


def _focal():
    # 1 km square in UTM metres (southern hemisphere northings).
    return box(500_000, 9_850_000, 501_000, 9_851_000)


def test_build_candidate_grid_cells_align_and_tier():
    grid = build_candidate_grid(_focal(), 10_000, 2000, (3, 6, 10))
    assert (grid["size"] == 2000).all()
    assert set(grid["tier"]) <= {1, 2, 3}
    assert grid["dist_to_focal_m"].max() <= 10_000
    polys = cell_polygons(grid)
    xs = np.array([p.bounds[0] for p in polys])
    assert np.all(xs % 2000 == 0)
    assert grid.index.is_unique


def test_refine_cells_nests_exactly():
    grid = build_candidate_grid(_focal(), 4000, 2000, (2, 3, 4)).head(5)
    for k in (2, 4, 8):
        fine = refine_cells(grid, k)
        assert len(fine) == len(grid) * k * k
        for cid, row in grid.iterrows():
            parent = cell_polygons(grid.loc[[cid]])[0]
            kids = cell_polygons(fine[fine["parent_id"] == cid])
            assert all(parent.contains(c) or parent.equals(c) for c in kids)
            assert math.isclose(sum(c.area for c in kids), parent.area)
    with pytest.raises(ValueError):
        refine_cells(grid, 3)


def test_apply_exclusions_keeps_adjacent_cells():
    # Focal = exactly one 1 km cell; the neighbouring cell touches it (Buda case).
    focal = box(500_000, 9_850_000, 501_000, 9_851_000)
    grid = build_candidate_grid(focal, 3000, 1000, (1, 2, 3))
    out = apply_exclusions(grid, focal.buffer(500), "EPSG:32737", 0.5)
    assert "1000_500_9850" not in out.index  # the focal cell itself is dropped
    assert "1000_501_9850" in out.index  # adjacent cell retained
    adj = out.loc["1000_501_9850"]
    assert 0 < adj["focal_buffer_share"] <= 0.5
    assert adj["effective_area_ha"] < adj["area_ha"]
    assert (out["effective_area_ha"] <= out["area_ha"] + 1e-9).all()
    assert out["area_ha"].between(99, 101).all()


def test_search_rungs_orders():
    labels = [r.label for r in search_rungs("before_tier3")]
    assert labels == [
        "T1 strict",
        "T1-2 strict",
        "T1-2 substrate related",
        "T1-2 related + group",
        "T1-3 related + group",
    ]
    after = search_rungs("after_tier3")
    assert [r.max_tier for r in after] == [1, 2, 3, 3, 3]
    assert not any(r.relax_substrate or r.relax_class for r in search_rungs("off"))
    # before_tier3 relaxes before ever building Tier 3
    first_t3 = next(r for r in search_rungs("before_tier3") if r.max_tier == 3)
    assert first_t3.relax_substrate and first_t3.relax_class
    with pytest.raises(ValueError):
        search_rungs("sometimes")
