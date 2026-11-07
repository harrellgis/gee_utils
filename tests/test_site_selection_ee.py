"""Earth Engine tests for the site-selection loaders and reductions.

Synthetic tests use ``ee.Image.constant`` on tiny UTM cells; ``slow`` tests touch real
datasets over a few cells only.
"""

import pandas as pd
import pytest
from shapely.geometry import box

pytestmark = pytest.mark.ee

CRS = "EPSG:32737"


def _cells(ids):
    rows = []
    for cid in ids:
        size, ix, iy = (int(v) for v in cid.split("_"))
        rows.append({"cell_id": cid, "ix": ix, "iy": iy, "size": float(size)})
    return pd.DataFrame(rows).set_index("cell_id")


def test_terraclimate_aridity_uses_unscaled_pr(ee_session, first_value):
    from eetools.sensors.terraclimate.preprocessing import (
        terraclimate_climatology_from_mean,
    )

    raw = ee_session.Image.constant([100, 1500, 300, 150, 280]).rename(
        ["pr", "pet", "def", "tmmn", "tmmx"]
    )
    out = terraclimate_climatology_from_mean(raw)
    assert first_value(out, "aridity") == pytest.approx((100 - 150) / 150, rel=1e-5)
    assert first_value(out, "cwd") == pytest.approx(360, rel=1e-5)
    assert first_value(out, "tmin") == pytest.approx(15, rel=1e-5)


def test_reduce_cells_column_naming(ee_session):
    from eetools.site_selection.predictors import (
        mean_percentile_reducer,
        reduce_cells,
    )

    cells = _cells(["1000_500_9850", "1000_501_9850"])
    one = ee_session.Image.constant(2.0).rename("a")
    df = reduce_cells(
        one,
        cells,
        bands=["a"],
        stats=["mean", "p10", "p90"],
        reducer=mean_percentile_reducer([10, 90]),
        scale=250,
        crs=CRS,
    )
    assert list(df.columns) == ["a_mean", "a_p10", "a_p90"]
    assert df["a_mean"].tolist() == pytest.approx([2.0, 2.0])
    two = ee_session.Image.constant([1.0, 3.0]).rename(["a", "b"])
    df2 = reduce_cells(
        two,
        cells,
        bands=["a", "b"],
        stats=None,
        reducer=ee_session.Reducer.median(),
        scale=250,
        crs=CRS,
    )
    assert df2.loc["1000_501_9850"].tolist() == pytest.approx([1.0, 3.0])


def test_focal_pixels_never_enter_retained_cells(ee_session):
    """A constant-1 focal image contributes 0 to an adjacent retained cell."""
    from eetools.site_selection.domain import to_lonlat
    from eetools.site_selection.predictors import (
        RunContext,
        reduce_cells,
        shapely_to_ee,
    )

    focal_utm = box(500_000, 9_850_000, 501_000, 9_851_000)
    excl = to_lonlat(focal_utm.buffer(500), CRS)
    focal_ll = to_lonlat(focal_utm, CRS)
    ctx = RunContext(
        focal_ee=shapely_to_ee(focal_ll),
        exclusion_ee=shapely_to_ee(excl),
        search_ee=shapely_to_ee(focal_ll).buffer(5000),
        crs=CRS,
        end_year=2024,
        trend_start_year=2010,
        state_start_year=2022,
        asset_ids={},
    )
    focal_one = (
        ee_session.Image.constant(0)
        .paint(ee_session.FeatureCollection([ee_session.Feature(ctx.focal_ee)]), 1)
        .rename("v")
        .toFloat()
        .updateMask(ctx.outside_mask())
    )
    df = reduce_cells(
        focal_one,
        _cells(["1000_501_9850"]),
        bands=["v"],
        stats=None,
        reducer=ee_session.Reducer.mean(),
        scale=30,
        crs=CRS,
    )
    assert df.iloc[0, 0] == pytest.approx(0.0)


@pytest.mark.slow
def test_new_loaders_return_expected_bands(ee_session, small_aoi):
    from eetools.sensors.human_modification.preprocessing import (
        get_human_modification,
    )
    from eetools.sensors.hydro.preprocessing import get_hand, get_ocean_mask
    from eetools.sensors.isda.preprocessing import get_soil_texture
    from eetools.sensors.pnv.preprocessing import get_pnv_biome
    from eetools.sensors.soilgrids.preprocessing import get_soilgrids_wrb
    from eetools.sensors.terraclimate.preprocessing import get_terraclimate_climatology
    from eetools.sensors.vecea.preprocessing import get_vecea_class
    from eetools.sensors.worldclim.preprocessing import get_worldclim_bio

    checks = {
        "ghm": get_human_modification(small_aoi),
        "hand": get_hand(small_aoi),
        "ocean": get_ocean_mask(),
        "texture": get_soil_texture(small_aoi),
        "pnv": get_pnv_biome(small_aoi),
        "wrb": get_soilgrids_wrb(small_aoi),
        "terraclimate": get_terraclimate_climatology(),
        "vecea": get_vecea_class(small_aoi),
        "worldclim": get_worldclim_bio(aoi=small_aoi),
    }
    bands = ee_session.Dictionary(
        {k: v.bandNames() for k, v in checks.items()}
    ).getInfo()
    assert bands["ghm"] == ["ghm"]
    assert bands["texture"] == ["texture_class", "sand", "clay"]
    assert bands["terraclimate"] == ["cwd", "aridity", "tmin", "tmax"]
    assert bands["vecea"] == ["vecea_class"] and bands["wrb"] == ["wrb_group"]
    assert bands["worldclim"] == ["bio01", "bio04", "bio12", "bio15"]


@pytest.mark.slow
def test_chirps_pentad_annual_sum_matches_daily(ee_session):
    from eetools.sensors.chirps.preprocessing import get_chirps_collection

    pt = ee_session.Geometry.Point([39.3, -4.3])
    sums = {}
    for cadence in ("daily", "pentad"):
        col = get_chirps_collection(pt, "2021-01-01", "2022-01-01", cadence=cadence)
        sums[cadence] = (
            col.sum()
            .reduceRegion(ee_session.Reducer.first(), pt, 5566)
            .get("precipitation")
        )
    out = ee_session.Dictionary(sums).getInfo()
    assert out["pentad"] == pytest.approx(out["daily"], abs=0.01)


@pytest.mark.slow
def test_nbar_and_annual_ndvi(ee_session, small_aoi):
    from eetools.sensors.modis.preprocessing import (
        get_modis_annual_nbar_indices,
        get_modis_annual_ndvi,
        get_modis_nbar_collection,
    )

    nbar = get_modis_nbar_collection(small_aoi, 2020, 2020)
    ndvi = get_modis_annual_ndvi(small_aoi, 2019, 2020)
    ann = get_modis_annual_nbar_indices(small_aoi, 2020, 2020)
    info = ee_session.Dictionary(
        {
            "n": nbar.size(),
            "bands": nbar.first().bandNames(),
            "ndvi_n": ndvi.size(),
            "ann_bands": ann.first().bandNames(),
        }
    ).getInfo()
    assert 44 <= info["n"] <= 47  # every 8th day of 2020
    assert info["bands"] == ["nir", "swir1", "swir2", "NDMI", "NBR"]
    assert info["ndvi_n"] == 2
    assert info["ann_bands"] == ["NDMI", "NBR"]


@pytest.mark.slow
def test_baseline_state_and_focal_anchor(ee_session, small_aoi):
    """Anchor constraint (D1): baseline-window bands and the focal reduction."""
    from eetools.site_selection.predictors import (
        RunContext,
        build_baseline_state,
        reduce_focal_anchor_state,
    )

    focal = ee_session.Geometry.Rectangle([39.22, -4.28, 39.225, -4.275])  # ~0.3 km2
    ctx = RunContext(
        focal_ee=focal,
        exclusion_ee=focal.buffer(500),
        search_ee=small_aoi,
        crs=CRS,
        end_year=2024,
        trend_start_year=2010,
        state_start_year=2022,
        asset_ids={},
        baseline_start_year=2019,
        baseline_end_year=2021,
    )
    assert build_baseline_state(ctx, "ndvi").bandNames().getInfo() == ["ndvi_base"]
    assert build_baseline_state(ctx, "ndmi").bandNames().getInfo() == ["ndmi_base"]
    vals = reduce_focal_anchor_state(ctx)
    assert set(vals) == {"ndvi_state", "ndmi_state", "ndvi_base", "ndmi_base"}
    for k, v in vals.items():
        assert -1.0 <= v <= 1.0, k  # finite and in index range
    no_base = reduce_focal_anchor_state(
        RunContext(**{**ctx.__dict__, "baseline_start_year": None})
    )
    assert pd.isna(no_base["ndvi_base"]) and pd.notna(no_base["ndvi_state"])
