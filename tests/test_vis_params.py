"""Tests for eetools.visualization.vis_params (plain dicts, no Earth Engine)."""

import pytest

from eetools.constants import HLS_COMMON_BANDS, L8_BANDS, S2_BANDS
from eetools.visualization import vis_params

PALETTE_PARAMS = [
    vis_params.NDVI_VIS_PARAMS,
    vis_params.DNDVI_VIS_PARAMS,
    vis_params.FPAR_VIS_PARAMS,
    vis_params.EVI_VIS_PARAMS,
    vis_params.NDWI_VIS_PARAMS,
    vis_params.MNDWI_VIS_PARAMS,
    vis_params.SAVI_VIS_PARAMS,
    vis_params.NDMI_VIS_PARAMS,
    vis_params.NBR_VIS_PARAMS,
    vis_params.NIRV_VIS_PARAMS,
    vis_params.NDRE_VIS_PARAMS,
    vis_params.ELEVATION_VIS_PARAMS,
    vis_params.SLOPE_VIS_PARAMS,
    vis_params.ASPECT_VIS_PARAMS,
    vis_params.HILLSHADE_VIS_PARAMS,
    vis_params.BII_VIS_PARAMS,
    vis_params.BII_CLASS_VIS_PARAMS,
]

RGB_PARAMS = [
    (vis_params.S2_TRUE_COLOR_VIS_PARAMS, S2_BANDS),
    (vis_params.L8_TRUE_COLOR_VIS_PARAMS, L8_BANDS),
    (vis_params.HLS_TRUE_COLOR_VIS_PARAMS, HLS_COMMON_BANDS),
]


@pytest.mark.parametrize("params", PALETTE_PARAMS)
def test_palette_params_well_formed(params):
    assert set(params) == {"min", "max", "palette"}
    assert params["min"] < params["max"]
    assert isinstance(params["palette"], list) and params["palette"]
    assert all(isinstance(color, str) for color in params["palette"])


@pytest.mark.parametrize("params, source_bands", RGB_PARAMS)
def test_rgb_params_well_formed(params, source_bands):
    assert {"bands", "min", "max"} <= set(params)
    assert len(params["bands"]) == 3
    assert params["min"] < params["max"]
    # RGB band names must exist in the sensor's band list in constants.py.
    assert all(band in source_bands for band in params["bands"])


def test_sites_vis_params_is_vector_styling():
    params = vis_params.SITES_VIS_PARAMS
    assert "color" in params
    assert isinstance(params["color"], str)
    assert params["width"] > 0


def test_change_vis_min_max_returns_new_values():
    out = vis_params.change_vis_min_max(vis_params.NDVI_VIS_PARAMS, 0.0, 0.8)
    assert out["min"] == 0.0
    assert out["max"] == 0.8


def test_change_vis_min_max_does_not_mutate_original():
    # The shared module-level preset must be untouched (it's a global).
    original_min = vis_params.NDVI_VIS_PARAMS["min"]
    original_max = vis_params.NDVI_VIS_PARAMS["max"]

    out = vis_params.change_vis_min_max(vis_params.NDVI_VIS_PARAMS, 0.0, 0.8)

    assert vis_params.NDVI_VIS_PARAMS["min"] == original_min
    assert vis_params.NDVI_VIS_PARAMS["max"] == original_max
    assert out is not vis_params.NDVI_VIS_PARAMS


def test_change_vis_min_max_preserves_other_keys():
    out = vis_params.change_vis_min_max(vis_params.NDVI_VIS_PARAMS, 0.0, 0.8)
    assert out["palette"] == vis_params.NDVI_VIS_PARAMS["palette"]
    # Palette is a deep copy, not an alias to the original list.
    assert out["palette"] is not vis_params.NDVI_VIS_PARAMS["palette"]


def test_change_vis_min_max_rejects_min_ge_max():
    with pytest.raises(ValueError, match="must be less than"):
        vis_params.change_vis_min_max(vis_params.NDVI_VIS_PARAMS, 1.0, 1.0)
