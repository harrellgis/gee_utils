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
