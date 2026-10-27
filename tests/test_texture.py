"""Tests for eetools.sensors.texture.

Pure tests cover the radius conversion, metric/stat validation, value-range resolution
and band naming; they need no Earth Engine session. The ``ee`` tests run texture on
synthetic images on a fixed 10 m UTM grid (constant, checkerboard, ramp, random), so
the GLCM identities and masking behaviour are checked against known answers.
"""

import pytest

from eetools.constants import TEXTURE_MAX_RADIUS
from eetools.sensors.texture import (
    glcm_radius,
    quantize_band,
    texture_band_name,
    validate_first_order_stats,
    validate_glcm_metrics,
)

CRS = "EPSG:32737"  # UTM 37S, matching the conftest point region
SCALE = 10
COL0, ROW0 = (
    50_000,
    960_000,
)  # an arbitrary grid cell (easting 500 km, northing 9600 km)


# --------------------------------------------------------------------------- #
# Pure tests                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("window_m", "scale_m", "expected"),
    [(750, 10, 37), (750, 30, 12), (750, 250, 1), (50, 10, 2), (250, 10, 12)],
)
def test_glcm_radius(window_m, scale_m, expected):
    assert glcm_radius(window_m, scale_m) == expected


@pytest.mark.parametrize(
    ("window_m", "scale_m"),
    [(10, 10), (20, 10), ((2 * TEXTURE_MAX_RADIUS + 3) * 10, 10), (0, 10), (50, 0)],
)
def test_glcm_radius_out_of_bounds_raises(window_m, scale_m):
    with pytest.raises(ValueError):
        glcm_radius(window_m, scale_m)


def test_validate_glcm_metrics_defaults_pass():
    assert validate_glcm_metrics(("idm", "ent")) == ("idm", "ent")


@pytest.mark.parametrize(
    ("metrics", "match"),
    [
        (("idm", "bogus"), "Unknown"),
        (("maxcorr",), "not computed"),
        (("contrast", "inertia"), "same quantity"),
        (("idm", "idm"), "Duplicate"),
    ],
)
def test_validate_glcm_metrics_rejects(metrics, match):
    with pytest.raises(ValueError, match=match):
        validate_glcm_metrics(metrics)


def test_validate_first_order_stats():
    assert validate_first_order_stats(("mean", "sd", "ent1")) == ("mean", "sd", "ent1")
    with pytest.raises(ValueError, match="Unknown"):
        validate_first_order_stats(("var",))


def test_quantize_band_no_default_range_raises():
    # Validation happens before any ee call, so no image is needed.
    with pytest.raises(ValueError, match="No default quantization range"):
        quantize_band(None, "B4")


@pytest.mark.parametrize("levels", [8, 23, 257, 1024])
def test_quantize_band_levels_out_of_bounds_raises(levels):
    with pytest.raises(ValueError, match="levels"):
        quantize_band(None, "NDVI", levels=levels)


def test_quantize_band_inverted_range_raises():
    with pytest.raises(ValueError, match="lo < hi"):
        quantize_band(None, "NDVI", value_range=(1.0, 0.0))


def test_texture_band_name():
    assert texture_band_name("EVI", "idm", 2) == "EVI_idm_r2"


def test_calc_texture_validation_before_ee():
    from eetools.sensors.texture import calc_texture

    with pytest.raises(ValueError, match="at least one"):
        calc_texture(None, "EVI", 50, 10, glcm_metrics=(), first_order=())
    with pytest.raises(ValueError, match="duplicate radii"):
        calc_texture(None, "EVI", [50, 55], 10)
    with pytest.raises(ValueError, match="No default quantization range"):
        calc_texture(None, "B4", 50, 10)


# --------------------------------------------------------------------------- #
# Earth Engine tests (synthetic images on a fixed 10 m grid)                   #
# --------------------------------------------------------------------------- #
@pytest.fixture
def proj(ee_session):
    return ee_session.Projection(CRS).atScale(SCALE)


@pytest.fixture
def grid_col(ee_session, proj):
    """Grid column index (proj is atScale(SCALE), so pixelCoordinates are grid units)."""
    return ee_session.Image.pixelCoordinates(proj).select("x").floor()


@pytest.fixture
def value_at(ee_session):
    """Return a helper reading one band at the centre of grid cell (col, row)."""

    def _value_at(image, band, col=COL0, row=ROW0):
        pt = ee_session.Geometry.Point([(col + 0.5) * SCALE, (row + 0.5) * SCALE], CRS)
        return (
            ee_session.Image(image)
            .select(band)
            .reduceRegion(ee_session.Reducer.first(), pt, crs=CRS, scale=SCALE)
            .get(band)
            .getInfo()
        )

    return _value_at


@pytest.fixture
def ndvi_ramp(ee_session, proj, grid_col):
    """Float NDVI ramp in [0, 1) repeating every 100 columns."""
    return grid_col.mod(100).divide(100).toFloat().rename("NDVI")


@pytest.mark.ee
def test_quantize_band_clamps(ee_session, first_value):
    hi = quantize_band(ee_session.Image.constant(1.5).rename("NDVI"), "NDVI")
    lo = quantize_band(ee_session.Image.constant(-0.3).rename("NDVI"), "NDVI")
    assert first_value(hi, "NDVI") == 255
    assert first_value(lo, "NDVI") == 0


@pytest.mark.ee
def test_constant_image_haralick_identities(ee_session, proj, value_at):
    from eetools.sensors.texture import calc_glcm_texture

    img = ee_session.Image.constant(0.5).rename("NDVI").reproject(proj)
    tex = calc_glcm_texture(img, "NDVI", 2, metrics=("idm", "ent", "contrast", "asm"))
    assert value_at(tex, "NDVI_idm_r2") == pytest.approx(1)
    assert value_at(tex, "NDVI_ent_r2") == pytest.approx(0)
    assert value_at(tex, "NDVI_contrast_r2") == pytest.approx(0)
    assert value_at(tex, "NDVI_asm_r2") == pytest.approx(1)


@pytest.mark.ee
def test_checkerboard_has_contrast(ee_session, proj, value_at):
    from eetools.sensors.texture import calc_glcm_texture

    xy = ee_session.Image.pixelCoordinates(proj).floor()
    board = (
        xy.select("x").add(xy.select("y")).mod(2).multiply(255).toUint8().rename("B")
    )
    tex = calc_glcm_texture(
        board, "B", 1, metrics=("idm", "contrast"), prequantized=True
    )
    assert value_at(tex, "B_contrast_r1") > 0
    assert value_at(tex, "B_idm_r1") < 1


@pytest.mark.ee
def test_float_input_regression(ee_session, ndvi_ramp, value_at):
    from eetools.sensors.texture import calc_glcm_texture

    # Raw glcmTexture rejects float input outright...
    with pytest.raises(ee_session.EEException, match="integer"):
        value_at(ndvi_ramp.glcmTexture(size=2), "NDVI_ent")
    # ...while calc_glcm_texture quantizes first and returns real texture.
    tex = calc_glcm_texture(ndvi_ramp, "NDVI", 2)
    assert value_at(tex, "NDVI_ent_r2") > 0


@pytest.mark.ee
def test_masked_pixels_do_not_spread(ee_session, proj, grid_col, value_at):
    """Masking one column leaves only that column masked; neighbours within the
    radius change (fewer pairs), neighbours beyond it are untouched."""
    from eetools.sensors.texture import calc_glcm_texture

    base = (
        ee_session.Image.random(seed=42)
        .reproject(proj)
        .multiply(64)
        .floor()
        .add(1)
        .toUint8()
        .rename("B")
    )
    masked = base.updateMask(grid_col.neq(COL0))
    tex_base = calc_glcm_texture(base, "B", 2, prequantized=True)
    tex_masked = calc_glcm_texture(masked, "B", 2, prequantized=True)

    assert value_at(tex_masked, "B_ent_r2", col=COL0) is None
    near_b = value_at(tex_base, "B_ent_r2", col=COL0 + 1)
    near_m = value_at(tex_masked, "B_ent_r2", col=COL0 + 1)
    assert near_m is not None and near_m != pytest.approx(near_b)
    far_b = value_at(tex_base, "B_ent_r2", col=COL0 + 3)
    assert value_at(tex_masked, "B_ent_r2", col=COL0 + 3) == pytest.approx(far_b)


@pytest.mark.ee
def test_first_order_texture(ee_session, proj, value_at):
    from eetools.sensors.texture import calc_first_order_texture

    img = ee_session.Image.constant(0.4).rename("NDVI").reproject(proj)
    tex = calc_first_order_texture(img, "NDVI", 1, stats=("mean", "sd", "ent1"))
    assert tex.bandNames().getInfo() == ["NDVI_mean_r1", "NDVI_sd_r1", "NDVI_ent1_r1"]
    assert value_at(tex, "NDVI_mean_r1") == pytest.approx(0.4)
    assert value_at(tex, "NDVI_sd_r1") == pytest.approx(0, abs=1e-6)
    assert value_at(tex, "NDVI_ent1_r1") == pytest.approx(0, abs=1e-6)


@pytest.mark.ee
def test_calc_texture_multiscale_names_and_properties(ee_session, ndvi_ramp):
    from eetools.sensors.texture import calc_texture

    img = ndvi_ramp.rename("EVI").set(
        "system:time_start", ee_session.Date("2024-06-01").millis()
    )
    tex = calc_texture(img, ["EVI"], window_m=[50, 250], scale_m=10)
    info = tex.getInfo()
    assert [b["id"] for b in info["bands"]] == [
        "EVI_idm_r2",
        "EVI_ent_r2",
        "EVI_idm_r12",
        "EVI_ent_r12",
    ]
    props = info["properties"]
    assert (
        props["system:time_start"] == ee_session.Date("2024-06-01").millis().getInfo()
    )
    assert props["texture_window_m"] == [50, 250]
    assert props["texture_radius_px"] == [2, 12]
    assert props["texture_scale_m"] == 10
    assert props["texture_levels"] == 256
    assert props["texture_value_ranges"] == {"EVI": [0, 1]}
    assert props["texture_pinned"] is False


@pytest.mark.ee
def test_calc_texture_pin_scale_matches_at_scale_m(ee_session, ndvi_ramp, value_at):
    from eetools.sensors.texture import calc_texture

    unpinned = calc_texture(ndvi_ramp, "NDVI", 50, 10)
    pinned = calc_texture(ndvi_ramp, "NDVI", 50, 10, pin_scale=True, crs=CRS)
    assert value_at(pinned, "NDVI_ent_r2") == pytest.approx(
        value_at(unpinned, "NDVI_ent_r2")
    )
    assert pinned.get("texture_pinned").getInfo() is True


@pytest.mark.ee
def test_add_texture_to_collection(ee_session, ndvi_ramp):
    from eetools.sensors.texture import add_texture_to_collection

    col = ee_session.ImageCollection([ndvi_ramp, ndvi_ramp])
    out = add_texture_to_collection(col, bands="NDVI", window_m=50, scale_m=10)
    assert out.first().bandNames().getInfo() == ["NDVI", "NDVI_idm_r2", "NDVI_ent_r2"]
