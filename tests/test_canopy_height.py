"""Tests for the Meta/WRI 1m Global Canopy Height module.

``get_canopy_height`` and the real-data woody-fraction check construct the real
canopy-height mosaic, so they are marked ``slow``. The woody-fraction graph is
checked with mocks, and its maths on a synthetic 1 m CHM under the ``ee`` marker.
"""

from unittest.mock import MagicMock, patch

import pytest

from eetools.sensors.canopy_height import preprocessing as chm_mod

TRANSFORM = [10, 0, 500000, 0, -10, 9500000]


# --------------------------------------------------------------------------- #
# Pure / mock                                                                  #
# --------------------------------------------------------------------------- #
def test_woody_fraction_band_names():
    assert chm_mod.woody_fraction_band_names([2, 3, 5]) == [
        "woody_frac_h2",
        "woody_frac_h3",
        "woody_frac_h5",
    ]
    assert chm_mod.woody_fraction_band_names([2.5, 0.75, 10.0]) == [
        "woody_frac_h2p5",
        "woody_frac_h0p75",
        "woody_frac_h10",
    ]


@pytest.fixture
def mock_ee():
    with patch.object(chm_mod, "ee") as mock:
        yield mock


def test_woody_fraction_graph_uses_max_pixels_and_transform(mock_ee):
    chm = mock_ee.Image.return_value.select.return_value
    thresholded = {}
    chm.gte.side_effect = lambda h: thresholded.setdefault(h, MagicMock())

    out = chm_mod.woody_fraction_from_chm(
        "CHM", heights_m=(2, 2.5), crs="EPSG:32737", crs_transform=TRANSFORM
    )

    assert sorted(thresholded) == [2.0, 2.5]
    for h, name in [(2.0, "woody_frac_h2"), (2.5, "woody_frac_h2p5")]:
        as_float = thresholded[h].toFloat.return_value
        as_float.reduceResolution.assert_called_once_with(
            reducer=mock_ee.Reducer.mean.return_value, maxPixels=256
        )
        reproject = as_float.reduceResolution.return_value.reproject
        reproject.assert_called_once_with(
            crs="EPSG:32737", crsTransform=[10.0, 0.0, 500000.0, 0.0, -10.0, 9500000.0]
        )
        reproject.return_value.rename.assert_called_once_with(name)

    valid = chm.mask.return_value.gt.return_value.toFloat.return_value
    chm.mask.return_value.gt.assert_called_once_with(0)
    valid.reduceResolution.return_value.reproject.return_value.rename.assert_called_once_with(
        "chm_valid_frac"
    )
    (parts,), _ = mock_ee.Image.cat.call_args
    assert len(parts) == 3
    assert out is mock_ee.Image.cat.return_value.toFloat.return_value


def test_woody_fraction_scale_and_custom_max_pixels(mock_ee):
    chm = mock_ee.Image.return_value.select.return_value
    chm_mod.woody_fraction_from_chm(
        "CHM", heights_m=(3,), crs="EPSG:32737", scale=30, max_pixels=1024
    )
    reduced = chm.gte.return_value.toFloat.return_value.reduceResolution
    assert reduced.call_args.kwargs["maxPixels"] == 1024
    reduced.return_value.reproject.assert_called_once_with(crs="EPSG:32737", scale=30)


def test_woody_fraction_defaults_to_meta_chm(mock_ee):
    with patch.object(chm_mod, "get_canopy_height", return_value="META") as meta:
        chm_mod.woody_fraction_from_chm(crs="EPSG:32737")
    # The mosaic is pinned to its tiles' native projection before aggregating.
    meta.assert_called_once_with(pin_projection=True)
    mock_ee.Image.assert_any_call("META")
    chm = mock_ee.Image.return_value.select.return_value
    assert [c.args[0] for c in chm.gte.call_args_list] == [2.0, 3.0, 5.0]


def test_get_canopy_height_pin_projection(mock_ee):
    collection = mock_ee.ImageCollection.return_value
    mosaic = collection.mosaic.return_value.select.return_value
    assert chm_mod.get_canopy_height() is mosaic
    mosaic.setDefaultProjection.assert_not_called()

    out = chm_mod.get_canopy_height(pin_projection=True)
    collection.first.return_value.select.assert_called_with("cover_code")
    mosaic.setDefaultProjection.assert_called_once_with(
        collection.first.return_value.select.return_value.projection.return_value
    )
    assert out is mosaic.setDefaultProjection.return_value


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"heights_m": ()}, "at least one"),
        ({"heights_m": (2, 0)}, "> 0"),
        ({"heights_m": (2, 2.0)}, "duplicates"),
        ({"crs_transform": [10, 0, 0]}, "6 elements"),
        ({"scale": 0}, "> 0"),
        ({"max_pixels": 0}, "> 0"),
    ],
)
def test_woody_fraction_validation(mock_ee, kwargs, match):
    with pytest.raises(ValueError, match=match):
        chm_mod.woody_fraction_from_chm("CHM", crs="EPSG:32737", **kwargs)
    mock_ee.Image.cat.assert_not_called()


# --------------------------------------------------------------------------- #
# Live: synthetic 1 m CHM                                                      #
# --------------------------------------------------------------------------- #
def _sample(ee, image, x, y):
    point = ee.Geometry.Point([x, y], "EPSG:32737")
    return image.reduceRegion(
        ee.Reducer.first(), point, crs="EPSG:32737", crsTransform=TRANSFORM
    ).getInfo()


@pytest.mark.ee
def test_woody_fraction_synthetic_exact_values(ee_session):
    ee = ee_session
    fine = ee.Projection("EPSG:32737").atScale(1)
    coords = ee.Image.pixelCoordinates(fine)
    # Height = x mod 10 (0..9 m) -> each 10 m output pixel holds every height once
    # per 1 m column: shares 0.8 (>= 2 m), 0.7 (>= 3 m), 0.5 (>= 5 m).
    chm = coords.select("x").floor().mod(10).rename("height").reproject(fine)
    # Mask the southern half (y mod 10 < 5) of every output pixel: valid share 0.5,
    # fractions unchanged because they are over valid pixels only.
    half = coords.select("y").floor().mod(10).gte(5)
    masked = chm.updateMask(half)

    full = chm_mod.woody_fraction_from_chm(
        chm, crs="EPSG:32737", crs_transform=TRANSFORM
    )
    part = chm_mod.woody_fraction_from_chm(
        masked, crs="EPSG:32737", crs_transform=TRANSFORM
    )
    expected = {"woody_frac_h2": 0.8, "woody_frac_h3": 0.7, "woody_frac_h5": 0.5}

    values = _sample(ee, full, 500005, 9499995)
    assert values == pytest.approx({**expected, "chm_valid_frac": 1.0})
    values = _sample(ee, part, 500015, 9499985)
    assert values == pytest.approx({**expected, "chm_valid_frac": 0.5})
    assert full.bandNames().getInfo() == [*expected, "chm_valid_frac"]
    assert {t["precision"] for t in full.bandTypes().getInfo().values()} == {"float"}


# --------------------------------------------------------------------------- #
# Live: real Meta CHM                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.ee
@pytest.mark.slow
def test_get_canopy_height_band(ee_session, small_aoi):
    from eetools.sensors.canopy_height.preprocessing import get_canopy_height

    img = get_canopy_height(aoi=small_aoi)
    assert img.bandNames().getInfo() == ["canopy_height"]


@pytest.mark.ee
@pytest.mark.slow
def test_woody_fraction_real_meta_bounds_and_ordering(ee_session):
    ee = ee_session
    # 200 m x 200 m in coastal Kwale (UTM 37S), 20 x 20 output pixels at 10 m.
    region = ee.Geometry.Rectangle(
        [523000, 9525800, 523200, 9526000], "EPSG:32737", False
    )
    transform = [10, 0, 523000, 0, -10, 9526000]
    frac = chm_mod.woody_fraction_from_chm(crs="EPSG:32737", crs_transform=transform)
    check = ee.Image.cat(
        [
            frac.reduce(ee.Reducer.min()).rename("min"),
            frac.reduce(ee.Reducer.max()).rename("max"),
            frac.select("woody_frac_h5")
            .subtract(frac.select("woody_frac_h2"))
            .rename("h5_minus_h2"),
        ]
    )
    stats = check.reduceRegion(
        ee.Reducer.minMax(), region, crs="EPSG:32737", crsTransform=transform
    ).getInfo()
    assert stats["min_min"] >= 0
    assert stats["max_max"] <= 1
    assert stats["h5_minus_h2_max"] <= 1e-6


# --------------------------------------------------------------------------- #
# E9c: ETH 2020 and GLAD 2019 static canopy heights                            #
# --------------------------------------------------------------------------- #
def test_eth_glad_constants():
    from eetools import constants

    assert constants.ETH_CANOPY_HEIGHT_2020 == (
        "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1"
    )
    assert constants.ETH_CANOPY_HEIGHT_SD_2020 == (
        "users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1"
    )
    assert constants.ETH_CANOPY_HEIGHT_NODATA == 255
    assert constants.GLAD_CANOPY_HEIGHT_2019 == (
        "projects/sat-io/open-datasets/GLAD/GEDI_V27"
    )
    assert constants.GLAD_CANOPY_HEIGHT_MAX_M == 60


def test_get_eth_canopy_height_masks_nodata_and_names_bands(mock_ee):
    chm_mod.get_eth_canopy_height_2020()
    assert [c.args for c in mock_ee.Image.call_args_list] == [
        ("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1",),
        ("users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1",),
    ]
    selected = mock_ee.Image.return_value.select
    assert [c.args for c in selected.call_args_list] == [
        (["b1"], ["canopy_height"]),
        (["b1"], ["canopy_height_sd"]),
    ]
    selected.return_value.neq.assert_called_with(255)
    mock_ee.Image.cat.return_value.clip.assert_not_called()
    chm_mod.get_eth_canopy_height_2020("AOI")
    mock_ee.Image.cat.return_value.clip.assert_called_once_with("AOI")


@pytest.mark.parametrize("pin", [False, True])
def test_get_glad_canopy_height_masks_codes_and_pins(mock_ee, pin):
    chm_mod.get_glad_canopy_height_2019(pin_projection=pin)
    collection = mock_ee.ImageCollection.return_value
    mock_ee.ImageCollection.assert_called_once_with(
        "projects/sat-io/open-datasets/GLAD/GEDI_V27"
    )
    height = collection.mosaic.return_value.select.return_value
    collection.mosaic.return_value.select.assert_called_once_with(
        ["b1"], ["canopy_height"]
    )
    height.lte.assert_called_once_with(60)
    pinned = height.updateMask.return_value.setDefaultProjection
    assert pinned.called is pin


@pytest.mark.ee
@pytest.mark.parametrize(
    "value, kept", [(0, True), (25, True), (101, False), (103, False)]
)
def test_glad_code_mask_synthetic(ee_session, first_value, value, kept):
    import ee

    with patch.object(chm_mod.ee, "ImageCollection") as col:
        col.return_value.mosaic.return_value = ee.Image.constant(value).rename("b1")
        image = chm_mod.get_glad_canopy_height_2019()
    got = first_value(image, "canopy_height")
    assert (got == value) if kept else got is None


@pytest.mark.ee
@pytest.mark.slow
def test_eth_and_glad_real_forest_point(ee_session):
    """Real data at a montane forest point: plausible heights, codes masked."""
    import ee

    point = ee.Geometry.Point(36.62, -0.40)  # Aberdare forest, Kenya
    eth = chm_mod.get_eth_canopy_height_2020(point)
    glad = chm_mod.get_glad_canopy_height_2019(point, pin_projection=True)
    eth_v = eth.reduceRegion(ee.Reducer.first(), point, 10).getInfo()
    glad_v = glad.reduceRegion(ee.Reducer.first(), point, 30).getInfo()
    assert 0 < eth_v["canopy_height"] <= 60 and eth_v["canopy_height_sd"] >= 0
    assert 0 <= glad_v["canopy_height"] <= 60
    assert glad.projection().nominalScale().getInfo() == pytest.approx(30, rel=0.1)
    assert eth.bandNames().getInfo() == ["canopy_height", "canopy_height_sd"]
