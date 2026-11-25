"""Tests for the ALOS-2 PALSAR-2 yearly mosaic builder (DN -> gamma0 dB)."""

import math
from unittest.mock import patch

import pytest

from eetools import constants
from eetools.sensors.palsar import preprocessing as palsar


def test_palsar_constants():
    assert constants.PALSAR2_YEARLY_COLLECTION == "JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH"
    assert constants.PALSAR2_CALIBRATION_DB == -83.0
    assert constants.PALSAR2_QA_LAND == 255


def test_get_palsar2_yearly_validates_years_and_builds_per_year():
    with (
        patch.object(palsar, "ee") as mock_ee,
        patch.object(palsar, "validate_collection_year_range") as validate,
    ):
        palsar.get_palsar2_yearly("AOI", 2019, 2021)
    validate.assert_called_once_with(
        constants.PALSAR2_YEARLY_COLLECTION, "AOI", 2019, 2021, "PALSAR-2 yearly mosaic"
    )
    source = mock_ee.ImageCollection.return_value.filterBounds.return_value
    assert [c.args for c in source.filterDate.call_args_list] == [
        ("2019-01-01", "2020-01-01"),
        ("2020-01-01", "2021-01-01"),
        ("2021-01-01", "2022-01-01"),
    ]
    mosaic = source.filterDate.return_value.map.return_value.mosaic.return_value
    assert mosaic.setDefaultProjection.call_count == 3


@pytest.mark.ee
@pytest.mark.parametrize("dn", [1000, 5000])
def test_palsar2_dn_to_db_formula(ee_session, first_value, dn):
    import ee

    image = ee.Image.constant([dn, dn // 2, 255]).rename(["HH", "HV", "qa"])
    out = palsar.palsar2_dn_to_db(image)
    assert first_value(out, "HH") == pytest.approx(
        10 * math.log10(dn**2) - 83.0, abs=1e-4
    )
    assert first_value(out, "HV") == pytest.approx(
        10 * math.log10((dn // 2) ** 2) - 83.0, abs=1e-4
    )


@pytest.mark.ee
@pytest.mark.parametrize("hh, qa", [(0, 255), (3000, 50), (3000, 100), (3000, 150)])
def test_palsar2_dn_to_db_masks_nodata_and_non_land(ee_session, first_value, hh, qa):
    import ee

    image = ee.Image.constant([hh, 1000, qa]).rename(["HH", "HV", "qa"])
    assert first_value(palsar.palsar2_dn_to_db(image), "HH") is None


@pytest.mark.ee
@pytest.mark.slow
def test_get_palsar2_yearly_real(ee_session):
    import ee

    point = ee.Geometry.Point(36.62, -0.40)  # Aberdare forest, Kenya
    col = palsar.get_palsar2_yearly(point, 2020, 2021)
    assert col.size().getInfo() == 2
    image = ee.Image(col.first())
    values = image.reduceRegion(ee.Reducer.first(), point, 25).getInfo()
    assert -25 < values["HV"] < -5 and -20 < values["HH"] < 0  # forest L-band range
    assert values["HV"] < values["HH"]
    assert image.projection().nominalScale().getInfo() == pytest.approx(25, rel=0.1)


# --------------------------------------------------------------------------- #
# PALSAR-2 ScanSAR L2.2
# --------------------------------------------------------------------------- #
def test_scansar_constants():
    assert constants.PALSAR2_SCANSAR_COLLECTION == "JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR"
    assert constants.PALSAR2_SCANSAR_POL_PROPERTY == "Polarizations"
    assert constants.PALSAR2_SCANSAR_QA_VALID == 1
    assert constants.PALSAR2_SCANSAR_LIN_SCALE == 0.01
    assert set(constants.PALSAR2_SCANSAR_DEFAULT_POLS) <= set(
        constants.PALSAR2_SCANSAR_POLARIZATIONS
    )


@pytest.mark.parametrize(
    "pols, match",
    [((), "at least one"), (("HH", "XX"), "Unknown"), (("HV", "HV"), "Duplicate")],
)
def test_get_palsar2_scansar_rejects_bad_polarizations(pols, match):
    with (
        patch.object(palsar, "validate_collection_date_range") as validate,
        pytest.raises(ValueError, match=match),
    ):
        palsar.get_palsar2_scansar(
            "AOI", "2020-01-01", "2021-01-01", polarizations=pols
        )
    validate.assert_not_called()


def test_get_palsar2_scansar_validates_dates_and_filters_each_pol():
    with (
        patch.object(palsar, "ee") as mock_ee,
        patch.object(palsar, "validate_collection_date_range") as validate,
    ):
        palsar.get_palsar2_scansar("AOI", "2020-01-01", "2021-01-01")
    assert validate.call_args.kwargs["collection_id"] == (
        constants.PALSAR2_SCANSAR_COLLECTION
    )
    assert validate.call_args.kwargs["aoi"] == "AOI"
    mock_ee.ImageCollection.assert_called_once_with(
        constants.PALSAR2_SCANSAR_COLLECTION
    )
    assert [c.args for c in mock_ee.Filter.listContains.call_args_list] == [
        ("Polarizations", "HH"),
        ("Polarizations", "HV"),
    ]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"months": (10, 2)}, "year end"),
        ({"months": (0, 5)}, "1-12"),
        ({"months": (6, 13)}, "1-12"),
        ({"months": (6, 9), "reducer": "max"}, "reducer"),
        ({"months": (6, 9), "polarizations": ("HH", "QQ")}, "Unknown"),
    ],
)
def test_palsar2_scansar_composite_rejects_bad_args(kwargs, match):
    with patch.object(palsar, "ee"), pytest.raises(ValueError, match=match):
        palsar.palsar2_scansar_composite("COL", 2025, **kwargs)


def _scansar_scene(ee, hh_dn, hv_dn, msk=1, lin=3500, date="2025-07-01"):
    return (
        ee.Image.constant([hh_dn, hv_dn, lin, msk])
        .rename(["HH", "HV", "LIN", "MSK"])
        .toUint16()
        .set("system:time_start", ee.Date(date).millis())
        .set("Polarizations", ["HH", "HV"])
    )


def _dn_for_db(db):
    """DN whose gamma0 is ``db`` dB (inverse of 20*log10(DN) - 83)."""
    return round(10 ** ((db + 83.0) / 20))


@pytest.mark.ee
def test_scansar_to_gamma0_db_and_linear(ee_session, first_value):
    scene = _scansar_scene(ee_session, 5000, 2000, lin=4891)
    db = palsar.palsar2_scansar_to_gamma0(scene)
    assert first_value(db, "HH") == pytest.approx(20 * math.log10(5000) - 83, abs=1e-4)
    assert first_value(db, "HV") == pytest.approx(20 * math.log10(2000) - 83, abs=1e-4)
    assert first_value(db, "LIN") == pytest.approx(48.91, abs=1e-4)
    assert db.get("backscatter_unit").getInfo() == "dB"
    lin = palsar.palsar2_scansar_to_gamma0(scene, linear=True)
    assert first_value(lin, "HH") == pytest.approx(5000**2 * 10**-8.3, rel=1e-4)
    assert lin.get("backscatter_unit").getInfo() == "linear"
    assert lin.get("system:time_start").getInfo() is not None


@pytest.mark.ee
@pytest.mark.parametrize(
    "msk, kept",
    [(1, True), (2, False), (3, False), (4, False), (5, False), (0b1001, True)],
)
def test_scansar_msk_keeps_only_valid_low_bits(ee_session, first_value, msk, kept):
    out = palsar.palsar2_scansar_to_gamma0(_scansar_scene(ee_session, 3000, 1000, msk))
    assert (first_value(out, "HV") is not None) is kept
    assert (first_value(out, "LIN") is not None) is kept
    unmasked = palsar.palsar2_scansar_to_gamma0(
        _scansar_scene(ee_session, 3000, 1000, msk), mask_quality=False
    )
    assert first_value(unmasked, "HV") is not None


@pytest.mark.ee
def test_scansar_dn_zero_is_masked(ee_session, first_value):
    out = palsar.palsar2_scansar_to_gamma0(_scansar_scene(ee_session, 0, 1000))
    assert first_value(out, "HH") is None
    assert first_value(out, "HV") is not None


@pytest.mark.ee
@pytest.mark.parametrize("linear", [False, True])
def test_scansar_composite_averages_in_linear_power(ee_session, first_value, linear):
    scenes = ee_session.ImageCollection(
        [
            _scansar_scene(ee_session, _dn_for_db(-10), _dn_for_db(-10), lin=3000),
            _scansar_scene(ee_session, _dn_for_db(-20), _dn_for_db(-20), lin=4000),
        ]
    ).map(lambda img: palsar.palsar2_scansar_to_gamma0(img, linear=linear))
    out = palsar.palsar2_scansar_composite(scenes, 2025, (6, 9), reducer="mean")
    expected = 10 * math.log10((0.1 + 0.01) / 2)  # ~ -12.6 dB, not -15 dB
    assert first_value(out, "HV") == pytest.approx(expected, abs=0.01)
    assert first_value(out, "scene_count") == 2
    assert first_value(out, "lin_mean") == pytest.approx(35.0, abs=1e-4)
    assert out.get("n_scenes").getInfo() == 2
    assert out.get("backscatter_unit").getInfo() == "dB"


@pytest.mark.ee
@pytest.mark.parametrize("linear", [False, True])
def test_scansar_composite_single_polarisation(ee_session, first_value, linear):
    # One band (HV only): 10 ** x keeps the band name, so the composite does not fail
    # with "Band pattern 'HV' did not match any bands. Available bands: [constant]".
    scenes = ee_session.ImageCollection(
        [
            _scansar_scene(ee_session, _dn_for_db(-10), _dn_for_db(-10)),
            _scansar_scene(ee_session, _dn_for_db(-20), _dn_for_db(-20)),
        ]
    ).map(lambda img: palsar.palsar2_scansar_to_gamma0(img, ("HV",), linear=linear))
    out = palsar.palsar2_scansar_composite(
        scenes, 2025, (6, 9), polarizations=("HV",), reducer="mean"
    )
    assert out.bandNames().getInfo() == ["HV", "lin_mean", "scene_count"]
    assert first_value(out, "HV") == pytest.approx(10 * math.log10(0.055), abs=0.01)
    assert first_value(out, "scene_count") == 2


@pytest.mark.ee
def test_scansar_composite_window_and_count(ee_session, first_value):
    scenes = ee_session.ImageCollection(
        [
            _scansar_scene(ee_session, 3000, 1000, date="2025-06-15"),
            _scansar_scene(ee_session, 3000, 1000, date="2025-09-30"),
            _scansar_scene(ee_session, 3000, 1000, msk=3, date="2025-08-01"),
            _scansar_scene(ee_session, 3000, 1000, date="2025-10-01"),  # out of window
            _scansar_scene(ee_session, 3000, 1000, date="2024-07-01"),  # other year
        ]
    ).map(palsar.palsar2_scansar_to_gamma0)
    out = palsar.palsar2_scansar_composite(scenes, 2025, (6, 9))
    assert out.get("n_scenes").getInfo() == 3  # shadow scene is in the window
    assert first_value(out, "scene_count") == 2  # but its pixel is not valid
    assert out.get("system:time_start").getInfo() == (
        ee_session.Date("2025-06-01").millis().getInfo()
    )


@pytest.mark.ee
def test_scansar_composite_empty_window(ee_session, first_value):
    scenes = ee_session.ImageCollection(
        [_scansar_scene(ee_session, 3000, 1000, date="2025-02-01")]
    ).map(palsar.palsar2_scansar_to_gamma0)
    out = palsar.palsar2_scansar_composite(scenes, 2025, (6, 9))
    assert out.bandNames().getInfo() == ["HH", "HV", "lin_mean", "scene_count"]
    assert first_value(out, "scene_count") == 0
    assert first_value(out, "HV") is None
    assert out.get("n_scenes").getInfo() == 0


@pytest.mark.ee
@pytest.mark.slow
def test_get_palsar2_scansar_rejects_dates_outside_record(ee_session):
    import ee

    point = ee.Geometry.Point(37.24, -1.10)
    with pytest.raises(ValueError):
        palsar.get_palsar2_scansar(point, "2030-01-01", "2031-01-01")


@pytest.mark.ee
@pytest.mark.slow
def test_scansar_dry_window_composite_real(ee_session):
    import ee

    point = ee.Geometry.Point(37.24, -1.10)  # Ecosia project area, Kenya
    col = palsar.get_palsar2_scansar(point, "2025-01-01", "2025-10-01")
    out = palsar.palsar2_scansar_composite(col, 2025, (6, 9))
    values = out.reduceRegion(ee.Reducer.first(), point, 25).getInfo()
    assert values["scene_count"] >= 1
    assert -30 < values["HV"] < -5 and -25 < values["HH"] < 0
    assert values["HV"] < values["HH"]
    assert 15 < values["lin_mean"] < 70
    assert out.projection().nominalScale().getInfo() == pytest.approx(25, rel=0.1)
