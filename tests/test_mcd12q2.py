"""Tests for the MODIS MCD12Q2 phenology builder (day-of-year conversion)."""

from unittest.mock import patch

import pytest

from eetools import constants
from eetools.sensors.modis import preprocessing as modis


def test_mcd12q2_constants():
    assert constants.MODIS_MCD12Q2_COLLECTION == "MODIS/061/MCD12Q2"
    bands = constants.MODIS_MCD12Q2_DATE_BANDS
    assert len(bands) == 14
    assert bands[0] == "Greenup_1" and bands[-1] == "Dormancy_2"
    defaults = constants.MODIS_MCD12Q2_DEFAULT_BANDS
    assert defaults[0] == "NumCycles" and defaults[-2:] == [
        "QA_Overall_1",
        "QA_Overall_2",
    ]


@pytest.mark.parametrize(
    "year, days", [(1970, 0), (1971, 365), (2001, 11323), (2020, 18262), (2021, 18628)]
)
def test_days_since_epoch(year, days):
    assert modis.days_since_epoch(year) == days


def test_get_mcd12q2_rejects_bad_args():
    with pytest.raises(ValueError, match="end_year"):
        modis.get_mcd12q2("AOI", 2021, 2020)
    with pytest.raises(ValueError, match="at least one band"):
        modis.get_mcd12q2("AOI", 2020, 2021, bands=[])


def test_get_mcd12q2_one_image_per_year():
    with (
        patch.object(modis, "ee") as mock_ee,
        patch.object(modis, "validate_collection_year_range") as validate,
        patch.object(modis, "_mcd12q2_year") as per_year,
    ):
        modis.get_mcd12q2("AOI", 2018, 2020, bands=["Greenup_1"])
    validate.assert_called_once_with(
        "MODIS/061/MCD12Q2", "AOI", 2018, 2020, "MODIS MCD12Q2 phenology"
    )
    assert [c.args[1:] for c in per_year.call_args_list] == [
        (2018, ["Greenup_1"]),
        (2019, ["Greenup_1"]),
        (2020, ["Greenup_1"]),
    ]
    (images,), _ = mock_ee.ImageCollection.call_args
    assert images == [per_year.return_value] * 3


@pytest.mark.ee
def test_mcd12q2_year_converts_dates_and_scales(ee_session, first_value):
    """Synthetic: days since 1970 -> DOY of the product year; fill masked; EVI scaled."""
    import ee

    jan_1_2020 = modis.days_since_epoch(2020)
    image = ee.Image.constant(
        [2, jan_1_2020 + 99, jan_1_2020 - 10, constants.MODIS_MCD12Q2_FILL, 5000]
    ).rename(["NumCycles", "Greenup_1", "Greenup_2", "Peak_2", "EVI_Amplitude_1"])
    out = modis._mcd12q2_year(
        image,
        2020,
        ["NumCycles", "Greenup_1", "Greenup_2", "Peak_2", "EVI_Amplitude_1"],
    )
    assert first_value(out, "NumCycles") == 2
    assert first_value(out, "Greenup_1") == 100  # 9 April 2020 = DOY 100
    assert first_value(out, "Greenup_2") == -9  # previous December, kept
    assert first_value(out, "Peak_2") is None  # fill value masked
    assert first_value(out, "EVI_Amplitude_1") == pytest.approx(0.5)
    assert out.get("year").getInfo() == 2020


@pytest.mark.ee
@pytest.mark.slow
def test_get_mcd12q2_real(ee_session):
    import ee

    # A bimodal pixel in central Kenya (2 cycles in 2015 and 2019); many urban and
    # savanna pixels have no detected cycle and are masked.
    point = ee.Geometry.Point(37.4761, -0.8128)
    col = modis.get_mcd12q2(point, 2019, 2020)
    assert col.size().getInfo() == 2
    values = (
        ee.Image(col.first()).reduceRegion(ee.Reducer.first(), point, 500).getInfo()
    )
    raw = (
        ee.Image(
            ee.ImageCollection("MODIS/061/MCD12Q2")
            .filterDate("2019-01-01", "2020-01-01")
            .first()
        )
        .reduceRegion(ee.Reducer.first(), point, 500)
        .getInfo()
    )
    assert values["NumCycles"] == raw["NumCycles"] == 2
    assert values["Greenup_1"] == raw["Greenup_1"] - modis.days_since_epoch(2019) + 1
    assert -200 < values["Greenup_1"] < 550
