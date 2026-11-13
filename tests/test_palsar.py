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
