"""Tests for the ESA CCI Biomass v6 AGB and change builders."""

from unittest.mock import patch

import pytest

from eetools import constants
from eetools.sensors.esa_cci_biomass import preprocessing as cci


def test_cci_constants():
    assert constants.ESA_CCI_AGB_COLLECTION.endswith("/ESA/ESA_CCI_AGB")
    assert constants.ESA_CCI_AGB_BANDS == ["AGB", "SD"]
    assert constants.ESA_CCI_AGB_YEARS == (2007, 2010, *range(2015, 2023))
    assert len(constants.ESA_CCI_AGB_DIFF_PAIRS) == 9
    assert all(s < e for s, e in constants.ESA_CCI_AGB_DIFF_PAIRS)


@pytest.mark.parametrize("years, match", [([], "at least one"), ([2012], r"\[2012\]")])
def test_get_esa_cci_agb_rejects_bad_years(years, match):
    with pytest.raises(ValueError, match=match):
        cci.get_esa_cci_agb("AOI", years)


def test_get_esa_cci_agb_one_image_per_year():
    with (
        patch.object(cci, "ee") as mock_ee,
        patch.object(cci, "_agb_year") as per_year,
    ):
        cci.get_esa_cci_agb("AOI", [2010, 2020])
        assert [c.args for c in per_year.call_args_list] == [
            ("AOI", 2010),
            ("AOI", 2020),
        ]
        cci.get_esa_cci_agb("AOI")
    assert per_year.call_count == 2 + len(constants.ESA_CCI_AGB_YEARS)
    (images,), _ = mock_ee.ImageCollection.call_args
    assert len(images) == len(constants.ESA_CCI_AGB_YEARS)


@pytest.mark.parametrize("pair", [(2016, 2018), (2010, 2007), (2010, 2015)])
def test_get_esa_cci_agb_change_rejects_unpublished_pairs(pair):
    with pytest.raises(ValueError, match="No published"):
        cci.get_esa_cci_agb_change("AOI", *pair)


def test_get_esa_cci_agb_change_selects_diff_image_by_name():
    with (
        patch.object(cci, "ee") as mock_ee,
        patch.object(cci, "_agb_year") as per_year,
    ):
        cci.get_esa_cci_agb_change("AOI", 2010, 2020)
    assert [c.args for c in per_year.call_args_list] == [("AOI", 2020), ("AOI", 2010)]
    mock_ee.Filter.eq.assert_called_once_with(
        "system:index", "CCI_BIOMASS_DIFF_AGB_100m_2020_2010"
    )
    mock_ee.ImageCollection.assert_called_once_with(
        constants.ESA_CCI_AGB_DIFF_COLLECTION
    )
    mock_ee.Image.cat.return_value.rename.assert_called_once_with(
        ["agb_change", "agb_change_sd", "agb_change_qf"]
    )


@pytest.mark.ee
@pytest.mark.slow
def test_cci_agb_and_change_real(ee_session):
    """Real data over a forest point: change equals the AGB difference."""
    import ee

    point = ee.Geometry.Point(36.62, -0.40)  # Aberdare forest, Kenya
    agb = cci.get_esa_cci_agb(point, [2019, 2020])
    assert agb.size().getInfo() == 2
    first = ee.Image(agb.first())
    assert first.bandNames().getInfo() == ["AGB", "SD"]
    assert first.get("year").getInfo() == 2019
    change = cci.get_esa_cci_agb_change(point, 2019, 2020)
    values = (
        change.addBands(ee.Image(agb.toList(2).get(0)).select(["AGB"], ["a19"]))
        .addBands(ee.Image(agb.toList(2).get(1)).select(["AGB"], ["a20"]))
        .reduceRegion(ee.Reducer.first(), point, 100)
        .getInfo()
    )
    assert values["a19"] > 0
    assert values["agb_change"] == pytest.approx(values["a20"] - values["a19"])
    assert values["agb_change_sd"] is not None and values["agb_change_sd"] >= 0
    assert values["agb_change_qf"] is not None
