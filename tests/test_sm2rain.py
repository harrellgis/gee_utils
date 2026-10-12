"""Tests for the OpenLandMap SM2RAIN precipitation climatology module.

Month validation runs without Earth Engine; the real-image builder is marked
``ee`` + ``slow``.
"""

import pytest

# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("months", [[0], [13], [1, 6, 14]])
def test_get_sm2rain_climatology_rejects_invalid_months(months):
    from eetools.sensors.sm2rain.preprocessing import get_sm2rain_climatology

    with pytest.raises(ValueError, match="months must be integers 1-12"):
        get_sm2rain_climatology(months)


# --------------------------------------------------------------------------- #
# Real-image builder                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.ee
@pytest.mark.slow
def test_get_sm2rain_climatology_real(ee_session, first_value):
    from eetools.constants import SM2RAIN_CLIMATOLOGY_IMAGE
    from eetools.sensors.sm2rain.preprocessing import get_sm2rain_climatology

    col = get_sm2rain_climatology()
    assert col.size().getInfo() == 12
    assert col.aggregate_array("month").getInfo() == list(range(1, 13))
    april = ee_session.Image(col.filter(ee_session.Filter.eq("month", 4)).first())
    assert april.bandNames().getInfo() == ["precipitation"]
    assert april.get("month_name").getInfo() == "apr"
    # The renamed band carries the source 'apr' band's values unchanged.
    raw = ee_session.Image(SM2RAIN_CLIMATOLOGY_IMAGE)
    assert first_value(april, "precipitation", scale=1000) == first_value(
        raw, "apr", scale=1000
    )


@pytest.mark.ee
@pytest.mark.slow
def test_get_sm2rain_climatology_subset_real(ee_session):
    from eetools.sensors.sm2rain.preprocessing import get_sm2rain_climatology

    col = get_sm2rain_climatology([12, 1, 2])
    assert col.aggregate_array("month_name").getInfo() == ["dec", "jan", "feb"]
