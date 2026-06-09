"""Tests for the Biodiversity Intactness Index (BII) module.

The validation guards run without Earth Engine (they raise before any ``ee`` call).
The selection/clip behaviour is confirmed against the real sat-io BII asset over a
sub-Saharan AOI, so those tests are marked ``slow``.
"""

import pytest


# --------------------------------------------------------------------------- #
# Pure validation (no Earth Engine)
# --------------------------------------------------------------------------- #
def test_get_bii_rejects_unknown_band():
    from eetools.sensors.bii.preprocessing import get_bii

    with pytest.raises(ValueError, match="Unknown BII band"):
        get_bii(aoi=None, bands=["BII All", "Not A Band"])


def test_get_bii_rejects_empty_bands():
    from eetools.sensors.bii.preprocessing import get_bii

    with pytest.raises(ValueError, match="At least one band"):
        get_bii(aoi=None, bands=[])


def test_get_bii_image_rejects_unknown_resolution():
    from eetools.sensors.bii.preprocessing import get_bii_image

    with pytest.raises(ValueError, match="Invalid resolution"):
        get_bii_image(resolution="500m")


# --------------------------------------------------------------------------- #
# Real-asset integration (sub-Saharan AOI)
# --------------------------------------------------------------------------- #
@pytest.mark.ee
@pytest.mark.slow
def test_get_bii_single_band_image(ee_session, small_aoi):
    from eetools.sensors.bii.preprocessing import get_bii

    img = get_bii(aoi=small_aoi, bands=["BII All"])
    assert img.bandNames().getInfo() == ["BII All"]


@pytest.mark.ee
@pytest.mark.slow
def test_get_bii_multiband_preserves_order(ee_session, small_aoi):
    from eetools.sensors.bii.preprocessing import get_bii

    img = get_bii(aoi=small_aoi, bands=["BII Mammals", "BII Birds"])
    assert img.bandNames().getInfo() == ["BII Mammals", "BII Birds"]


@pytest.mark.ee
@pytest.mark.slow
def test_get_bii_image_band_set(ee_session):
    from eetools.constants import BII_PROCESSED_BANDS
    from eetools.sensors.bii.preprocessing import get_bii_image

    img = get_bii_image(resolution="8km")
    assert img.bandNames().getInfo() == BII_PROCESSED_BANDS
