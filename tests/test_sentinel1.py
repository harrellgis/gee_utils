"""Tests for the Sentinel-1 GRD masking and preprocessing modules.

Edge masking is checked on constant images (simple per-band threshold). The
focal-median speckle filter is checked structurally (band names + time property
preserved) rather than by reducing values, since focal ops over a projection-less
constant image are best not evaluated. The real-collection builder is marked slow.
"""

import pytest

pytestmark = pytest.mark.ee


def test_s1_mask_edges_drops_low_backscatter(ee_session, first_value):
    from eetools.sensors.sentinel1.masking import mask_edges

    low = ee_session.Image.constant(-40).rename("VV")
    high = ee_session.Image.constant(-12).rename("VV")
    # Default threshold is -30 dB: -40 is masked out, -12 is kept.
    assert first_value(mask_edges(low), "VV") is None
    assert first_value(mask_edges(high), "VV") == pytest.approx(-12)


def test_s1_mask_edges_custom_threshold(ee_session, first_value):
    from eetools.sensors.sentinel1.masking import mask_edges

    img = ee_session.Image.constant(-25).rename("VV")
    # -25 survives the default -30 threshold but not a -20 threshold.
    assert first_value(mask_edges(img), "VV") == pytest.approx(-25)
    assert first_value(mask_edges(img, edge_threshold_db=-20), "VV") is None


def test_s1_speckle_filter_preserves_bands_and_time(ee_session):
    from eetools.sensors.sentinel1.masking import apply_speckle_filter

    img = ee_session.Image.cat(
        [
            ee_session.Image.constant(-12).rename("VV"),
            ee_session.Image.constant(-18).rename("VH"),
        ]
    ).set("system:time_start", ee_session.Date("2024-01-01").millis())
    out = ee_session.Image(apply_speckle_filter(img))
    assert out.bandNames().getInfo() == ["VV", "VH"]
    assert out.get("system:time_start").getInfo() is not None


@pytest.mark.slow
def test_get_s1_grd_collection_real(ee_session, small_aoi):
    from eetools.sensors.sentinel1.preprocessing import get_s1_grd_collection

    col = get_s1_grd_collection(
        small_aoi,
        ee_session.Date("2024-01-01"),
        ee_session.Date("2024-03-01"),
    )
    assert isinstance(col, ee_session.ImageCollection)
    if col.size().getInfo() == 0:
        pytest.skip("no Sentinel-1 scenes for the AOI/window")
    # Only the requested dual-pol bands survive the homogeneous filter + select.
    assert set(col.first().bandNames().getInfo()) == {"VV", "VH"}
