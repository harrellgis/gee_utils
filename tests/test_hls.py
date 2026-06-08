"""Tests for the HLS masking and preprocessing modules.

Fmask bit logic and band harmonization are verified on synthetic constant
images. The merged-collection builder that hits the real HLS archive is
marked ``slow``.
"""

import pytest

pytestmark = pytest.mark.ee


def _fmask_image(ee, fmask_value):
    return ee.Image.cat(
        [
            ee.Image.constant(fmask_value).toInt().rename("Fmask"),
            ee.Image.constant(0.4).rename("NIR"),
        ]
    )


@pytest.mark.parametrize(
    "fmask_value, expected",
    [
        (0, 0),  # clear
        (1 << 1, 1),  # cloud bit
        (1 << 3, 1),  # cloud-shadow bit
        (1 << 2, 1),  # adjacent (HLS_MASK_ADJACENT is True by default)
        (1 << 4, 1),  # snow (HLS_MASK_SNOW is True)
        (1 << 5, 0),  # water (HLS_MASK_WATER_IN_QA is False -> not flagged)
        (3 << 6, 1),  # high aerosol (HLS_MASK_HIGH_AEROSOL is True)
        (2 << 6, 0),  # moderate aerosol (HLS_MASK_MODERATE_AEROSOL is False)
    ],
)
def test_hls_fmask_cloud_mask_bits(ee_session, first_value, fmask_value, expected):
    from eetools.sensors.hls.masking import add_fmask_cloud_mask

    out = add_fmask_cloud_mask(_fmask_image(ee_session, fmask_value))
    assert "cloudmask" in out.bandNames().getInfo()
    assert first_value(out, "cloudmask") == expected


def test_hls_apply_cld_shdw_mask(ee_session, first_value):
    from eetools.sensors.hls.masking import add_fmask_cloud_mask, apply_cld_shdw_mask

    cloudy = apply_cld_shdw_mask(add_fmask_cloud_mask(_fmask_image(ee_session, 1 << 1)))
    clear = apply_cld_shdw_mask(add_fmask_cloud_mask(_fmask_image(ee_session, 0)))
    assert first_value(cloudy, "NIR") is None
    assert first_value(clear, "NIR") == pytest.approx(0.4)


def test_hls_non_water_mask(ee_session, first_value):
    from eetools.sensors.hls.masking import build_hls_non_water_mask

    col = ee_session.ImageCollection(
        [
            ee_session.Image.cat(
                [
                    ee_session.Image.constant(0.5).rename("MNDWI"),
                    ee_session.Image.constant(0.1).rename("NDVI"),
                    ee_session.Image.constant(0.05).rename("NIR"),
                ]
            )
        ]
    )
    assert first_value(build_hls_non_water_mask(col), "non_water") == 0


def test_harmonize_hls_l30_bands(ee_session):
    from eetools.sensors.hls.preprocessing import harmonize_hls_l30_bands

    raw = (
        ee_session.Image.constant([1, 2, 3, 4, 5, 6])
        .rename(["B2", "B3", "B4", "B5", "B6", "B7"])
        .set("system:time_start", 1_600_000_000_000)
    )
    out = harmonize_hls_l30_bands(raw)
    assert out.bandNames().getInfo() == [
        "BLUE",
        "GREEN",
        "RED",
        "NIR",
        "SWIR1",
        "SWIR2",
    ]


def test_harmonize_hls_s30_bands(ee_session):
    from eetools.sensors.hls.preprocessing import harmonize_hls_s30_bands

    raw = (
        ee_session.Image.constant([1, 2, 3, 4, 5, 6])
        .rename(["B2", "B3", "B4", "B8A", "B11", "B12"])
        .set("system:time_start", 1_600_000_000_000)
    )
    out = harmonize_hls_s30_bands(raw)
    assert out.bandNames().getInfo() == [
        "BLUE",
        "GREEN",
        "RED",
        "NIR",
        "SWIR1",
        "SWIR2",
    ]


def test_process_hls_l30_image_adds_indices(ee_session):
    from eetools.sensors.hls.preprocessing import process_hls_l30_image

    raw = (
        ee_session.Image.constant([0.1, 0.2, 0.3, 0.5, 0.25, 0.15])
        .rename(["B2", "B3", "B4", "B5", "B6", "B7"])
        .set("system:time_start", 1_600_000_000_000)
    )
    out = ee_session.Image(process_hls_l30_image(raw))
    names = out.bandNames().getInfo()
    assert "NDVI" in names
    assert "RED" in names


def test_add_native_crs_sets_property(ee_session):
    from eetools.sensors.hls.preprocessing import add_native_crs

    img = ee_session.Image.constant(1).rename("RED")
    out = add_native_crs(img, reference_band="RED")
    assert out.get("native_crs").getInfo() is not None


@pytest.mark.slow
def test_get_hls_merged_collection_real(ee_session, small_aoi):
    from eetools.sensors.hls.preprocessing import get_hls_merged_collection

    col = get_hls_merged_collection(
        small_aoi,
        ee_session.Date("2021-06-01"),
        ee_session.Date("2021-07-01"),
        apply_water_masking=False,
    )
    assert isinstance(col, ee_session.ImageCollection)
    if col.size().getInfo() == 0:
        pytest.skip("no HLS scenes for the AOI/window")
    assert "NDVI" in col.first().bandNames().getInfo()
