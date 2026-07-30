"""Tests for eetools.utils.

Pure helpers (``add_year_month``, reducers) are exercised on synthetic images
so no real dataset is needed. The date-range validator is checked for its error
branches using a constant collection built in-memory. Vector utilities have
moved to ``tests/test_vectors.py``.
"""

import pytest

pytestmark = pytest.mark.ee


def test_add_year_month(ee_session, first_value):
    from eetools.utils import add_year_month

    # 2023-06-15 in epoch millis (UTC).
    millis = ee_session.Date("2023-06-15").millis()
    img = ee_session.Image.constant(1).set("system:time_start", millis)
    out = add_year_month(img)
    assert out.get("year").getInfo() == 2023
    assert out.get("month").getInfo() == 6


def test_get_image_min_max_constant(ee_session):
    from eetools.utils import get_image_min_max

    img = (
        ee_session.Image.constant(0.42)
        .rename("b")
        .clip(ee_session.Geometry.Rectangle([36.8, -3.4, 36.81, -3.39]))
    )
    mn, mx = get_image_min_max(img, band_name="b", scale=100)
    assert mn == pytest.approx(0.42)
    assert mx == pytest.approx(0.42)


def test_get_collection_min_max_across_images(ee_session, small_aoi):
    from eetools.utils import get_collection_min_max

    col = ee_session.ImageCollection(
        [
            ee_session.Image.constant(0.2).rename("b").clip(small_aoi),
            ee_session.Image.constant(0.8).rename("b").clip(small_aoi),
        ]
    )
    mn, mx = get_collection_min_max(col, band_name="b", scale=500)
    assert mn == pytest.approx(0.2)
    assert mx == pytest.approx(0.8)


def test_generate_diff_image_value(ee_session, first_value):
    from eetools.utils import generate_diff_image

    initial = ee_session.Image.constant(0.3).rename("NDVI")
    final = ee_session.Image.constant(0.5).rename("NDVI")
    diff = generate_diff_image(initial, final)
    assert first_value(diff, "NDVI") == pytest.approx(0.2)


def test_generate_diff_image_negative_change(ee_session, first_value):
    from eetools.utils import generate_diff_image

    initial = ee_session.Image.constant(0.6).rename("NDVI")
    final = ee_session.Image.constant(0.4).rename("NDVI")
    diff = generate_diff_image(initial, final)
    assert first_value(diff, "NDVI") == pytest.approx(-0.2)


def test_generate_diff_image_selects_bands(ee_session):
    from eetools.utils import generate_diff_image

    initial = ee_session.Image.constant([0.3, 0.1]).rename(["NDVI", "EVI"])
    final = ee_session.Image.constant([0.5, 0.2]).rename(["NDVI", "EVI"])
    diff = generate_diff_image(initial, final, bands=["NDVI"])
    assert diff.bandNames().getInfo() == ["NDVI"]


def test_generate_diff_image_output_suffix(ee_session, first_value):
    from eetools.utils import generate_diff_image

    initial = ee_session.Image.constant(0.3).rename("NDVI")
    final = ee_session.Image.constant(0.5).rename("NDVI")
    diff = generate_diff_image(initial, final, output_suffix="_diff")
    assert diff.bandNames().getInfo() == ["NDVI_diff"]
    assert first_value(diff, "NDVI_diff") == pytest.approx(0.2)


def test_temporal_reducer_band_names(ee_session):
    from eetools.utils import temporal_reducer

    col = ee_session.ImageCollection(
        [
            ee_session.Image.constant(1).rename("b"),
            ee_session.Image.constant(3).rename("b"),
        ]
    )
    out = temporal_reducer(col, percentiles=[10, 90])
    names = out.bandNames().getInfo()
    assert "b_mean" in names
    assert "b_p10" in names
    assert "b_p90" in names
    assert "b_stdDev" in names


def test_clip_image_to_fc_sets_site_name(ee_session):
    from eetools.utils import clip_image_to_fc

    fc = ee_session.FeatureCollection(
        [
            ee_session.Feature(
                ee_session.Geometry.Rectangle([36.8, -3.4, 36.9, -3.3]),
                {"site_name": "A"},
            )
        ]
    )
    img = ee_session.Image.constant(1).rename("b")
    out = clip_image_to_fc(fc, img)
    assert out.size().getInfo() == 1
    assert out.first().get("site_name").getInfo() == "A"


def test_get_date_window_annual(ee_session):
    from eetools.utils import get_date_window

    start, end = get_date_window(year_start=2015, year_end=2018, season="annual")
    assert start.format("YYYY-MM-dd").getInfo() == "2015-01-01"
    assert end.format("YYYY-MM-dd").getInfo() == "2019-01-01"


def test_get_date_window_season_from_mapping(ee_session):
    from eetools.utils import get_date_window

    season_months = {"wet": (3, 6), "dry": (7, 11)}
    start, end = get_date_window(
        year_start=2020, year_end=2022, season="wet", season_months=season_months
    )
    assert start.format("YYYY-MM-dd").getInfo() == "2020-03-01"
    assert end.format("YYYY-MM-dd").getInfo() == "2022-06-01"


def test_get_date_window_unknown_season_raises(ee_session):
    from eetools.utils import get_date_window

    with pytest.raises(ValueError, match="Unknown season"):
        get_date_window(year_start=2020, year_end=2020, season="spring")


def test_get_available_window_returns_extent(ee_session):
    from eetools.utils import get_available_window

    aoi = ee_session.Geometry.Rectangle([39.2, -4.3, 39.25, -4.25])
    start, end = get_available_window("UCSB-CHC/CHIRPS/V3/DAILY_RNL", aoi)
    assert start.millis().getInfo() < end.millis().getInfo()


def test_get_available_window_no_images_raises(ee_session):
    from eetools.utils import get_available_window

    # NAIP only covers the continental US, so an East Africa AOI has no footprint.
    non_us_aoi = ee_session.Geometry.Rectangle([39.2, -4.3, 39.25, -4.25])
    with pytest.raises(ValueError, match="No imagery found"):
        get_available_window("USDA/NAIP/DOQQ", non_us_aoi)


def test_validate_collection_date_range_rejects_inverted_range(ee_session):
    from eetools.utils import validate_collection_date_range

    aoi = ee_session.Geometry.Rectangle([39.2, -4.3, 39.25, -4.25])
    with pytest.raises(ValueError, match="must be earlier"):
        validate_collection_date_range(
            collection_id="UCSB-CHC/CHIRPS/V3/DAILY_RNL",
            aoi=aoi,
            start_date=ee_session.Date("2021-02-01"),
            end_date=ee_session.Date("2021-01-01"),
            sensor_label="CHIRPS imagery",
        )


@pytest.mark.slow
def test_validate_collection_date_range_accepts_valid_window(ee_session):
    from eetools.utils import validate_collection_date_range

    aoi = ee_session.Geometry.Rectangle([39.2, -4.3, 39.25, -4.25])
    validate_collection_date_range(
        collection_id="UCSB-CHC/CHIRPS/V3/DAILY_RNL",
        aoi=aoi,
        start_date=ee_session.Date("2020-01-01"),
        end_date=ee_session.Date("2020-02-01"),
        sensor_label="CHIRPS imagery",
    )


def test_clip_collection_to_geometry_preserves_size_and_time(ee_session):
    from eetools.utils import clip_collection_to_geometry

    geom = ee_session.Geometry.Rectangle([36.8, -3.4, 36.9, -3.3])
    col = ee_session.ImageCollection(
        [
            ee_session.Image.constant(1).rename("b").set("system:time_start", 1000),
            ee_session.Image.constant(2).rename("b").set("system:time_start", 2000),
        ]
    )
    out = clip_collection_to_geometry(col, geom)
    assert out.size().getInfo() == 2
    assert out.first().get("system:time_start").getInfo() == 1000


def test_clip_collection_to_geometry_masks_outside(ee_session):
    from eetools.utils import clip_collection_to_geometry

    geom = ee_session.Geometry.Rectangle([36.8, -3.4, 36.9, -3.3])
    col = ee_session.ImageCollection([ee_session.Image.constant(5).rename("b")])
    clipped = ee_session.Image(clip_collection_to_geometry(col, geom).first())

    def _sample(lon, lat):
        return (
            clipped.reduceRegion(
                reducer=ee_session.Reducer.first(),
                geometry=ee_session.Geometry.Point([lon, lat]),
                scale=1000,
            )
            .get("b")
            .getInfo()
        )

    assert _sample(36.85, -3.35) == 5
    assert _sample(0.0, 0.0) is None


def test_clip_collection_to_geometry_clip_only(ee_session):
    from eetools.utils import clip_collection_to_geometry

    geom = ee_session.Geometry.Rectangle([36.8, -3.4, 36.9, -3.3])
    col = ee_session.ImageCollection([ee_session.Image.constant(5).rename("b")])
    out = clip_collection_to_geometry(col, geom, mask_outside=False)
    assert out.size().getInfo() == 1
