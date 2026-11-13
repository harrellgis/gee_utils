"""Tests for eetools.zonal.

All functions build server-side EE graphs and are exercised on small synthetic
constant images/collections under the ``ee`` marker.
"""

from unittest.mock import patch

import pytest

from eetools import zonal


@pytest.mark.ee
def test_reduce_image_over_region_combines_props_and_stats(ee_session, small_aoi):
    from eetools.zonal import reduce_image_over_region

    img = (
        ee_session.Image.constant(0.5)
        .rename("b")
        .clip(small_aoi)
        .set("year", 2021)
        .set("date", "2021-06-01")
        .set("temporal_scale", "annual")
    )
    feat = reduce_image_over_region(img, region=small_aoi, bands=["b"], scale=100)
    props = feat.toDictionary().getInfo()
    assert props["b"] == pytest.approx(0.5)
    assert props["year"] == 2021
    assert props["temporal_scale"] == "annual"


@pytest.mark.ee
def test_collection_to_region_timeseries_one_feature_per_image(
    ee_session, timed_collection, small_aoi
):
    from eetools.zonal import collection_to_region_timeseries

    annotated = timed_collection.map(
        lambda img: ee_session.Image(img).set("year", 2020)
    )
    fc = collection_to_region_timeseries(
        annotated, region=small_aoi, bands=["b"], scale=100
    )
    assert isinstance(fc, ee_session.FeatureCollection)
    assert fc.size().getInfo() == 2


@pytest.mark.ee
def test_image_collection_to_region_stats_fc(ee_session, small_aoi):
    from eetools.zonal import image_collection_to_region_stats_fc

    image = (
        ee_session.Image.constant(0.5)
        .rename("b")
        .set("year", 2021)
        .set("product", "test")
        .set("param_set", "default")
    )
    collection = ee_session.ImageCollection([image])
    regions = ee_session.FeatureCollection(
        [ee_session.Feature(small_aoi, {"site_name": "A"})]
    )
    fc = image_collection_to_region_stats_fc(
        collection, regions_fc=regions, bands=["b"], scale=100
    )
    assert isinstance(fc, ee_session.FeatureCollection)
    assert fc.size().getInfo() == 1
    props = fc.first().toDictionary().getInfo()
    assert props["site_name"] == "A"
    assert props["year"] == 2021


@pytest.mark.ee
def test_summarize_collection_histograms(ee_session, small_aoi):
    from eetools.zonal import summarize_collection_histograms

    image = (
        ee_session.Image.constant(0.5).rename("b").clip(small_aoi).set("site_name", "A")
    )
    collection = ee_session.ImageCollection([image])
    result = summarize_collection_histograms(
        collection,
        band_name="b",
        min_value=0.0,
        max_value=1.0,
        steps=4,
        scale=100,
    )
    assert isinstance(result, list)
    assert result[0]["site_name"] == "A"
    assert result[0]["histogram"] is not None


@pytest.mark.ee
def test_summarize_collection_histograms_custom_name_field(ee_session, small_aoi):
    from eetools.zonal import summarize_collection_histograms

    image = (
        ee_session.Image.constant(0.5)
        .rename("b")
        .clip(small_aoi)
        .set("region_id", "R1")
    )
    collection = ee_session.ImageCollection([image])
    result = summarize_collection_histograms(
        collection,
        band_name="b",
        min_value=0.0,
        max_value=1.0,
        steps=4,
        scale=100,
        name_field="region_id",
    )
    assert result[0]["region_id"] == "R1"
    assert "site_name" not in result[0]


# --------------------------------------------------------------------------- #
# image_collection_to_region_stats_fc default image_properties
# --------------------------------------------------------------------------- #
@pytest.mark.ee
def test_region_stats_fc_copies_composite_props_by_default(ee_session, small_aoi):
    from eetools.compositing import build_period_composites
    from eetools.zonal import image_collection_to_region_stats_fc

    col = ee_session.ImageCollection(
        [
            ee_session.Image.constant(0.5)
            .toFloat()
            .rename("b")
            .set("system:time_start", ee_session.Date("2020-06-01").millis()),
            ee_session.Image.constant(0.7)
            .toFloat()
            .rename("b")
            .set("system:time_start", ee_session.Date("2021-06-01").millis()),
        ]
    )
    composites = build_period_composites(
        col, bands=["b"], start_date="2020-01-01", end_date="2022-01-01"
    )
    regions = ee_session.FeatureCollection(
        [ee_session.Feature(small_aoi, {"site_name": "A"})]
    )
    fc = image_collection_to_region_stats_fc(
        composites, regions_fc=regions, bands=["b"], scale=100
    )
    props = fc.first().toDictionary().getInfo()
    assert props.get("year") is not None
    assert props.get("composite_stat") is not None
    assert "product" not in props
    assert "param_set" not in props


# --------------------------------------------------------------------------- #
# reduce_image_over_region — generalised property copy
# --------------------------------------------------------------------------- #
@pytest.mark.ee
def test_reduce_image_over_region_copies_seasonal_props(ee_session, small_aoi):
    from eetools.zonal import reduce_image_over_region

    img = (
        ee_session.Image.constant(0.5)
        .rename("b")
        .clip(small_aoi)
        .set("year", 2021)
        .set("season", "wet")
        .set("season_months", "3-5")
        .set("composite_stat", "median")
    )
    feat = reduce_image_over_region(img, region=small_aoi, bands=["b"], scale=100)
    props = feat.toDictionary().getInfo()
    assert props["season"] == "wet"
    assert props["season_months"] == "3-5"
    assert props["year"] == 2021
    assert props["b"] == pytest.approx(0.5)


@pytest.mark.ee
def test_reduce_image_over_region_no_spurious_none_props(ee_session, small_aoi):
    from eetools.zonal import reduce_image_over_region

    # An image with only 'year' set should not produce None-valued keys for
    # the old hardcoded list (date, month, day, temporal_scale, etc.).
    img = ee_session.Image.constant(0.3).rename("b").clip(small_aoi).set("year", 2022)
    feat = reduce_image_over_region(img, region=small_aoi, bands=["b"], scale=100)
    props = feat.toDictionary().getInfo()
    assert set(props.keys()) == {"year", "b"}
    assert props["b"] == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# summarize_class_areas
# --------------------------------------------------------------------------- #
@pytest.mark.ee
def test_summarize_class_areas_sums_only_matching_class(ee_session, small_aoi):
    from eetools.zonal import summarize_class_areas

    classified = (
        ee_session.Image.constant(10).toInt().rename("land_cover").clip(small_aoi)
    )
    regions = ee_session.FeatureCollection(
        [ee_session.Feature(small_aoi, {"site_name": "A"})]
    )
    class_map = {"tree_cover": 10, "shrubland": 20}

    fc = summarize_class_areas(regions, classified, class_map, scale=1000)
    props = fc.first().toDictionary().getInfo()

    assert props["site_name"] == "A"
    assert props["tree_cover_area_m2"] > 0
    assert props["shrubland_area_m2"] == 0


@pytest.mark.ee
def test_summarize_class_areas_chains_onto_prior_reduceregions(ee_session, small_aoi):
    from eetools.zonal import summarize_class_areas

    classified = (
        ee_session.Image.constant(20).toInt().rename("land_cover").clip(small_aoi)
    )
    regions = ee_session.FeatureCollection(
        [ee_session.Feature(small_aoi, {"existing_stat": 1.5})]
    )
    class_map = {"tree_cover": 10, "shrubland": 20}

    fc = summarize_class_areas(regions, classified, class_map, scale=1000)
    props = fc.first().toDictionary().getInfo()

    # Prior properties on the input collection survive the chained reduceRegions.
    assert props["existing_stat"] == pytest.approx(1.5)
    assert props["shrubland_area_m2"] > 0


# --------------------------------------------------------------------------- #
# scale is required + small-region warning                                     #
# --------------------------------------------------------------------------- #


def test_reduce_image_over_region_requires_scale_keyword():
    from eetools.zonal import reduce_image_over_region

    with pytest.raises(TypeError, match="scale"):
        reduce_image_over_region(None, None, ["b"])


def test_collection_to_region_timeseries_requires_scale_keyword():
    from eetools.zonal import collection_to_region_timeseries

    with pytest.raises(TypeError, match="scale"):
        collection_to_region_timeseries(None, None, ["b"])


def test_small_region_message_flags_site_inside_one_chirps_pixel():
    from eetools.zonal import small_region_warning_message

    # 5.8 ha (TGBS_Ecosia-sized) and 97.6 ha sites at the old 5566 m default.
    msg = small_region_warning_message(
        [58_000.0, 976_000.0], ["ecosia", "itf"], scale=5566
    )
    assert msg is not None
    assert "2 of 2 region(s)" in msg
    assert "'ecosia'" in msg  # names the smallest region
    assert "5.80 ha" in msg
    # sqrt(58000 / 4) = 120.4 -> suggest <= 120 m
    assert "scale <= 120 m" in msg


def test_small_region_message_none_when_regions_cover_enough_pixels():
    from eetools.zonal import small_region_warning_message

    # 5.8 ha at 30 m is ~64 pixels.
    assert small_region_warning_message([58_000.0], ["ecosia"], scale=30) is None
    assert small_region_warning_message([], [], scale=30) is None


def test_small_region_message_threshold_is_configurable():
    from eetools.zonal import small_region_warning_message

    # 10 000 m^2 at 30 m = ~11 pixels: fine at 4, flagged at 20.
    assert small_region_warning_message([10_000.0], ["a"], 30, 4) is None
    assert small_region_warning_message([10_000.0], ["a"], 30, 20) is not None


@pytest.mark.parametrize(("scale", "min_pixels"), [(0, 4), (-30, 4), (30, 0), (30, -1)])
def test_small_region_message_rejects_non_positive(scale, min_pixels):
    from eetools.zonal import small_region_warning_message

    with pytest.raises(ValueError, match="must be positive"):
        small_region_warning_message([1.0], ["a"], scale, min_pixels)


@pytest.mark.ee
def test_check_region_pixel_coverage_warns_for_feature_collection(
    ee_session, small_aoi
):
    from eetools.zonal import check_region_pixel_coverage

    # ~0.01 deg square (~1.2 km^2) inside small_aoi: ~0.04 pixels at 5566 m.
    tiny = ee_session.Geometry.Rectangle([39.20, -4.30, 39.21, -4.29])
    regions = ee_session.FeatureCollection(
        [
            ee_session.Feature(small_aoi, {"site_name": "big"}),
            ee_session.Feature(tiny, {"site_name": "tiny"}),
        ]
    )
    with pytest.warns(UserWarning, match="'tiny'"):
        check_region_pixel_coverage(regions, scale=5566)

    assert check_region_pixel_coverage(regions, scale=30) is None


@pytest.mark.ee
def test_collection_to_region_timeseries_warns_on_coarse_scale(
    ee_session, timed_collection, small_aoi
):
    from eetools.zonal import collection_to_region_timeseries

    # small_aoi is ~5.5 x 5.5 km: about one pixel at 5566 m.
    with pytest.warns(UserWarning, match="fewer than 4 pixels"):
        collection_to_region_timeseries(
            timed_collection, region=small_aoi, bands=["b"], scale=5566
        )


# --------------------------------------------------------------------------- #
# E8: per-scene site table                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"bands": [], "scale": 10}, "at least one band"),
        ({"bands": ["nirv"], "scale": 0}, "scale must be positive"),
    ],
)
def test_scene_site_rows_rejects_bad_args(kwargs, match):
    with pytest.raises(ValueError, match=match):
        zonal.scene_site_rows("COL", "REGIONS", **kwargs)


def test_export_scene_site_table_columns_and_dispatch():
    with (
        patch.object(zonal, "scene_site_rows") as rows,
        patch.object(zonal, "export_csv_table") as export,
    ):
        task = zonal.export_scene_site_table(
            "COL",
            "REGIONS",
            bands=["nirv", "ndvi"],
            scale=10,
            dest="gcs",
            bucket="bkt",
            prefix="proj/scenes",
            crs="EPSG:32737",
            start=False,
        )
    rows.assert_called_once_with(
        "COL",
        "REGIONS",
        bands=["nirv", "ndvi"],
        scale=10,
        id_property="site_id",
        crs="EPSG:32737",
    )
    kwargs = export.call_args.kwargs
    assert export.call_args.args == (rows.return_value,)
    assert kwargs["selectors"] == [
        "date",
        "scene_id",
        "region_id",
        "nirv",
        "ndvi",
        "clear_frac",
        "footprint_frac",
    ]
    assert kwargs["name"] == "scene_site_table"
    assert (kwargs["dest"], kwargs["bucket"], kwargs["prefix"]) == (
        "gcs",
        "bkt",
        "proj/scenes",
    )
    assert kwargs["start"] is False and kwargs["skip_existing"] is True
    assert task is export.return_value


def test_export_scene_site_table_validates_dest_before_building():
    with patch.object(zonal, "scene_site_rows") as rows:
        with pytest.raises(ValueError, match="bucket is required"):
            zonal.export_scene_site_table(
                "COL", "REGIONS", bands=["nirv"], scale=10, dest="gcs", prefix="p"
            )
    rows.assert_not_called()


@pytest.mark.ee
def test_scene_site_rows_live_fractions(ee_session):
    """Synthetic scenes: partial footprint, half-cloudy, missed and fully masked."""
    import ee

    def rect(x0, x1):
        return ee.Geometry.Rectangle([x0, -1.0, x1, -0.9])

    regions = ee.FeatureCollection(
        [
            ee.Feature(rect(36.0, 36.1), {"site_id": "a"}),
            ee.Feature(rect(36.2, 36.3), {"site_id": "b"}),
            ee.Feature(rect(37.0, 37.1), {"site_id": "c"}),  # outside both scenes
        ]
    )
    lon = ee.Image.pixelLonLat().select("longitude")

    def scene(clear_from_lon, footprint, date):
        nirv = ee.Image.constant(0.3).rename("nirv").updateMask(lon.gte(clear_from_lon))
        ndvi = ee.Image.constant(0.6).rename("ndvi")
        image = nirv.addBands(ndvi).clip(footprint)
        return image.set("system:time_start", ee.Date(date).millis())

    collection = ee.ImageCollection(
        [
            # Covers a fully (west half cloudy) and the west half of b (clear).
            scene(
                36.05, ee.Geometry.Rectangle([35.9, -1.1, 36.25, -0.8]), "2021-01-01"
            ),
            # Covers a and b fully; a is all cloud.
            scene(36.15, ee.Geometry.Rectangle([35.9, -1.1, 36.4, -0.8]), "2021-01-06"),
        ]
    )
    rows = zonal.scene_site_rows(collection, regions, bands=["nirv", "ndvi"], scale=500)
    got = {
        (p["date"], p["region_id"]): p
        for p in (f["properties"] for f in rows.getInfo()["features"])
    }
    assert set(got) == {
        ("2021-01-01", "a"),
        ("2021-01-01", "b"),
        ("2021-01-06", "a"),
        ("2021-01-06", "b"),
    }
    tol = 0.03
    a1, b1, a2, b2 = (got[k] for k in sorted(got))
    assert a1["scene_id"] == "0" and a2["scene_id"] == "1"
    assert a1["footprint_frac"] == pytest.approx(1.0, abs=tol)
    assert a1["clear_frac"] == pytest.approx(0.5, abs=tol)
    assert a1["nirv"] == pytest.approx(0.3) and a1["ndvi"] == pytest.approx(0.6)
    assert b1["footprint_frac"] == pytest.approx(0.5, abs=tol)
    assert b1["clear_frac"] == pytest.approx(1.0, abs=tol)
    assert a2["clear_frac"] == pytest.approx(0.0, abs=1e-9)
    assert "nirv" not in a2 or a2["nirv"] is None  # no clear pixel -> empty mean
    assert a2["ndvi"] == pytest.approx(0.6)
    assert b2["footprint_frac"] == pytest.approx(1.0, abs=tol)
    assert b2["clear_frac"] == pytest.approx(1.0, abs=tol)
