"""Tests for eetools.zonal.

All functions build server-side EE graphs and are exercised on small synthetic
constant images/collections under the ``ee`` marker.
"""

import pytest


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
