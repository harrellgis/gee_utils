"""Tests for eetools.utils.

Pure helpers (``add_year_month``, geometry/feature builders, reducers) are
exercised on synthetic images so no real dataset is needed. The date-range
validator is checked for its error branches using a constant collection built
in-memory. ``vector_file_to_ee_geometry`` is tested against temporary vector
files written with geopandas (no Earth Engine required for the guard branches).
"""

import warnings

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

pytestmark = pytest.mark.ee


def _make_timed_image(ee, value, millis):
    return ee.Image.constant(value).rename("b").set("system:time_start", millis)


def test_add_year_month(ee_session, first_value):
    from eetools.utils import add_year_month

    # 2023-06-15 in epoch millis (UTC).
    millis = ee_session.Date("2023-06-15").millis()
    img = ee_session.Image.constant(1).set("system:time_start", millis)
    out = add_year_month(img)
    assert out.get("year").getInfo() == 2023
    assert out.get("month").getInfo() == 6


def test_build_site_feature_properties(ee_session):
    from eetools.utils import build_site_feature

    geom = ee_session.Geometry.Point([36.8, -3.4])
    feat = build_site_feature(geom, site_id="s1", site_name="Site One")
    props = feat.toDictionary().getInfo()
    assert props["site_id"] == "s1"
    assert props["site_name"] == "Site One"
    assert "source_file" not in props


def test_build_site_feature_includes_source_file(ee_session):
    from eetools.utils import build_site_feature

    geom = ee_session.Geometry.Point([36.8, -3.4])
    feat = build_site_feature(
        geom, site_id="s1", site_name="Site One", source_file="boundary.gpkg"
    )
    assert feat.get("source_file").getInfo() == "boundary.gpkg"


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
    # Global extremes aggregate across both images.
    assert mn == pytest.approx(0.2)
    assert mx == pytest.approx(0.8)


def test_generate_diff_image_value(ee_session, first_value):
    from eetools.utils import generate_diff_image

    initial = ee_session.Image.constant(0.3).rename("NDVI")
    final = ee_session.Image.constant(0.5).rename("NDVI")
    diff = generate_diff_image(initial, final)
    # final - initial = 0.5 - 0.3
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
    # A window comfortably inside the CHIRPS archive must not raise.
    validate_collection_date_range(
        collection_id="UCSB-CHC/CHIRPS/V3/DAILY_RNL",
        aoi=aoi,
        start_date=ee_session.Date("2020-01-01"),
        end_date=ee_session.Date("2020-02-01"),
        sensor_label="CHIRPS imagery",
    )


def test_vector_file_to_ee_geometry_roundtrip(ee_session, tmp_path):
    from eetools.utils import vector_file_to_ee_geometry

    poly = Polygon([(39.2, -4.3), (39.25, -4.3), (39.25, -4.25), (39.2, -4.25)])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326")
    path = tmp_path / "boundary.gpkg"
    gdf.to_file(path, driver="GPKG")

    geom = vector_file_to_ee_geometry(path)
    assert geom.type().getInfo() in {"Polygon", "MultiPolygon", "GeometryCollection"}


def test_vector_files_to_feature_collection_one_feature_per_file(ee_session, tmp_path):
    from eetools.utils import vector_files_to_feature_collection

    poly_a = Polygon([(39.2, -4.3), (39.25, -4.3), (39.25, -4.25), (39.2, -4.25)])
    poly_b = Polygon([(36.8, -3.4), (36.9, -3.4), (36.9, -3.3), (36.8, -3.3)])
    path_a = tmp_path / "site_a.geojson"
    path_b = tmp_path / "site_b.geojson"
    gpd.GeoDataFrame({"id": [1]}, geometry=[poly_a], crs="EPSG:4326").to_file(
        path_a, driver="GeoJSON"
    )
    gpd.GeoDataFrame({"id": [1]}, geometry=[poly_b], crs="EPSG:4326").to_file(
        path_b, driver="GeoJSON"
    )

    fc = vector_files_to_feature_collection(
        [
            (path_a, "KLF", "Kilifi"),
            (path_b, "KWL", "Kwale"),
        ]
    )

    # One feature per input file, in input order, with caller-supplied identity.
    assert fc.size().getInfo() == 2
    info = fc.getInfo()
    ids = [f["properties"]["site_id"] for f in info["features"]]
    names = [f["properties"]["site_name"] for f in info["features"]]
    sources = [f["properties"]["source_file"] for f in info["features"]]
    assert ids == ["KLF", "KWL"]
    assert names == ["Kilifi", "Kwale"]
    # source_file is provenance, set automatically from the filename.
    assert sources == ["site_a.geojson", "site_b.geojson"]


def test_vector_files_to_feature_collection_dissolves_multifeature_file(
    ee_session, tmp_path
):
    from eetools.utils import vector_files_to_feature_collection

    # Two polygons in one file collapse to a single feature in the collection.
    poly_1 = Polygon([(39.2, -4.3), (39.25, -4.3), (39.25, -4.25), (39.2, -4.25)])
    poly_2 = Polygon([(39.3, -4.3), (39.35, -4.3), (39.35, -4.25), (39.3, -4.25)])
    path = tmp_path / "multi.geojson"
    gpd.GeoDataFrame(
        {"id": [1, 2]}, geometry=[poly_1, poly_2], crs="EPSG:4326"
    ).to_file(path, driver="GeoJSON")

    fc = vector_files_to_feature_collection([(path, "MLT", "Multi")])
    assert fc.size().getInfo() == 1


def test_vector_files_to_feature_collection_rejects_empty(tmp_path):
    # No Earth Engine needed: the guard raises before any ee call.
    from eetools.utils import vector_files_to_feature_collection

    with pytest.raises(ValueError, match="No files provided"):
        vector_files_to_feature_collection([])


def test_vector_file_to_ee_geometry_requires_crs(tmp_path):
    # No Earth Engine needed: this branch raises before any ee call.
    from eetools.utils import vector_file_to_ee_geometry

    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs=None)
    path = tmp_path / "no_crs.gpkg"
    with warnings.catch_warnings():
        # Writing a deliberately CRS-less layer warns; that is the point here.
        warnings.simplefilter("ignore", UserWarning)
        gdf.to_file(path, driver="GPKG")

    with pytest.raises(ValueError, match="no CRS"):
        vector_file_to_ee_geometry(path)


def test_vector_file_to_features_preserves_records_and_attributes(ee_session, tmp_path):
    from eetools.utils import vector_file_to_features

    # Two records with attribute columns: each must survive as its own feature.
    poly_1 = Polygon([(39.2, -4.3), (39.25, -4.3), (39.25, -4.25), (39.2, -4.25)])
    poly_2 = Polygon([(39.3, -4.3), (39.35, -4.3), (39.35, -4.25), (39.3, -4.25)])
    path = tmp_path / "plots.geojson"
    gpd.GeoDataFrame(
        {"plot_id": ["P1", "P2"], "area_ha": [1.5, 2.5]},
        geometry=[poly_1, poly_2],
        crs="EPSG:4326",
    ).to_file(path, driver="GeoJSON")

    fc = vector_file_to_features(path)

    assert fc.size().getInfo() == 2
    info = fc.getInfo()
    plot_ids = [f["properties"]["plot_id"] for f in info["features"]]
    areas = [f["properties"]["area_ha"] for f in info["features"]]
    assert plot_ids == ["P1", "P2"]
    assert areas == pytest.approx([1.5, 2.5])


def test_vector_file_to_features_can_drop_properties(ee_session, tmp_path):
    from eetools.utils import vector_file_to_features

    poly = Polygon([(39.2, -4.3), (39.25, -4.3), (39.25, -4.25), (39.2, -4.25)])
    path = tmp_path / "plots.geojson"
    gpd.GeoDataFrame(
        {"plot_id": ["P1"], "area_ha": [1.5]}, geometry=[poly], crs="EPSG:4326"
    ).to_file(path, driver="GeoJSON")

    fc = vector_file_to_features(path, keep_properties=False)
    props = fc.first().toDictionary().getInfo()
    assert props == {}


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
    # Per-image properties are carried through the clip.
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

    # Inside the AOI keeps its value; a point far outside is masked away.
    assert _sample(36.85, -3.35) == 5
    assert _sample(0.0, 0.0) is None


def test_clip_collection_to_geometry_clip_only(ee_session):
    from eetools.utils import clip_collection_to_geometry

    geom = ee_session.Geometry.Rectangle([36.8, -3.4, 36.9, -3.3])
    col = ee_session.ImageCollection([ee_session.Image.constant(5).rename("b")])
    out = clip_collection_to_geometry(col, geom, mask_outside=False)
    assert out.size().getInfo() == 1
