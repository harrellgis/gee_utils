"""Tests for eetools.vectors.

Vector-file readers and FeatureCollection helpers are exercised against
temporary files written with geopandas. Guard branches that raise before any
EE call are still marked ``ee`` because the module itself imports ee.
"""

import warnings

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

pytestmark = pytest.mark.ee


def test_build_site_feature_properties(ee_session):
    from eetools.vectors import build_site_feature

    geom = ee_session.Geometry.Point([36.8, -3.4])
    feat = build_site_feature(geom, site_id="s1", site_name="Site One")
    props = feat.toDictionary().getInfo()
    assert props["site_id"] == "s1"
    assert props["site_name"] == "Site One"
    assert "source_file" not in props


def test_build_site_feature_includes_source_file(ee_session):
    from eetools.vectors import build_site_feature

    geom = ee_session.Geometry.Point([36.8, -3.4])
    feat = build_site_feature(
        geom, site_id="s1", site_name="Site One", source_file="boundary.gpkg"
    )
    assert feat.get("source_file").getInfo() == "boundary.gpkg"


def test_vector_file_to_ee_geometry_roundtrip(ee_session, tmp_path):
    from eetools.vectors import vector_file_to_ee_geometry

    poly = Polygon([(39.2, -4.3), (39.25, -4.3), (39.25, -4.25), (39.2, -4.25)])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326")
    path = tmp_path / "boundary.gpkg"
    gdf.to_file(path, driver="GPKG")

    geom = vector_file_to_ee_geometry(path)
    assert geom.type().getInfo() in {"Polygon", "MultiPolygon", "GeometryCollection"}


def test_vector_files_to_feature_collection_one_feature_per_file(ee_session, tmp_path):
    from eetools.vectors import vector_files_to_feature_collection

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

    assert fc.size().getInfo() == 2
    info = fc.getInfo()
    ids = [f["properties"]["site_id"] for f in info["features"]]
    names = [f["properties"]["site_name"] for f in info["features"]]
    sources = [f["properties"]["source_file"] for f in info["features"]]
    assert ids == ["KLF", "KWL"]
    assert names == ["Kilifi", "Kwale"]
    assert sources == ["site_a.geojson", "site_b.geojson"]


def test_vector_files_to_feature_collection_dissolves_multifeature_file(
    ee_session, tmp_path
):
    from eetools.vectors import vector_files_to_feature_collection

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
    from eetools.vectors import vector_files_to_feature_collection

    with pytest.raises(ValueError, match="No files provided"):
        vector_files_to_feature_collection([])


def test_vector_file_to_ee_geometry_requires_crs(tmp_path):
    # No Earth Engine needed: this branch raises before any ee call.
    from eetools.vectors import vector_file_to_ee_geometry

    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs=None)
    path = tmp_path / "no_crs.gpkg"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gdf.to_file(path, driver="GPKG")

    with pytest.raises(ValueError, match="no CRS"):
        vector_file_to_ee_geometry(path)


def test_vector_file_to_features_preserves_records_and_attributes(ee_session, tmp_path):
    from eetools.vectors import vector_file_to_features

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
    from eetools.vectors import vector_file_to_features

    poly = Polygon([(39.2, -4.3), (39.25, -4.3), (39.25, -4.25), (39.2, -4.25)])
    path = tmp_path / "plots.geojson"
    gpd.GeoDataFrame(
        {"plot_id": ["P1"], "area_ha": [1.5]}, geometry=[poly], crs="EPSG:4326"
    ).to_file(path, driver="GeoJSON")

    fc = vector_file_to_features(path, keep_properties=False)
    props = fc.first().toDictionary().getInfo()
    assert props == {}


# --------------------------------------------------------------------------- #
# merge_vector_files
# --------------------------------------------------------------------------- #
def test_merge_vector_files_raises_on_empty_paths():
    # No Earth Engine needed: the guard raises before any ee call.
    from eetools.vectors import merge_vector_files

    with pytest.raises(ValueError, match="No files"):
        merge_vector_files([])


def test_merge_vector_files_preserves_individual_features(ee_session, tmp_path):
    from eetools.vectors import merge_vector_files

    poly_a1 = Polygon([(39.20, -4.30), (39.25, -4.30), (39.25, -4.25), (39.20, -4.25)])
    poly_a2 = Polygon([(39.30, -4.30), (39.35, -4.30), (39.35, -4.25), (39.30, -4.25)])
    poly_b1 = Polygon([(36.80, -3.40), (36.90, -3.40), (36.90, -3.30), (36.80, -3.30)])
    poly_b2 = Polygon([(36.90, -3.40), (37.00, -3.40), (37.00, -3.30), (36.90, -3.30)])

    path_a = tmp_path / "group_a.geojson"
    path_b = tmp_path / "group_b.geojson"
    gpd.GeoDataFrame(
        {"label": ["A1", "A2"]}, geometry=[poly_a1, poly_a2], crs="EPSG:4326"
    ).to_file(path_a, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"label": ["B1", "B2"]}, geometry=[poly_b1, poly_b2], crs="EPSG:4326"
    ).to_file(path_b, driver="GeoJSON")

    fc = merge_vector_files([path_a, path_b])
    assert fc.size().getInfo() == 4


def test_merge_vector_files_drops_properties_when_requested(ee_session, tmp_path):
    from eetools.vectors import merge_vector_files

    poly = Polygon([(39.20, -4.30), (39.25, -4.30), (39.25, -4.25), (39.20, -4.25)])
    path = tmp_path / "site.geojson"
    gpd.GeoDataFrame({"name": ["X"]}, geometry=[poly], crs="EPSG:4326").to_file(
        path, driver="GeoJSON"
    )

    fc = merge_vector_files([path], keep_properties=False)
    props = fc.first().toDictionary().getInfo()
    assert props == {}


# --------------------------------------------------------------------------- #
# fc_select_properties
# --------------------------------------------------------------------------- #
def test_fc_select_properties_raises_on_name_length_mismatch():
    # No Earth Engine needed: the guard raises before any ee call.
    from eetools.vectors import fc_select_properties

    with pytest.raises(ValueError, match="length"):
        fc_select_properties(None, ["a", "b"], new_names=["x"])  # type: ignore[arg-type]


def test_fc_select_properties_renames_column(ee_session):
    from eetools.vectors import fc_select_properties

    fc = ee_session.FeatureCollection(
        [
            ee_session.Feature(
                ee_session.Geometry.Point([36.8, -3.4]),
                {"region_label": "Zone A", "extra": 99},
            )
        ]
    )
    renamed = fc_select_properties(fc, ["region_label"], new_names=["site_name"])
    props = renamed.first().toDictionary().getInfo()
    assert props.get("site_name") == "Zone A"
    assert "region_label" not in props
    assert "extra" not in props


def test_fc_select_properties_preserves_names_when_no_new_names(ee_session):
    from eetools.vectors import fc_select_properties

    fc = ee_session.FeatureCollection(
        [
            ee_session.Feature(
                ee_session.Geometry.Point([36.8, -3.4]),
                {"site_name": "Zone A", "extra": 99},
            )
        ]
    )
    selected = fc_select_properties(fc, ["site_name"])
    props = selected.first().toDictionary().getInfo()
    assert props.get("site_name") == "Zone A"
    assert "extra" not in props


# --------------------------------------------------------------------------- #
# buffer_feature_collection
# --------------------------------------------------------------------------- #
def test_buffer_feature_collection_expands_point_to_polygon(ee_session):
    from eetools.vectors import buffer_feature_collection

    fc = ee_session.FeatureCollection(
        [ee_session.Feature(ee_session.Geometry.Point([36.85, -3.35]), {"name": "P"})]
    )
    buffered = buffer_feature_collection(fc, distance_m=1000.0)
    area = buffered.first().geometry().area(maxError=10).getInfo()
    assert area > 0


def test_buffer_feature_collection_preserves_properties(ee_session):
    from eetools.vectors import buffer_feature_collection

    fc = ee_session.FeatureCollection(
        [
            ee_session.Feature(
                ee_session.Geometry.Point([36.85, -3.35]), {"site_name": "P1"}
            )
        ]
    )
    buffered = buffer_feature_collection(fc, distance_m=500.0)
    assert buffered.first().get("site_name").getInfo() == "P1"
