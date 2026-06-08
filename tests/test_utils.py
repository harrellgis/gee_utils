"""Tests for eetools.utils.

Pure helpers (``add_year_month``, geometry/feature builders, reducers) are
exercised on synthetic images so no real dataset is needed. The date-range
validator is checked for its error branches using a constant collection built
in-memory. ``gpkg_to_ee_geometry`` is tested against a temporary GeoPackage
written with geopandas (no Earth Engine required).
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

    img = ee_session.Image.constant(0.42).rename("b").clip(
        ee_session.Geometry.Rectangle([36.8, -3.4, 36.81, -3.39])
    )
    mn, mx = get_image_min_max(img, band_name="b", scale=100)
    assert mn == pytest.approx(0.42)
    assert mx == pytest.approx(0.42)


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


def test_gpkg_to_ee_geometry_roundtrip(ee_session, tmp_path):
    from eetools.utils import gpkg_to_ee_geometry

    poly = Polygon([(39.2, -4.3), (39.25, -4.3), (39.25, -4.25), (39.2, -4.25)])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326")
    path = tmp_path / "boundary.gpkg"
    gdf.to_file(path, driver="GPKG")

    geom = gpkg_to_ee_geometry(path)
    assert geom.type().getInfo() in {"Polygon", "MultiPolygon", "GeometryCollection"}


def test_gpkg_to_ee_geometry_requires_crs(tmp_path):
    # No Earth Engine needed: this branch raises before any ee call.
    from eetools.utils import gpkg_to_ee_geometry

    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs=None)
    path = tmp_path / "no_crs.gpkg"
    with warnings.catch_warnings():
        # Writing a deliberately CRS-less layer warns; that is the point here.
        warnings.simplefilter("ignore", UserWarning)
        gdf.to_file(path, driver="GPKG")

    with pytest.raises(ValueError, match="no CRS"):
        gpkg_to_ee_geometry(path)
