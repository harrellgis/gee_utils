"""Tests for the CHIRPS preprocessing module."""

from unittest.mock import patch

import pytest

from eetools import constants
from eetools.sensors.chirps import preprocessing


def test_export_total_rainfall_table_rejects_bad_temporal_scale():
    with pytest.raises(ValueError, match="temporal_scale must be one of"):
        preprocessing.export_total_rainfall_table(
            start_date="2020-01-01",
            end_date="2021-01-01",
            aoi="AOI",
            temporal_scale="daily",
            export_folder="folder",
        )


def test_export_total_rainfall_table_wires_workflow_and_exports():
    # Mock every collaborator so no real EE graph is built and no export task fires.
    with (
        patch.object(preprocessing, "get_chirps_collection") as get_col,
        patch.object(preprocessing, "build_period_composites") as build,
        patch.object(preprocessing, "collection_to_region_timeseries") as reduce_ts,
        patch.object(preprocessing, "export_table_to_drive") as export_table,
    ):
        get_col.return_value = "daily"
        build.return_value = "period_totals"
        reduce_ts.return_value = "timeseries"

        result = preprocessing.export_total_rainfall_table(
            start_date="2020-01-01",
            end_date="2026-01-01",
            aoi="AOI",
            temporal_scale="monthly",
            export_folder="CERK_Franklinia",
        )

    get_col.assert_called_once_with("AOI", "2020-01-01", "2026-01-01")

    # Daily precip is summed into per-period totals on the requested scale.
    _, build_kwargs = build.call_args
    assert build_kwargs["temporal_scale"] == "monthly"
    assert build_kwargs["composite_stat"] == "sum"
    assert build_kwargs["bands"] == [preprocessing.CHIRPS_PRECIP_BAND]

    # Each period is reduced over the AOI.
    _, reduce_kwargs = reduce_ts.call_args
    assert reduce_kwargs["region"] == "AOI"
    assert reduce_kwargs["scale"] == 5566
    # CHIRPS is already at native scale, so the small-region check is skipped.
    assert reduce_kwargs["min_pixels_per_region"] is None

    # The timeseries is exported as a CSV with the auto-derived prefix.
    _, export_kwargs = export_table.call_args
    assert export_kwargs["collection"] == "timeseries"
    assert export_kwargs["folder"] == "CERK_Franklinia"
    assert export_kwargs["fileNamePrefix"] == "chirps_monthly_total_rainfall"
    assert export_kwargs["fileFormat"] == "CSV"

    assert result == "timeseries"


@pytest.mark.ee
def test_process_chirps_image_selects_precip_and_keeps_props(ee_session, first_value):
    from eetools.constants import CHIRPS_PRECIP_BAND
    from eetools.sensors.chirps.preprocessing import process_chirps_image

    raw = (
        ee_session.Image.cat(
            [
                ee_session.Image.constant(12.5).rename(CHIRPS_PRECIP_BAND),
                ee_session.Image.constant(1).rename("extra"),
            ]
        )
        .set("system:time_start", ee_session.Date("2021-06-01").millis())
        .set("year", 2021)
        .set("month", 6)
        .set("day", 1)
    )
    out = ee_session.Image(process_chirps_image(raw))
    assert out.bandNames().getInfo() == [CHIRPS_PRECIP_BAND]
    assert first_value(out, CHIRPS_PRECIP_BAND) == pytest.approx(12.5)
    assert out.get("year").getInfo() == 2021
    assert out.get("month").getInfo() == 6


@pytest.mark.ee
@pytest.mark.slow
def test_get_chirps_collection_real(ee_session, small_aoi):
    from eetools.constants import CHIRPS_PRECIP_BAND
    from eetools.sensors.chirps.preprocessing import get_chirps_collection

    col = get_chirps_collection(small_aoi, "2020-01-01", "2020-01-15")
    assert isinstance(col, ee_session.ImageCollection)
    # CHIRPS is daily and global, so this window always has images.
    assert col.size().getInfo() > 0
    assert col.first().bandNames().getInfo() == [CHIRPS_PRECIP_BAND]


# --------------------------------------------------------------------------- #
# E7: daily CHIRPS versions, daily rainfall table, monthly climatology         #
# --------------------------------------------------------------------------- #


def test_chirps_daily_version_ids():
    assert constants.CHIRPS_DAILY_VERSIONS == {
        "v2": "UCSB-CHG/CHIRPS/DAILY",
        "v3_rnl": "UCSB-CHC/CHIRPS/V3/DAILY_RNL",
        "v3_sat": "UCSB-CHC/CHIRPS/V3/DAILY_SAT",
    }
    assert constants.CHIRPS_DAILY_TABLE_COLUMNS == (
        "date",
        "region_id",
        "ppt_mm",
        "chirps_version",
    )


def test_chirps_collection_id_versions_and_defaults():
    cid = preprocessing._chirps_collection_id
    assert cid("daily") == constants.CHIRPS_COLLECTION  # unchanged default
    assert cid("pentad") == constants.CHIRPS_PENTAD_COLLECTION
    for version, collection_id in constants.CHIRPS_DAILY_VERSIONS.items():
        assert cid("daily", version) == collection_id
    with pytest.raises(ValueError, match="daily CHIRPS only"):
        cid("pentad", "v2")
    with pytest.raises(ValueError, match="version must be one of"):
        cid("daily", "v4")


@pytest.mark.parametrize(
    "version, expected",
    [(None, "UCSB-CHC/CHIRPS/V3/DAILY_RNL"), ("v2", "UCSB-CHG/CHIRPS/DAILY")],
)
def test_get_chirps_collection_uses_version(version, expected):
    with (
        patch.object(preprocessing, "ee") as mock_ee,
        patch.object(preprocessing, "validate_chirps_date_range") as validate,
    ):
        kwargs = {} if version is None else {"version": version}
        preprocessing.get_chirps_collection("AOI", "2020-01-01", "2021-01-01", **kwargs)
    mock_ee.ImageCollection.assert_called_once_with(expected)
    assert validate.call_args.kwargs["collection_id"] == expected


def test_year_chunks_split_on_calendar_years():
    assert preprocessing._year_chunks("2019-07-01", "2021-03-01") == [
        (2019, "2019-07-01", "2020-01-01"),
        (2020, "2020-01-01", "2021-01-01"),
        (2021, "2021-01-01", "2021-03-01"),
    ]
    # An exclusive end on Jan 1 adds no empty chunk.
    assert preprocessing._year_chunks("2020-01-01", "2021-01-01") == [
        (2020, "2020-01-01", "2021-01-01")
    ]
    assert preprocessing._year_chunks("2021-01-01", "2021-01-01") == []


@pytest.fixture
def daily_export():
    with (
        patch.object(preprocessing, "ee") as mock_ee,
        patch.object(preprocessing, "validate_chirps_date_range") as validate,
        patch.object(preprocessing, "_daily_rows") as rows,
        patch.object(preprocessing, "export_table_to_gcs") as to_gcs,
    ):
        rows.side_effect = lambda col, regions, idp, version, scale: (
            f"rows_{version}",
            idp,
            scale,
        )
        yield mock_ee, validate, rows, to_gcs


def test_export_daily_rainfall_table_one_gcs_task_per_version_year(daily_export):
    mock_ee, validate, rows, to_gcs = daily_export
    tasks = preprocessing.export_daily_rainfall_table(
        "REGIONS",
        "2020-06-01",
        "2022-01-01",
        id_property="site_id",
        dest="gcs",
        bucket="bkt",
        prefix="proj/rain",
    )
    assert len(tasks) == 4  # (v2, v3_sat) x (2020, 2021)
    names = [c.kwargs["file_name_prefix"] for c in to_gcs.call_args_list]
    assert names == [
        "proj/rain/chirps_v2_daily_2020",
        "proj/rain/chirps_v2_daily_2021",
        "proj/rain/chirps_v3_sat_daily_2020",
        "proj/rain/chirps_v3_sat_daily_2021",
    ]
    first = to_gcs.call_args_list[0]
    assert first.args[0] == ("rows_v2", "site_id", 5566)
    assert first.kwargs["bucket"] == "bkt"
    assert first.kwargs["selectors"] == [
        "date",
        "region_id",
        "ppt_mm",
        "chirps_version",
    ]
    assert first.kwargs["skip_existing"] is True and first.kwargs["start"] is True
    # Versions never mixed: each table is built from one version's collection.
    assert [c.args[3] for c in rows.call_args_list] == ["v2", "v2", "v3_sat", "v3_sat"]
    # Date range validated once per version, against that version's collection.
    assert [c.kwargs["collection_id"] for c in validate.call_args_list] == [
        "UCSB-CHG/CHIRPS/DAILY",
        "UCSB-CHC/CHIRPS/V3/DAILY_SAT",
    ]
    source = mock_ee.ImageCollection.return_value.filterBounds.return_value
    source.select.return_value.filterDate.assert_any_call("2020-06-01", "2021-01-01")
    source.select.return_value.filterDate.assert_any_call("2021-01-01", "2022-01-01")


def test_export_daily_rainfall_table_drive(daily_export):
    mock_ee, _, _, to_gcs = daily_export
    tasks = preprocessing.export_daily_rainfall_table(
        "REGIONS",
        "2021-01-01",
        "2022-01-01",
        versions=("v3_rnl",),
        dest="drive",
        prefix="rain_folder",
        start=False,
    )
    to_gcs.assert_not_called()
    to_drive = mock_ee.batch.Export.table.toDrive
    kwargs = to_drive.call_args.kwargs
    assert kwargs["folder"] == "rain_folder"
    assert kwargs["fileNamePrefix"] == "chirps_v3_rnl_daily_2021"
    assert kwargs["fileFormat"] == "CSV"
    assert kwargs["selectors"] == list(constants.CHIRPS_DAILY_TABLE_COLUMNS)
    assert tasks == [to_drive.return_value]
    to_drive.return_value.start.assert_not_called()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"dest": "gcs", "prefix": "p"}, "bucket is required"),
        ({"dest": "s3", "prefix": "p"}, "dest must be"),
        ({"dest": "drive", "prefix": "p", "versions": ("v4",)}, "versions must be"),
        ({"dest": "drive", "prefix": "p", "versions": ()}, "versions must be"),
        ({"dest": "drive", "prefix": "p", "chunk": "month"}, "chunk must be"),
        ({"dest": "drive", "prefix": "p", "end_date": "2020-01-01"}, "must be after"),
    ],
)
def test_export_daily_rainfall_table_rejects_bad_args(daily_export, kwargs, match):
    params = {"start_date": "2020-01-01", "end_date": "2021-01-01", **kwargs}
    with pytest.raises(ValueError, match=match):
        preprocessing.export_daily_rainfall_table("REGIONS", **params)
    daily_export[1].assert_not_called()


def test_get_chirps_monthly_climatology_bands_and_errors():
    with pytest.raises(ValueError, match="end_year"):
        preprocessing.get_chirps_monthly_climatology("AOI", 2020, 1991)
    with (
        patch.object(preprocessing, "ee") as mock_ee,
        patch.object(preprocessing, "get_chirps_collection") as get_col,
    ):
        col = get_col.return_value.select.return_value
        months = []
        col.filter.return_value.sum.return_value.divide.return_value.rename.side_effect = (
            lambda name: months.append(name) or name
        )
        preprocessing.get_chirps_monthly_climatology("AOI", 1991, 2020)
    get_col.assert_called_once_with("AOI", "1991-01-01", "2021-01-01", cadence="pentad")
    assert months == [f"m{m:02d}" for m in range(1, 13)]
    col.filter.return_value.sum.return_value.divide.assert_called_with(30)
    mock_ee.Filter.calendarRange.assert_any_call(12, 12, "month")
    (bands,), _ = mock_ee.Image.cat.call_args
    assert bands == months


@pytest.mark.ee
def test_daily_rows_live_columns_and_means(ee_session):
    """Synthetic: two days x two regions -> four rows with the region means."""
    import ee

    def day(value, date):
        image = ee.Image.constant(value).rename("precipitation")
        return image.set("system:time_start", ee.Date(date).millis())

    col = ee.ImageCollection([day(3.0, "2021-01-01"), day(7.5, "2021-01-02")])
    regions = ee.FeatureCollection(
        [
            ee.Feature(ee.Geometry.Rectangle([36.0, -1.0, 36.1, -0.9]), {"rid": "a"}),
            ee.Feature(ee.Geometry.Rectangle([37.0, -1.0, 37.1, -0.9]), {"rid": "b"}),
        ]
    )
    rows = preprocessing._daily_rows(col, regions, "rid", "v2", 5566)
    props = [f["properties"] for f in rows.getInfo()["features"]]
    assert sorted(props, key=lambda p: (p["date"], p["region_id"])) == [
        {"date": "2021-01-01", "region_id": "a", "ppt_mm": 3.0, "chirps_version": "v2"},
        {"date": "2021-01-01", "region_id": "b", "ppt_mm": 3.0, "chirps_version": "v2"},
        {"date": "2021-01-02", "region_id": "a", "ppt_mm": 7.5, "chirps_version": "v2"},
        {"date": "2021-01-02", "region_id": "b", "ppt_mm": 7.5, "chirps_version": "v2"},
    ]


@pytest.mark.ee
@pytest.mark.slow
def test_monthly_climatology_sums_to_annual_climatology(ee_session):
    import ee

    point = ee.Geometry.Point(36.8, -1.3)
    monthly = preprocessing.get_chirps_monthly_climatology(point)
    annual = preprocessing.get_chirps_climatology(point)
    total = monthly.reduce(ee.Reducer.sum()).rename("total")
    values = (
        total.addBands(annual)
        .reduceRegion(ee.Reducer.first(), point, constants.CHIRPS_NATIVE_SCALE_M)
        .getInfo()
    )
    assert monthly.bandNames().getInfo() == [f"m{m:02d}" for m in range(1, 13)]
    assert abs(values["total"] - values["rain_mean_annual"]) < 0.01
