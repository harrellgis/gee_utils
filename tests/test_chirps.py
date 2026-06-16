"""Tests for the CHIRPS preprocessing module."""

from unittest.mock import patch

import pytest

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
