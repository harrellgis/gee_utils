"""Tests for eetools.workflows.

Both functions are thin orchestration wrappers: all collaborators are mocked so
the tests are pure Python with no Earth Engine session required.  Validation errors
that originate inside the compositing layer (bad composite_stat / season_months)
propagate without being caught, so those paths are also verified here without mocks
on the compositing functions.
"""

from unittest.mock import MagicMock, patch

import pytest

import eetools.workflows as workflows

# --------------------------------------------------------------------------- #
# run_site_timeseries
# --------------------------------------------------------------------------- #


def test_run_site_timeseries_wires_collaborators_and_returns_stats():
    mock_sites = MagicMock(name="sites_fc")
    mock_builder = MagicMock(name="collection_builder")
    mock_aoi = "mock_aoi"
    mock_collection = "mock_collection"
    mock_composites = "mock_composites"
    mock_stats = "mock_stats"

    with (
        patch.object(
            workflows, "get_sites_geometry", return_value=mock_aoi
        ) as get_geom,
        patch.object(
            workflows, "build_period_composites", return_value=mock_composites
        ) as build,
        patch.object(
            workflows, "image_collection_to_region_stats_fc", return_value=mock_stats
        ) as reduce_fc,
        patch.object(workflows, "export_table_to_drive") as export_table,
    ):
        mock_builder.return_value = mock_collection

        result = workflows.run_site_timeseries(
            sites_fc=mock_sites,
            collection_builder=mock_builder,
            bands=["NDVI", "EVI"],
            start_date="2020-01-01",
            end_date="2023-01-01",
            export_folder="my_folder",
            file_prefix="s2_ndvi",
            temporal_scale="annual",
            composite_stat="median",
        )

    # geometry derived from sites
    get_geom.assert_called_once_with(mock_sites)
    # builder called with derived aoi and date range
    mock_builder.assert_called_once_with(mock_aoi, "2020-01-01", "2023-01-01")
    # compositing called with builder output and correct args
    build.assert_called_once_with(
        mock_collection, ["NDVI", "EVI"], "2020-01-01", "2023-01-01", "annual", "median"
    )
    # zonal reduction called with composites and original sites_fc
    reduce_fc.assert_called_once_with(
        mock_composites, mock_sites, ["NDVI", "EVI"], 5566, None, None, 4
    )
    # export started with correct folder and prefix
    export_table.assert_called_once_with(
        collection=mock_stats,
        description="s2_ndvi",
        folder="my_folder",
        fileNamePrefix="s2_ndvi",
    )
    # return value is the stats FeatureCollection
    assert result is mock_stats


def test_run_site_timeseries_passes_through_optional_kwargs():
    mock_sites = MagicMock(name="sites_fc")
    mock_builder = MagicMock(name="collection_builder", return_value="col")
    mock_reducer = MagicMock(name="reducer")

    with (
        patch.object(workflows, "get_sites_geometry", return_value="aoi"),
        patch.object(workflows, "build_period_composites", return_value="composites"),
        patch.object(
            workflows, "image_collection_to_region_stats_fc", return_value="stats"
        ) as reduce_fc,
        patch.object(workflows, "export_table_to_drive"),
    ):
        workflows.run_site_timeseries(
            sites_fc=mock_sites,
            collection_builder=mock_builder,
            bands=["NDVI"],
            start_date="2021-01-01",
            end_date="2022-01-01",
            export_folder="f",
            file_prefix="p",
            composite_stat="mean",
            reducers=mock_reducer,
            scale=10,
            image_properties=["year", "date"],
            tile_scale=2,
        )

    reduce_fc.assert_called_once_with(
        "composites", mock_sites, ["NDVI"], 10, mock_reducer, ["year", "date"], 2
    )


def test_run_site_timeseries_rejects_bad_composite_stat():
    """ValueError from _validate_composite_stat propagates without being caught."""
    mock_sites = MagicMock(name="sites_fc")
    mock_builder = MagicMock(name="collection_builder", return_value="col")

    with patch.object(workflows, "get_sites_geometry", return_value="aoi"):
        with pytest.raises(ValueError, match="composite_stat"):
            workflows.run_site_timeseries(
                sites_fc=mock_sites,
                collection_builder=mock_builder,
                bands=["NDVI"],
                start_date="2020-01-01",
                end_date="2023-01-01",
                export_folder="f",
                file_prefix="p",
                composite_stat="max",
            )


# --------------------------------------------------------------------------- #
# run_seasonal_site_timeseries
# --------------------------------------------------------------------------- #


def test_run_seasonal_site_timeseries_wires_collaborators_and_returns_stats():
    mock_sites = MagicMock(name="sites_fc")
    mock_builder = MagicMock(name="collection_builder")
    mock_aoi = "mock_aoi"
    mock_collection = "mock_collection"
    mock_composites = "mock_composites"
    mock_stats = "mock_stats"

    with (
        patch.object(
            workflows, "get_sites_geometry", return_value=mock_aoi
        ) as get_geom,
        patch.object(
            workflows, "build_seasonal_composites", return_value=mock_composites
        ) as build,
        patch.object(
            workflows, "image_collection_to_region_stats_fc", return_value=mock_stats
        ) as reduce_fc,
        patch.object(workflows, "export_table_to_drive") as export_table,
    ):
        mock_builder.return_value = mock_collection

        result = workflows.run_seasonal_site_timeseries(
            sites_fc=mock_sites,
            collection_builder=mock_builder,
            bands=["NDVI", "NBR"],
            start_date="2018-01-01",
            end_date="2023-01-01",
            start_year=2018,
            end_year=2022,
            season_months=(3, 5),
            season_name="wet",
            export_folder="out_folder",
            file_prefix="wet_season",
            composite_stat="median",
        )

    get_geom.assert_called_once_with(mock_sites)
    mock_builder.assert_called_once_with(mock_aoi, "2018-01-01", "2023-01-01")
    build.assert_called_once_with(
        mock_collection, ["NDVI", "NBR"], 2018, 2022, (3, 5), "wet", "median"
    )
    reduce_fc.assert_called_once_with(
        mock_composites, mock_sites, ["NDVI", "NBR"], 5566, None, None, 4
    )
    export_table.assert_called_once_with(
        collection=mock_stats,
        description="wet_season",
        folder="out_folder",
        fileNamePrefix="wet_season",
    )
    assert result is mock_stats


def test_run_seasonal_site_timeseries_rejects_bad_composite_stat():
    mock_sites = MagicMock(name="sites_fc")
    mock_builder = MagicMock(name="collection_builder", return_value="col")

    with patch.object(workflows, "get_sites_geometry", return_value="aoi"):
        with pytest.raises(ValueError, match="composite_stat"):
            workflows.run_seasonal_site_timeseries(
                sites_fc=mock_sites,
                collection_builder=mock_builder,
                bands=["NDVI"],
                start_date="2018-01-01",
                end_date="2023-01-01",
                start_year=2018,
                end_year=2022,
                season_months=(3, 5),
                season_name="wet",
                export_folder="f",
                file_prefix="p",
                composite_stat="max",
            )


def test_run_seasonal_site_timeseries_rejects_bad_season_months():
    mock_sites = MagicMock(name="sites_fc")
    mock_builder = MagicMock(name="collection_builder", return_value="col")

    with patch.object(workflows, "get_sites_geometry", return_value="aoi"):
        with pytest.raises(ValueError, match="season_months"):
            workflows.run_seasonal_site_timeseries(
                sites_fc=mock_sites,
                collection_builder=mock_builder,
                bands=["NDVI"],
                start_date="2018-01-01",
                end_date="2023-01-01",
                start_year=2018,
                end_year=2022,
                season_months=(8, 3),  # start > end
                season_name="bad",
                export_folder="f",
                file_prefix="p",
            )
