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


# --------------------------------------------------------------------------- #
# build_baseline_layers
# --------------------------------------------------------------------------- #


def test_build_baseline_layers_wires_all_sensors():
    mock_aoi = "mock_aoi"
    mock_terrain = MagicMock(name="terrain")
    mock_terrain.select.side_effect = lambda band: f"terrain_{band}"

    with (
        patch.object(
            workflows, "get_terrain", return_value=mock_terrain
        ) as get_terrain,
        patch.object(
            workflows, "get_canopy_height", return_value="canopy"
        ) as get_canopy,
        patch.object(workflows, "get_land_cover", return_value="land_cover") as get_lc,
        patch.object(workflows, "get_soil_carbon", return_value="soil") as get_soil,
        patch.object(workflows, "get_bii", return_value="bii") as get_bii,
        patch.object(
            workflows, "get_forest_2000", return_value="forest_2000"
        ) as get_f2000,
        patch.object(
            workflows, "get_forest_loss_image", return_value="forest_loss"
        ) as get_floss,
    ):
        layers = workflows.build_baseline_layers(mock_aoi)

    get_terrain.assert_called_once_with(aoi=mock_aoi)
    get_canopy.assert_called_once_with(aoi=mock_aoi)
    get_lc.assert_called_once_with(aoi=mock_aoi)
    get_soil.assert_called_once_with(aoi=mock_aoi)
    get_bii.assert_called_once_with(mock_aoi, "BII All", "1km")
    get_f2000.assert_called_once_with(mock_aoi)
    get_floss.assert_called_once_with(mock_aoi)

    assert layers == {
        "dem": "terrain_elevation",
        "slope": "terrain_slope",
        "hillshade": "terrain_hillshade",
        "canopy_height": "canopy",
        "land_cover": "land_cover",
        "soil_carbon": "soil",
        "bii_all": "bii",
        "forest_2000": "forest_2000",
        "forest_loss": "forest_loss",
    }


# --------------------------------------------------------------------------- #
# summarize_baseline_layers (real EE graph-building — synthetic images)
# --------------------------------------------------------------------------- #


@pytest.mark.ee
def test_summarize_baseline_layers_chains_continuous_and_landcover(
    ee_session, small_aoi
):
    sites_fc = ee_session.FeatureCollection(
        [ee_session.Feature(small_aoi, {"site_name": "A"})]
    )
    layers = {
        name: ee_session.Image.constant(value).toFloat().rename(band).clip(small_aoi)
        for name, band, value in [
            ("dem", "elevation", 100),
            ("slope", "slope", 5),
            ("hillshade", "hillshade", 200),
            ("canopy_height", "canopy_height", 10),
            ("soil_carbon", "soil_carbon", 30),
            ("bii_all", "BII All", 0.7),
        ]
    }
    layers["land_cover"] = (
        ee_session.Image.constant(10).toInt().rename("land_cover").clip(small_aoi)
    )

    summary = workflows.summarize_baseline_layers(
        sites_fc, layers, scale_continuous=1000, scale_landcover=1000
    )
    props = summary.first().toDictionary().getInfo()

    assert props["dem_mean"] == pytest.approx(100)
    assert props["bii_all_mean"] == pytest.approx(0.7)
    assert props["tree_cover_area_m2"] > 0
    assert props["shrubland_area_m2"] == 0


@pytest.mark.ee
def test_summarize_baseline_layers_skips_landcover_areas_when_disabled(
    ee_session, small_aoi
):
    sites_fc = ee_session.FeatureCollection(
        [ee_session.Feature(small_aoi, {"site_name": "A"})]
    )
    layers = {
        name: ee_session.Image.constant(1).toFloat().rename(band).clip(small_aoi)
        for name, band in [
            ("dem", "elevation"),
            ("slope", "slope"),
            ("hillshade", "hillshade"),
            ("canopy_height", "canopy_height"),
            ("soil_carbon", "soil_carbon"),
            ("bii_all", "BII All"),
        ]
    }

    summary = workflows.summarize_baseline_layers(
        sites_fc,
        layers,
        scale_continuous=1000,
        include_landcover_areas=False,
    )
    props = summary.first().toDictionary().getInfo()

    assert "tree_cover_area_m2" not in props
    assert props["dem_mean"] == pytest.approx(1)


# --------------------------------------------------------------------------- #
# export_baseline_layers
# --------------------------------------------------------------------------- #


def test_export_baseline_layers_uses_defaults_and_returns_task_dict():
    layers = {name: f"img_{name}" for name in workflows.BASELINE_LAYER_NAMES}
    mock_tasks = [f"task_{i}" for i in range(len(layers))]

    with patch.object(
        workflows, "export_image_list_to_drive", return_value=mock_tasks
    ) as export_list:
        result = workflows.export_baseline_layers(layers, "mock_aoi", "my_folder")

    export_list.assert_called_once()
    kwargs = export_list.call_args.kwargs
    assert kwargs["aoi"] == "mock_aoi"
    assert kwargs["folder"] == "my_folder"
    assert kwargs["crs"] == workflows.DEFAULT_CRS
    assert [name for _, name, _ in kwargs["images"]] == workflows.BASELINE_LAYER_NAMES
    assert [scale for _, _, scale in kwargs["images"]] == [30] * len(layers)
    assert result == dict(zip(workflows.BASELINE_LAYER_NAMES, mock_tasks))


def test_export_baseline_layers_selective_layer_names():
    layers = {name: f"img_{name}" for name in workflows.BASELINE_LAYER_NAMES}

    with patch.object(
        workflows, "export_image_list_to_drive", return_value=["t1", "t2"]
    ) as export_list:
        result = workflows.export_baseline_layers(
            layers, "aoi", "folder", layer_names=["dem", "slope"]
        )

    kwargs = export_list.call_args.kwargs
    assert [name for _, name, _ in kwargs["images"]] == ["dem", "slope"]
    assert result == {"dem": "t1", "slope": "t2"}


def test_export_baseline_layers_rejects_unknown_layer_name():
    with pytest.raises(ValueError, match="not found in layers dict"):
        workflows.export_baseline_layers(
            {"dem": "img"}, "aoi", "folder", layer_names=["dem", "bogus"]
        )


def test_export_baseline_layers_rejects_unknown_scale_key():
    with pytest.raises(ValueError, match="not found in scale_dict"):
        workflows.export_baseline_layers(
            {"dem": "img", "slope": "img2"},
            "aoi",
            "folder",
            layer_names=["dem", "slope"],
            scale_dict={"dem": 30},
        )


# --------------------------------------------------------------------------- #
# run_baseline_assessment
# --------------------------------------------------------------------------- #


def test_run_baseline_assessment_wires_collaborators_and_exports_summary():
    mock_sites = MagicMock(name="sites_fc")

    with (
        patch.object(workflows, "get_sites_geometry", return_value="aoi") as get_geom,
        patch.object(
            workflows, "build_baseline_layers", return_value="layers"
        ) as build_layers,
        patch.object(
            workflows, "summarize_baseline_layers", return_value="summary_fc"
        ) as summarize,
        patch.object(workflows, "export_baseline_layers") as export_layers,
        patch.object(workflows, "export_table_to_drive") as export_table,
    ):
        result = workflows.run_baseline_assessment(
            sites_fc=mock_sites,
            export_folder="my_folder",
        )

    get_geom.assert_called_once_with(mock_sites)
    build_layers.assert_called_once_with("aoi")
    summarize.assert_called_once_with(
        sites_fc=mock_sites,
        layers="layers",
        scale_continuous=30,
        scale_landcover=10,
        crs=workflows.DEFAULT_CRS,
        tile_scale=4,
        include_landcover_areas=True,
    )
    export_layers.assert_called_once_with(
        layers="layers",
        aoi="aoi",
        export_folder="my_folder",
        layer_names=None,
        scale_dict=None,
        crs=workflows.DEFAULT_CRS,
    )
    export_table.assert_called_once_with(
        collection="summary_fc",
        description="baseline_site_summaries",
        folder="my_folder",
        fileNamePrefix="baseline_site_summaries",
    )
    assert result == ("layers", "summary_fc")


def test_run_baseline_assessment_skips_summary_export_when_disabled():
    mock_sites = MagicMock(name="sites_fc")

    with (
        patch.object(workflows, "get_sites_geometry", return_value="aoi"),
        patch.object(workflows, "build_baseline_layers", return_value="layers"),
        patch.object(workflows, "summarize_baseline_layers", return_value="summary_fc"),
        patch.object(workflows, "export_baseline_layers"),
        patch.object(workflows, "export_table_to_drive") as export_table,
    ):
        workflows.run_baseline_assessment(
            sites_fc=mock_sites,
            export_folder="my_folder",
            export_summary=False,
        )

    export_table.assert_not_called()


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
