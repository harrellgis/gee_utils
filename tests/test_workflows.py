"""Tests for eetools.workflows.

Both functions are thin orchestration wrappers: all collaborators are mocked so
the tests are pure Python with no Earth Engine session required.  Validation errors
that originate inside the compositing layer (bad composite_stat / season_months)
propagate without being caught, so those paths are also verified here without mocks
on the compositing functions.
"""

from dataclasses import replace
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
        patch.object(workflows, "check_region_pixel_coverage") as check_px,
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
            scale=30,
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
        mock_composites, mock_sites, ["NDVI", "EVI"], 30, None, None, 4
    )
    # region size checked against the chosen scale before reducing
    check_px.assert_called_once_with(mock_sites, 30, 4, stacklevel=3)
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
        patch.object(workflows, "check_region_pixel_coverage") as check_px,
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
            min_pixels_per_region=9,
        )

    reduce_fc.assert_called_once_with(
        "composites", mock_sites, ["NDVI"], 10, mock_reducer, ["year", "date"], 2
    )
    check_px.assert_called_once_with(mock_sites, 10, 9, stacklevel=3)


def _run_site_timeseries_minimal(**kwargs):
    return workflows.run_site_timeseries(
        sites_fc=MagicMock(name="sites_fc"),
        collection_builder=MagicMock(return_value="col"),
        bands=["NDVI"],
        start_date="2021-01-01",
        end_date="2022-01-01",
        export_folder="f",
        file_prefix="p",
        **kwargs,
    )


def test_run_site_timeseries_requires_scale():
    """scale has no default: callers must choose the accuracy/EECU trade-off."""
    with pytest.raises(TypeError, match="scale"):
        _run_site_timeseries_minimal()


def test_run_seasonal_site_timeseries_requires_scale():
    with pytest.raises(TypeError, match="scale"):
        workflows.run_seasonal_site_timeseries(
            sites_fc=MagicMock(),
            collection_builder=MagicMock(),
            bands=["NDVI"],
            start_date="2018-01-01",
            end_date="2023-01-01",
            start_year=2018,
            end_year=2022,
            season_months=(3, 5),
            season_name="wet",
            export_folder="f",
            file_prefix="p",
        )


def test_run_site_timeseries_skips_pixel_check_when_disabled():
    with (
        patch.object(workflows, "get_sites_geometry", return_value="aoi"),
        patch.object(workflows, "build_period_composites", return_value="comp"),
        patch.object(workflows, "image_collection_to_region_stats_fc"),
        patch.object(workflows, "export_table_to_drive"),
        patch.object(workflows, "check_region_pixel_coverage") as check_px,
    ):
        _run_site_timeseries_minimal(scale=30, min_pixels_per_region=None)

    check_px.assert_not_called()


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
                scale=30,
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
        patch.object(workflows, "check_region_pixel_coverage") as check_px,
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
            scale=30,
            composite_stat="median",
        )

    get_geom.assert_called_once_with(mock_sites)
    mock_builder.assert_called_once_with(mock_aoi, "2018-01-01", "2023-01-01")
    build.assert_called_once_with(
        mock_collection, ["NDVI", "NBR"], 2018, 2022, (3, 5), "wet", "median"
    )
    reduce_fc.assert_called_once_with(
        mock_composites, mock_sites, ["NDVI", "NBR"], 30, None, None, 4
    )
    check_px.assert_called_once_with(mock_sites, 30, 4, stacklevel=3)
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
                scale=30,
                composite_stat="max",
            )


# --------------------------------------------------------------------------- #
# Baseline layer registry + build_baseline_layers
# --------------------------------------------------------------------------- #
ALL_LAYERS = workflows.BASELINE_LAYER_NAMES + workflows.BASELINE_EXTRA_LAYER_NAMES


def test_registry_is_complete_and_ordered():
    registry = workflows.BASELINE_LAYER_REGISTRY
    assert list(registry) == ALL_LAYERS
    for name, spec in registry.items():
        assert spec.name == name
        assert callable(spec.builder)
        for field in ("dataset_id", "version", "licence", "unit", "band"):
            assert getattr(spec, field), f"{name}.{field} is empty"
        assert spec.export_scale_m > 0 and spec.native_scale_m > 0
        assert spec.resample in ("mean", "mode", "nearest")
        # categorical layers aggregate by mode, continuous ones by mean
        assert spec.resample == ("mode" if spec.categorical else "mean")


def test_registry_spec_values_from_build_spec():
    registry = workflows.BASELINE_LAYER_REGISTRY
    chm = registry["canopy_height"]
    assert (chm.export_scale_m, chm.native_scale_m, chm.resample) == (10, 1, "mean")
    lossyear = registry["forest_lossyear"]
    assert lossyear.categorical and lossyear.export_scale_m == 30
    assert lossyear.dataset_id == "UMD/hansen/global_forest_change_2025_v1_13"
    wrb = registry["soilgrids_wrb"]
    assert wrb.categorical and wrb.export_scale_m == 230
    assert wrb.licence == "CC BY 4.0"
    ghm = registry["human_modification"]
    assert ghm.dataset_id == "TNC/HM/v3/90m_s" and ghm.export_scale_m == 90
    assert "vecea" not in registry


@pytest.fixture
def mock_builders():
    """Replace every registry builder with a mock returning a named MagicMock."""
    images = {name: MagicMock(name=f"img_{name}") for name in ALL_LAYERS}
    builders = {name: MagicMock(return_value=images[name]) for name in ALL_LAYERS}
    registry = {
        name: replace(spec, builder=builders[name])
        for name, spec in workflows.BASELINE_LAYER_REGISTRY.items()
    }
    with patch.object(workflows, "BASELINE_LAYER_REGISTRY", registry):
        yield builders, images


def test_build_baseline_layers_default_nine_unclipped(mock_builders):
    builders, images = mock_builders
    layers = workflows.build_baseline_layers("REGION")

    assert list(layers) == workflows.BASELINE_LAYER_NAMES
    for name in workflows.BASELINE_LAYER_NAMES:
        builders[name].assert_called_once_with("REGION")
        assert layers[name] is images[name]
        images[name].clip.assert_not_called()
    for name in workflows.BASELINE_EXTRA_LAYER_NAMES:
        builders[name].assert_not_called()


def test_build_baseline_layers_subset_and_clip(mock_builders):
    _, images = mock_builders
    layers = workflows.build_baseline_layers(
        "REGION", layers=["forest_lossyear", "dem"], clip=True
    )
    assert list(layers) == ["forest_lossyear", "dem"]
    images["dem"].clip.assert_called_once_with("REGION")
    assert layers["dem"] is images["dem"].clip.return_value


def test_build_baseline_layers_errors_and_deprecated_aoi(mock_builders):
    with pytest.raises(ValueError, match="Unknown baseline layer"):
        workflows.build_baseline_layers("REGION", layers=["vecea"])
    with pytest.raises(ValueError, match="needs a region"):
        workflows.build_baseline_layers(clip=True)
    with pytest.raises(ValueError, match="not both"):
        workflows.build_baseline_layers("REGION", aoi="AOI")
    with pytest.warns(DeprecationWarning, match="region"):
        layers = workflows.build_baseline_layers(aoi="AOI", layers=["dem"])
    assert list(layers) == ["dem"]


def test_registry_builders_wire_sensor_loaders():
    terrain = MagicMock(name="terrain")
    bii = MagicMock(name="bii")
    with (
        patch.object(workflows, "get_terrain", return_value=terrain) as get_terrain,
        patch.object(workflows, "get_canopy_height") as get_chm,
        patch.object(workflows, "get_land_cover") as get_lc,
        patch.object(workflows, "get_soil_carbon") as get_soil,
        patch.object(workflows, "get_bii_image", return_value=bii) as get_bii,
        patch.object(workflows, "get_forest_2000") as get_f2000,
        patch.object(workflows, "get_forest_loss_image") as get_floss,
        patch.object(workflows, "get_forest_loss_year_image") as get_fyear,
        patch.object(workflows, "get_soilgrids_wrb") as get_wrb,
        patch.object(workflows, "get_human_modification") as get_ghm,
    ):
        layers = workflows.build_baseline_layers("REGION", layers=ALL_LAYERS)

    assert layers["dem"] is terrain.select.return_value
    assert [c.args for c in terrain.select.call_args_list] == [
        ("elevation",),
        ("slope",),
        ("hillshade",),
    ]
    assert get_terrain.call_count == 3
    get_chm.assert_called_once_with(pin_projection=True)
    get_lc.assert_called_once_with(pin_projection=True)
    get_soil.assert_called_once_with()
    get_bii.assert_called_once_with("1km")
    bii.select.assert_called_once_with("BII All")
    for loader in (get_f2000, get_floss, get_fyear, get_wrb, get_ghm):
        loader.assert_called_once_with()  # unclipped; clipping is opt-in


def test_baseline_layer_register_rows():
    rows = workflows.baseline_layer_register(["canopy_height", "soilgrids_wrb"])
    assert [r["name"] for r in rows] == ["canopy_height", "soilgrids_wrb"]
    assert rows[0]["export_scale_m"] == 10 and rows[0]["licence"] == "CC BY 4.0"
    assert set(rows[1]) == {
        "name",
        "dataset_id",
        "version",
        "licence",
        "unit",
        "band",
        "categorical",
        "native_scale_m",
        "export_scale_m",
        "resample",
    }
    assert [r["name"] for r in workflows.baseline_layer_register()] == ALL_LAYERS
    with pytest.raises(ValueError, match="Unknown"):
        workflows.baseline_layer_register(["bogus"])


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
def _mock_layers(names):
    return {name: MagicMock(name=f"img_{name}") for name in names}


def test_export_baseline_layers_drive_uses_registry_scales_and_resample():
    layers = _mock_layers(workflows.BASELINE_LAYER_NAMES)
    mock_tasks = [f"task_{i}" for i in range(len(layers))]

    with (
        patch.object(
            workflows, "export_image_list_to_drive", return_value=mock_tasks
        ) as export_list,
        patch.object(workflows, "ee") as mock_ee,
    ):
        result = workflows.export_baseline_layers(layers, "mock_aoi", "my_folder")

    kwargs = export_list.call_args.kwargs
    assert kwargs["aoi"] == "mock_aoi"
    assert kwargs["folder"] == "my_folder"
    assert kwargs["crs"] == workflows.DEFAULT_CRS
    names = [name for _, name, _ in kwargs["images"]]
    assert names == workflows.BASELINE_LAYER_NAMES
    scales = dict((name, scale) for _, name, scale in kwargs["images"])
    assert scales == {
        name: workflows.BASELINE_LAYER_REGISTRY[name].export_scale_m for name in names
    }
    images = {name: image for image, name, _ in kwargs["images"]}
    # canopy_height: 1 m -> 10 m, aggregated by mean with enough maxPixels.
    chm = layers["canopy_height"]
    chm.reduceResolution.assert_called_once_with(
        reducer=mock_ee.Reducer.mean.return_value, maxPixels=121
    )
    assert images["canopy_height"] is chm.reduceResolution.return_value
    # Everything else exports at its native scale: no aggregation.
    for name in names:
        if name != "canopy_height":
            layers[name].reduceResolution.assert_not_called()
            assert images[name] is layers[name]
    assert result == dict(zip(names, mock_tasks))


def test_export_baseline_layers_coarser_categorical_uses_mode():
    layers = _mock_layers(["land_cover", "dem"])
    with (
        patch.object(workflows, "export_image_list_to_drive", return_value=["a", "b"]),
        patch.object(workflows, "ee") as mock_ee,
    ):
        workflows.export_baseline_layers(
            layers, "aoi", "folder", scale_dict={"land_cover": 30, "dem": 90}
        )
    layers["land_cover"].reduceResolution.assert_called_once_with(
        reducer=mock_ee.Reducer.mode.return_value, maxPixels=16
    )
    layers["dem"].reduceResolution.assert_called_once_with(
        reducer=mock_ee.Reducer.mean.return_value, maxPixels=16
    )


def test_export_baseline_layers_selective_layer_names():
    layers = _mock_layers(workflows.BASELINE_LAYER_NAMES)
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


def test_export_baseline_layers_needs_scale_for_non_registry_layer():
    custom = MagicMock(name="custom")
    with pytest.raises(ValueError, match="not found in scale_dict"):
        workflows.export_baseline_layers(
            {"custom": custom}, "aoi", "folder", scale_dict={"dem": 30}
        )
    with patch.object(
        workflows, "export_image_list_to_drive", return_value=["t"]
    ) as export_list:
        workflows.export_baseline_layers(
            {"custom": custom}, "aoi", "folder", scale_dict={"custom": 50}
        )
    ((image, name, scale),) = export_list.call_args.kwargs["images"]
    assert (image, name, scale) == (custom, "custom", 50)


def test_export_baseline_layers_gcs_config():
    layers = _mock_layers(["forest_lossyear", "human_modification", "bii_all"])
    transforms = {
        30: [30.0, 0.0, 0.0, 0.0, -30.0, 0.0],
        90: [90.0, 0.0, 0.0, 0.0, -90.0, 0.0],
        1000: [1000.0, 0.0, 0.0, 0.0, -1000.0, 0.0],
    }
    with (
        patch.object(
            workflows, "snapped_transform", side_effect=lambda r, c, s: transforms[s]
        ) as snap,
        patch.object(workflows, "export_image_to_gcs") as export_gcs,
        patch.object(workflows, "export_image_list_to_drive") as export_drive,
    ):
        result = workflows.export_baseline_layers(
            layers,
            "BBOX",
            dest="gcs",
            bucket="bkt",
            prefix="proj/baseline",
            crs="EPSG:32737",
            skip_existing=False,
            start=False,
        )

    export_drive.assert_not_called()
    assert snap.call_count == 3
    calls = {c.kwargs["file_name_prefix"]: c for c in export_gcs.call_args_list}
    lossyear = calls["proj/baseline/forest_lossyear"]
    assert lossyear.args == (layers["forest_lossyear"], "BBOX")
    assert lossyear.kwargs["dtype"] == "int16"
    assert lossyear.kwargs["crs_transform"] == transforms[30]
    ghm = calls["proj/baseline/human_modification"]
    assert ghm.kwargs["dtype"] == "float32"
    assert ghm.kwargs["crs_transform"] == transforms[90]
    assert ghm.kwargs["crs"] == "EPSG:32737" and ghm.kwargs["bucket"] == "bkt"
    assert ghm.kwargs["skip_existing"] is False and ghm.kwargs["start"] is False
    assert set(result) == set(layers)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"dest": "gcs", "prefix": "p", "crs": "EPSG:32737"}, "bucket and prefix"),
        ({"dest": "gcs", "bucket": "b", "prefix": "p"}, "projected crs"),
        ({"dest": "s3"}, "dest must be"),
        ({}, "export_folder is required"),
    ],
)
def test_export_baseline_layers_destination_errors(kwargs, match):
    with (
        patch.object(workflows, "export_image_to_gcs") as export_gcs,
        patch.object(workflows, "export_image_list_to_drive") as export_drive,
    ):
        with pytest.raises(ValueError, match=match):
            workflows.export_baseline_layers(_mock_layers(["dem"]), "BBOX", **kwargs)
    export_gcs.assert_not_called()
    export_drive.assert_not_called()


def test_export_baseline_layers_deprecated_aoi_keyword():
    with patch.object(workflows, "export_image_list_to_drive", return_value=["t"]):
        with pytest.warns(DeprecationWarning, match="region"):
            workflows.export_baseline_layers(
                _mock_layers(["dem"]), aoi="AOI", export_folder="folder"
            )
    with pytest.raises(ValueError, match="region is required"):
        workflows.export_baseline_layers(_mock_layers(["dem"]), export_folder="f")


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
    build_layers.assert_called_once_with("aoi", clip=True)
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
        region="aoi",
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
                scale=30,
            )
