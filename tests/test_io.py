"""Tests for eetools.io export helpers.

These are mock-based on purpose: the real functions submit batch export tasks
against the caller's Earth Engine account (``task.start()``), which must never
happen in a test run. We patch the module-level ``ee`` so we can assert the
wrapper builds the right export request and starts the task, with no side
effects.
"""

from unittest.mock import MagicMock, patch

import pytest

from eetools import io


@pytest.fixture
def mock_ee():
    """Patch ``eetools.io.ee`` with a MagicMock for the duration of a test."""
    with patch.object(io, "ee") as mock:
        yield mock


def test_export_image_to_drive_builds_request_and_starts(mock_ee):
    image = MagicMock(name="image")
    unmasked = image.unmask.return_value
    task = mock_ee.batch.Export.image.toDrive.return_value

    result = io.export_image_to_drive(
        image=image,
        aoi="AOI",
        description="desc",
        folder="folder",
        file_prefix="prefix",
        scale=10,
        crs="EPSG:32736",
    )

    # nodata is unmasked to -9999 before export, with sameFootprint=False so the
    # fill extends past a .clip(aoi) polygon's true boundary to the full export
    # region -- otherwise EE's exporter silently fills that fringe with a
    # hard-coded literal 0 instead of the registered NoData value.
    image.unmask.assert_called_once_with(-9999, sameFootprint=False)

    _, kwargs = mock_ee.batch.Export.image.toDrive.call_args
    assert kwargs["image"] is unmasked
    assert kwargs["description"] == "desc"
    assert kwargs["folder"] == "folder"
    assert kwargs["fileNamePrefix"] == "prefix"
    assert kwargs["region"] == "AOI"
    assert kwargs["scale"] == 10
    assert kwargs["crs"] == "EPSG:32736"
    assert kwargs["maxPixels"] == 1e13
    assert kwargs["formatOptions"] == {"noData": -9999}

    task.start.assert_called_once()
    assert result is task


def test_export_image_to_drive_unmask_extends_past_clip_footprint(mock_ee):
    """Regression test: sameFootprint=False must be passed to unmask().

    Without it, Image.unmask() only fills masked-but-in-footprint pixels --
    pixels outside a .clip(aoi) polygon but inside the export's rectangular
    region are left with no computed value, and Earth Engine's exporter then
    silently fills that fringe with a hard-coded literal 0 instead of the
    registered NoData value (bypassing formatOptions.noData entirely).
    """
    image = MagicMock(name="image")

    io.export_image_to_drive(
        image=image,
        aoi="AOI",
        description="desc",
        folder="folder",
        file_prefix="prefix",
        scale=10,
    )

    args, kwargs = image.unmask.call_args
    assert args == (-9999,)
    assert kwargs == {"sameFootprint": False}


def test_export_image_to_drive_default_crs(mock_ee):
    io.export_image_to_drive(
        image=MagicMock(),
        aoi="AOI",
        description="d",
        folder="f",
        file_prefix="p",
        scale=30,
    )
    _, kwargs = mock_ee.batch.Export.image.toDrive.call_args
    assert kwargs["crs"] == "EPSG:4326"


def test_export_image_to_asset_uses_explicit_project(mock_ee):
    task = mock_ee.batch.Export.image.toAsset.return_value
    task.status.return_value = {"state": "READY"}

    io.export_image_to_asset(
        image=MagicMock(),
        aoi="AOI",
        asset_id="projects/p/assets/foo",
        description="d",
        scale=10,
        project_id="explicit-project",
    )

    _, kwargs = mock_ee.batch.Export.image.toAsset.call_args
    assert kwargs["assetId"] == "projects/p/assets/foo"
    assert kwargs["region"] == "AOI"
    assert kwargs["scale"] == 10
    assert kwargs["maxPixels"] == 1e13
    task.start.assert_called_once()


def test_export_image_to_asset_falls_back_to_configured_project(mock_ee):
    # When project_id is None the function consults get_project(); patch the
    # name as imported into the io module.
    with patch.object(io, "get_project", return_value="configured-project") as gp:
        io.export_image_to_asset(
            image=MagicMock(),
            aoi="AOI",
            asset_id="projects/p/assets/foo",
            description="d",
        )
    gp.assert_called_once()


def test_export_table_to_drive_builds_request(mock_ee):
    task = mock_ee.batch.Export.table.toDrive.return_value
    task.status.return_value = {"state": "READY"}

    io.export_table_to_drive(
        collection="FC",
        description="d",
        folder="folder",
        fileNamePrefix="prefix",
        fileFormat="GeoJSON",
    )

    _, kwargs = mock_ee.batch.Export.table.toDrive.call_args
    assert kwargs["collection"] == "FC"
    assert kwargs["folder"] == "folder"
    assert kwargs["fileNamePrefix"] == "prefix"
    assert kwargs["fileFormat"] == "GeoJSON"
    task.start.assert_called_once()


def test_image_export_suffix_prefers_date(mock_ee):
    image = MagicMock()
    image.toDictionary.return_value.getInfo.return_value = {
        "date": "2023-01-01T00:00:00",
        "year": 2023,
    }
    suffix = io._image_export_suffix(image, index=0)
    # Colons are replaced so the suffix is filename-safe.
    assert suffix == "2023-01-01T00-00-00"


def test_image_export_suffix_falls_back_to_year(mock_ee):
    image = MagicMock()
    image.toDictionary.return_value.getInfo.return_value = {
        "date": None,
        "year": 2021,
    }
    assert io._image_export_suffix(image, index=0) == "2021"


def test_image_export_suffix_falls_back_to_index(mock_ee):
    image = MagicMock()
    image.toDictionary.return_value.getInfo.return_value = {"date": None, "year": None}
    assert io._image_export_suffix(image, index=7) == "007"


def test_export_image_collection_to_drive_one_task_per_image(mock_ee):
    # Two images in the collection; control size and per-image properties.
    collection = mock_ee.ImageCollection.return_value.sort.return_value
    collection.size.return_value.getInfo.return_value = 2
    collection.toList.return_value.get.side_effect = lambda i: f"img{i}"

    image_proxy = mock_ee.Image.return_value
    image_proxy.toDictionary.return_value.getInfo.side_effect = [
        {"year": 2022},
        {"year": 2023},
    ]

    with patch.object(io, "export_image_to_drive") as export_one:
        export_one.side_effect = ["task0", "task1"]
        tasks = io.export_image_collection_to_drive(
            collection="raw",
            aoi="AOI",
            folder="folder",
            file_prefix="site",
            scale=10,
        )

    assert tasks == ["task0", "task1"]
    assert export_one.call_count == 2
    prefixes = [call.kwargs["file_prefix"] for call in export_one.call_args_list]
    assert prefixes == ["site_2022", "site_2023"]


def test_export_image_list_to_drive_one_task_per_image(mock_ee):
    img_a, img_b = MagicMock(name="img_a"), MagicMock(name="img_b")

    with patch.object(io, "export_image_to_drive") as export_one:
        export_one.side_effect = ["task_a", "task_b"]
        tasks = io.export_image_list_to_drive(
            images=[(img_a, "layer_a", 10), (img_b, "layer_b", 30)],
            aoi="AOI",
            folder="folder",
            crs="EPSG:32734",
        )

    assert tasks == ["task_a", "task_b"]
    assert export_one.call_count == 2

    # First tuple -> name used for both description and file prefix; per-tuple scale.
    first = export_one.call_args_list[0].kwargs
    assert first["image"] is img_a
    assert first["aoi"] == "AOI"
    assert first["description"] == "layer_a"
    assert first["file_prefix"] == "layer_a"
    assert first["scale"] == 10
    assert first["folder"] == "folder"
    assert first["crs"] == "EPSG:32734"

    second = export_one.call_args_list[1].kwargs
    assert second["image"] is img_b
    assert second["file_prefix"] == "layer_b"
    assert second["scale"] == 30


def test_export_image_list_to_drive_empty_list(mock_ee):
    with patch.object(io, "export_image_to_drive") as export_one:
        tasks = io.export_image_list_to_drive(images=[], aoi="AOI", folder="folder")
    assert tasks == []
    export_one.assert_not_called()


def test_export_site_collections_delegates_per_site(mock_ee):
    site_collections = {
        "site_a": {"collection": "col_a", "aoi": "aoi_a"},
        "site_b": {"collection": "col_b", "aoi": "aoi_b"},
    }

    with patch.object(io, "export_image_collection_to_drive") as export_col:
        export_col.side_effect = [["t_a"], ["t_b"]]
        result = io.export_site_collections_to_drive(
            site_collections=site_collections,
            folder="folder",
            scale=10,
        )

    assert result == {"site_a": ["t_a"], "site_b": ["t_b"]}
    assert export_col.call_count == 2
    # Each site's id is threaded through as the file prefix.
    first_call = export_col.call_args_list[0]
    assert first_call.kwargs["file_prefix"] == "site_a"


def test_check_ee_task_status_returns_matching_task(mock_ee):
    task = MagicMock()
    task.id = "TASK123"
    task.status.return_value = {"state": "COMPLETED", "progress": 1.0}
    mock_ee.batch.Task.list.return_value = [task]

    status = io.check_ee_task_status("TASK123")

    assert status == {"state": "COMPLETED", "progress": 1.0}


def test_check_ee_task_status_returns_none_when_absent(mock_ee):
    other = MagicMock()
    other.id = "OTHER"
    mock_ee.batch.Task.list.return_value = [other]

    assert io.check_ee_task_status("MISSING") is None


@pytest.fixture
def mock_storage():
    """Patch ``eetools.io.storage`` with a MagicMock for the duration of a test."""
    with patch.object(io, "storage") as mock:
        yield mock


def test_upload_geotiff_to_asset_rejects_non_tif(mock_ee, mock_storage, tmp_path):
    bad_file = tmp_path / "raster.png"
    bad_file.write_bytes(b"data")

    with pytest.raises(ValueError, match=r"\.tif"):
        io.upload_geotiff_to_asset(
            str(bad_file), asset_id="projects/p/assets/foo", bucket="bucket"
        )

    mock_storage.Client.assert_not_called()


def test_upload_geotiff_to_asset_uploads_and_ingests(mock_ee, mock_storage, tmp_path):
    tif_file = tmp_path / "raster.tif"
    tif_file.write_bytes(b"data")

    mock_ee.data.newTaskId.return_value = ["req-1"]
    mock_ee.data.startIngestion.return_value = {"id": "TASK1"}

    result = io.upload_geotiff_to_asset(
        str(tif_file),
        asset_id="projects/p/assets/foo",
        bucket="my-bucket",
        gcs_folder="folder",
        project_id="explicit-project",
    )

    # Staged to GCS with EE's own persistent credentials, under the expected blob path.
    mock_storage.Client.assert_called_once_with(
        project="explicit-project",
        credentials=mock_ee.data.get_persistent_credentials.return_value,
    )
    blob = mock_storage.Client.return_value.bucket.return_value.blob
    blob.assert_called_once_with("folder/raster.tif")
    blob.return_value.upload_from_filename.assert_called_once_with(str(tif_file))

    # Ingestion request references the uploaded gs:// URI.
    args, kwargs = mock_ee.data.startIngestion.call_args
    assert args[0] == "req-1"
    manifest = args[1]
    assert manifest["name"] == "projects/p/assets/foo"
    assert manifest["tilesets"] == [
        {"sources": [{"uris": ["gs://my-bucket/folder/raster.tif"]}]}
    ]
    assert kwargs["allow_overwrite"] is False
    assert result == {"id": "TASK1"}


def test_upload_geotiff_to_asset_falls_back_to_configured_project(
    mock_ee, mock_storage, tmp_path
):
    tif_file = tmp_path / "raster.tif"
    tif_file.write_bytes(b"data")

    with patch.object(io, "get_project", return_value="configured-project") as gp:
        io.upload_geotiff_to_asset(
            str(tif_file), asset_id="projects/p/assets/foo", bucket="bucket"
        )

    gp.assert_called_once()
    assert mock_storage.Client.call_args.kwargs["project"] == "configured-project"


def test_upload_shapefile_to_asset_rejects_non_shp(mock_ee, mock_storage, tmp_path):
    bad_file = tmp_path / "vector.geojson"
    bad_file.write_text("{}")

    with pytest.raises(ValueError, match=r"\.shp"):
        io.upload_shapefile_to_asset(
            str(bad_file), asset_id="projects/p/assets/foo", bucket="bucket"
        )

    mock_storage.Client.assert_not_called()


def test_upload_shapefile_to_asset_requires_sidecars(mock_ee, mock_storage, tmp_path):
    shp_file = tmp_path / "vector.shp"
    shp_file.write_bytes(b"data")
    # No .shx/.dbf sidecar files written alongside it.

    with pytest.raises(ValueError, match="sidecar"):
        io.upload_shapefile_to_asset(
            str(shp_file), asset_id="projects/p/assets/foo", bucket="bucket"
        )

    mock_storage.Client.assert_not_called()


def test_upload_shapefile_to_asset_uploads_shp_and_sidecars(
    mock_ee, mock_storage, tmp_path
):
    shp_file = tmp_path / "vector.shp"
    shp_file.write_bytes(b"data")
    (tmp_path / "vector.shx").write_bytes(b"data")
    (tmp_path / "vector.dbf").write_bytes(b"data")
    (tmp_path / "vector.prj").write_text("PROJCS")
    # An unlisted sidecar extension must be ignored, not uploaded.
    (tmp_path / "vector.xml").write_text("<xml/>")

    mock_ee.data.newTaskId.return_value = ["req-1"]
    mock_ee.data.startTableIngestion.return_value = {"id": "TASK2"}

    result = io.upload_shapefile_to_asset(
        str(shp_file),
        asset_id="projects/p/assets/foo",
        bucket="my-bucket",
        gcs_folder="folder",
        project_id="explicit-project",
    )

    blob = mock_storage.Client.return_value.bucket.return_value.blob
    uploaded_names = {call.args[0] for call in blob.call_args_list}
    assert uploaded_names == {
        "folder/vector.shp",
        "folder/vector.shx",
        "folder/vector.dbf",
        "folder/vector.prj",
    }

    # Only the .shp URI is passed to the table manifest -- EE resolves sidecars
    # automatically from files sharing its GCS location.
    args, kwargs = mock_ee.data.startTableIngestion.call_args
    assert args[0] == "req-1"
    manifest = args[1]
    assert manifest["name"] == "projects/p/assets/foo"
    assert manifest["sources"] == [{"uris": ["gs://my-bucket/folder/vector.shp"]}]
    assert kwargs["allow_overwrite"] is False
    assert result == {"id": "TASK2"}


# --------------------------------------------------------------------------- #
# Cloud Storage exports                                                        #
# --------------------------------------------------------------------------- #
def _gcs_kwargs(mock_ee):
    _, kwargs = mock_ee.batch.Export.image.toCloudStorage.call_args
    return kwargs


def test_export_image_to_gcs_builds_task_config_with_transform(mock_ee):
    image = mock_ee.Image.return_value
    transform = [10, 0, 500000, 0, -10, 9600000]

    task = io.export_image_to_gcs(
        "img",
        "REGION",
        bucket="my-bucket",
        file_name_prefix="proj/ndvi/s01_ndvi",
        crs="EPSG:32737",
        crs_transform=transform,
        dtype="int16",
        skip_existing=False,
    )

    image.unmask.assert_called_once_with(-32768, sameFootprint=False)
    image.unmask.return_value.toInt16.assert_called_once()
    kwargs = _gcs_kwargs(mock_ee)
    assert kwargs["image"] is image.unmask.return_value.toInt16.return_value
    assert kwargs["bucket"] == "my-bucket"
    assert kwargs["fileNamePrefix"] == "proj/ndvi/s01_ndvi"
    assert kwargs["description"] == "proj_ndvi_s01_ndvi"
    assert kwargs["region"] == "REGION"
    assert kwargs["crs"] == "EPSG:32737"
    assert kwargs["crsTransform"] == [10.0, 0.0, 500000.0, 0.0, -10.0, 9600000.0]
    assert "scale" not in kwargs
    assert kwargs["fileFormat"] == "GeoTIFF"
    assert kwargs["formatOptions"] == {"cloudOptimized": True, "noData": -32768}
    assert kwargs["maxPixels"] == 1e10
    task.start.assert_called_once()
    assert task is mock_ee.batch.Export.image.toCloudStorage.return_value


def test_export_image_to_gcs_scale_and_float_default_nodata(mock_ee):
    image = mock_ee.Image.return_value
    io.export_image_to_gcs(
        "img",
        "REGION",
        bucket="b",
        file_name_prefix="p",
        scale=30,
        cloud_optimized=False,
        skip_existing=False,
    )
    image.unmask.assert_called_once_with(-9999, sameFootprint=False)
    image.unmask.return_value.toFloat.assert_called_once()
    kwargs = _gcs_kwargs(mock_ee)
    assert kwargs["scale"] == 30
    assert "crsTransform" not in kwargs
    assert kwargs["crs"] == "EPSG:4326"
    assert kwargs["formatOptions"] == {"cloudOptimized": False, "noData": -9999}


@pytest.mark.parametrize(
    "dtype, cast_method, nodata",
    [("uint8", "toUint8", 0), ("uint16", "toUint16", 65535)],
)
def test_export_image_to_gcs_dtype_defaults(mock_ee, dtype, cast_method, nodata):
    image = mock_ee.Image.return_value
    io.export_image_to_gcs(
        "img",
        "R",
        bucket="b",
        file_name_prefix="p",
        scale=10,
        dtype=dtype,
        skip_existing=False,
    )
    image.unmask.assert_called_once_with(nodata, sameFootprint=False)
    getattr(image.unmask.return_value, cast_method).assert_called_once()
    assert _gcs_kwargs(mock_ee)["formatOptions"]["noData"] == nodata


def test_export_image_to_gcs_explicit_nodata_overrides_default(mock_ee):
    io.export_image_to_gcs(
        "img",
        "R",
        bucket="b",
        file_name_prefix="p",
        scale=10,
        dtype="int16",
        nodata=-1,
        skip_existing=False,
    )
    assert _gcs_kwargs(mock_ee)["formatOptions"]["noData"] == -1


@pytest.mark.parametrize(
    "grid",
    [{}, {"scale": 10, "crs_transform": [10, 0, 0, 0, -10, 0]}],
    ids=["neither", "both"],
)
def test_export_image_to_gcs_scale_and_transform_mutually_exclusive(mock_ee, grid):
    with pytest.raises(ValueError, match="exactly one"):
        io.export_image_to_gcs(
            "img", "R", bucket="b", file_name_prefix="p", skip_existing=False, **grid
        )
    mock_ee.batch.Export.image.toCloudStorage.assert_not_called()


def test_export_image_to_gcs_rejects_bad_transform_and_dtype(mock_ee):
    with pytest.raises(ValueError, match="6 elements"):
        io.export_image_to_gcs(
            "img",
            "R",
            bucket="b",
            file_name_prefix="p",
            crs_transform=[10, 0, 0],
            skip_existing=False,
        )
    with pytest.raises(ValueError, match="dtype"):
        io.export_image_to_gcs(
            "img",
            "R",
            bucket="b",
            file_name_prefix="p",
            scale=10,
            dtype="int8",
            skip_existing=False,
        )


def test_export_image_to_gcs_start_false_returns_unstarted_task(mock_ee):
    task = io.export_image_to_gcs(
        "img",
        "R",
        bucket="b",
        file_name_prefix="p",
        scale=10,
        skip_existing=False,
        start=False,
    )
    task.start.assert_not_called()


@pytest.mark.parametrize(
    "existing",
    ["proj/p.tif", "proj/p-0000000000-0000000000.tif"],
    ids=["single-file", "first-shard"],
)
def test_export_image_to_gcs_skips_when_object_exists(mock_ee, existing):
    with patch.object(
        io, "gcs_object_exists", side_effect=lambda b, n: n == existing
    ) as exists:
        result = io.export_image_to_gcs(
            "img", "R", bucket="b", file_name_prefix="proj/p", scale=10
        )
    assert result is None
    assert exists.call_args_list[0].args == ("b", "proj/p.tif")
    mock_ee.batch.Export.image.toCloudStorage.assert_not_called()


def test_export_image_to_gcs_exports_when_nothing_exists(mock_ee):
    with patch.object(io, "gcs_object_exists", return_value=False) as exists:
        task = io.export_image_to_gcs(
            "img", "R", bucket="b", file_name_prefix="proj/p", scale=10
        )
    assert exists.call_count == 2
    task.start.assert_called_once()


def test_skip_existing_without_storage_raises_clear_import_error(mock_ee):
    with patch.object(io, "storage", None):
        with pytest.raises(ImportError, match=r"eetools\[gcs\]"):
            io.export_image_to_gcs(
                "img", "R", bucket="b", file_name_prefix="p", scale=10
            )
    mock_ee.batch.Export.image.toCloudStorage.assert_not_called()


def test_skip_existing_false_needs_no_storage(mock_ee):
    with patch.object(io, "storage", None):
        task = io.export_image_to_gcs(
            "img", "R", bucket="b", file_name_prefix="p", scale=10, skip_existing=False
        )
    task.start.assert_called_once()


def test_gcs_object_exists_uses_ee_credentials(mock_ee, mock_storage):
    blob = mock_storage.Client.return_value.bucket.return_value.blob
    blob.return_value.exists.return_value = True

    assert io.gcs_object_exists("bkt", "a/b.tif", project_id="proj") is True
    mock_storage.Client.assert_called_once_with(
        project="proj",
        credentials=mock_ee.data.get_persistent_credentials.return_value,
    )
    mock_storage.Client.return_value.bucket.assert_called_once_with("bkt")
    blob.assert_called_once_with("a/b.tif")


def test_export_table_to_gcs_builds_task(mock_ee):
    task = io.export_table_to_gcs(
        "FC",
        bucket="b",
        file_name_prefix="proj/rain/daily v2",
        file_format="CSV",
        selectors=["date", "region_id", "ppt_mm"],
        skip_existing=False,
    )
    _, kwargs = mock_ee.batch.Export.table.toCloudStorage.call_args
    assert kwargs["collection"] == "FC"
    assert kwargs["bucket"] == "b"
    assert kwargs["fileNamePrefix"] == "proj/rain/daily_v2"
    assert kwargs["fileFormat"] == "CSV"
    assert kwargs["selectors"] == ["date", "region_id", "ppt_mm"]
    task.start.assert_called_once()


def test_export_table_to_gcs_skip_and_bad_format(mock_ee):
    with patch.object(io, "gcs_object_exists", return_value=True) as exists:
        result = io.export_table_to_gcs(
            "FC", bucket="b", file_name_prefix="t", file_format="GeoJSON"
        )
    assert result is None
    exists.assert_called_once_with("b", "t.geojson")
    with pytest.raises(ValueError, match="file_format"):
        io.export_table_to_gcs(
            "FC", bucket="b", file_name_prefix="t", file_format="XLSX"
        )
    mock_ee.batch.Export.table.toCloudStorage.assert_not_called()


def test_sanitize_file_name_prefix():
    assert io.sanitize_file_name_prefix("/a//b c/d(1)/") == "a/b_c/d_1_"
    with pytest.raises(ValueError, match="empty"):
        io.sanitize_file_name_prefix("///")


def test_sanitize_task_description_deterministic_and_bounded():
    assert io.sanitize_task_description("proj/ndvi/s 01") == "proj_ndvi_s_01"
    long_a = "x" * 120 + "_site_a"
    long_b = "x" * 120 + "_site_b"
    out_a = io.sanitize_task_description(long_a)
    assert len(out_a) == 100
    assert out_a == io.sanitize_task_description(long_a)
    assert out_a != io.sanitize_task_description(long_b)


def test_snapped_transform_rounds_origin_outward(mock_ee):
    ring = [
        [500012.3, 9600001.0],
        [500987.0, 9600001.0],
        [500987.0, 9601995.2],
        [500012.3, 9601995.2],
        [500012.3, 9600001.0],
    ]
    coords = mock_ee.Geometry.return_value.bounds.return_value.coordinates
    coords.return_value.getInfo.return_value = [ring]

    transform = io.snapped_transform("REGION", "EPSG:32737", 10)

    assert transform == [10.0, 0.0, 500010.0, 0.0, -10.0, 9602000.0]
    mock_ee.Projection.assert_called_once_with("EPSG:32737")
    mock_ee.Geometry.return_value.bounds.assert_called_once_with(
        1, mock_ee.Projection.return_value
    )


def test_snapped_transform_rejects_non_positive_scale(mock_ee):
    with pytest.raises(ValueError, match="scale"):
        io.snapped_transform("REGION", "EPSG:32737", 0)


def test_fetch_stack_dispatches_to_gcs_batch(mock_ee):
    with patch.object(io, "export_image_to_gcs", return_value="TASK") as export:
        result = io.fetch_stack("R", "IMG", bucket="b", file_name_prefix="p", scale=10)
    assert result == "TASK"
    export.assert_called_once_with(
        "IMG", "R", bucket="b", file_name_prefix="p", scale=10
    )
    with pytest.raises(ValueError, match="backend"):
        io.fetch_stack("R", "IMG", backend="compute_pixels")


@pytest.mark.ee
def test_snapped_transform_live_grid_origin(ee_session):
    import ee as real_ee

    region = real_ee.Geometry.Rectangle([39.20, -4.30, 39.22, -4.28])
    transform = io.snapped_transform(region, "EPSG:32737", 10)

    scale, _, x0, _, neg_scale, y0 = transform
    assert (scale, neg_scale) == (10.0, -10.0)
    assert x0 % 10 == 0 and y0 % 10 == 0
    # Origin is outside (top-left of) the region's bounds in UTM 37S.
    ring = region.bounds(1, real_ee.Projection("EPSG:32737")).coordinates().getInfo()[0]
    assert x0 <= min(p[0] for p in ring) < x0 + 10
    assert y0 - 10 < max(p[1] for p in ring) <= y0


# --------------------------------------------------------------------------- #
# Per-site composite stacks + band manifest                                    #
# --------------------------------------------------------------------------- #
RING = [
    [500012.0, 9600001.0],
    [502003.0, 9600001.0],
    [502003.0, 9601995.0],
    [500012.0, 9601995.0],
    [500012.0, 9600001.0],
]
BANDS = ["NDVI_2022_dry_jf", "clear_count_2022_dry_jf", "0_clear_count", "EVI"]


def _manifest(**overrides):
    params = dict(
        region_id="s01",
        product="s2",
        file="s01_s2.tif",
        crs="EPSG:32737",
        crs_transform=[10, 0, 500010, 0, -10, 9602000],
        dtype="int16",
        nodata=-32768,
        bands=io.build_band_entries(["NDVI", "clear_count"]),
    )
    params.update(overrides)
    return io.build_site_manifest(**params)


def test_build_band_entries_order_scaling_and_meta_passthrough():
    meta = {
        "NDVI_2022_dry_jf": {
            "metric": "NDVI",
            "period_type": "dry_window",
            "period": "2022-01/2022-02",
            "unit": "",
            "source": "COPERNICUS/S2_SR_HARMONIZED",
            "anything_else": {"nested": [1, 2]},
        }
    }
    entries = io.build_band_entries(BANDS, band_meta=meta)

    assert [e["index"] for e in entries] == [1, 2, 3, 4]
    assert [e["name"] for e in entries] == BANDS
    # Default unscaled patterns catch bare, <metric>_<period> and toBands() names.
    assert [e["scale_factor"] for e in entries] == [10000, 1, 1, 10000]
    assert all(e["offset"] == 0 for e in entries)
    first = entries[0]
    for key, value in meta["NDVI_2022_dry_jf"].items():
        assert first[key] == value
    assert set(entries[3]) == {"index", "name", "scale_factor", "offset"}


def test_build_band_entries_exact_unscaled_and_errors():
    entries = io.build_band_entries(
        ["a", "b"], scale_factor=100, unscaled_bands=["b", "missing"]
    )
    assert [e["scale_factor"] for e in entries] == [100, 1]
    with pytest.raises(ValueError, match="not in the image"):
        io.build_band_entries(["a"], band_meta={"zzz": {"unit": "m"}})
    with pytest.raises(ValueError, match="reserved"):
        io.build_band_entries(["a"], band_meta={"a": {"scale_factor": 5}})


def test_manifest_shape_and_version():
    manifest = _manifest()
    assert manifest["manifest_version"] == 1
    assert manifest["file"] == "s01_s2.tif"
    assert manifest["crs_transform"] == [10, 0, 500010, 0, -10, 9602000]
    assert manifest["nodata"] == -32768
    assert manifest["created"].endswith("Z")
    assert manifest["params_hash"] == io.manifest_params_hash(manifest)
    from eetools import __version__

    assert manifest["eetools_version"] == __version__


def test_params_hash_stable_and_sensitive_to_every_parameter():
    base = _manifest()
    again = _manifest()
    again["created"] = "1999-01-01T00:00:00Z"
    assert io.manifest_params_hash(again) == base["params_hash"]

    variants = [
        {"region_id": "s02", "file": "s02_s2.tif"},
        {"product": "s2b"},
        {"crs": "EPSG:32736"},
        {"crs_transform": [10, 0, 500020, 0, -10, 9602000]},
        {"dtype": "float32"},
        {"nodata": -9999},
        {"bands": io.build_band_entries(["NDVI", "clear_count", "EVI"])},
        {"bands": io.build_band_entries(["NDVI", "clear_count"], scale_factor=1000)},
        {
            "bands": io.build_band_entries(
                ["NDVI", "clear_count"], band_meta={"NDVI": {"unit": "x"}}
            )
        },
    ]
    hashes = {_manifest(**v)["params_hash"] for v in variants}
    assert base["params_hash"] not in hashes
    assert len(hashes) == len(variants)


def test_scale_stack_scales_rounds_and_keeps_order(mock_ee):
    image = MagicMock(name="image")
    selected = {}
    image.select.side_effect = lambda name: selected.setdefault(name, MagicMock())
    entries = io.build_band_entries(["NDVI", "clear_count"])

    out = io._scale_stack(image, entries, "int16")

    ndvi, count = selected["NDVI"], selected["clear_count"]
    ndvi.multiply.assert_called_once_with(10000)
    ndvi.multiply.return_value.round.assert_called_once_with()
    count.multiply.assert_not_called()
    (parts,), _ = mock_ee.Image.cat.call_args
    assert parts == [ndvi.multiply.return_value.round.return_value, count]
    mock_ee.Image.cat.return_value.rename.assert_called_once_with(
        ["NDVI", "clear_count"]
    )
    assert out is mock_ee.Image.cat.return_value.rename.return_value


def test_scale_stack_float32_does_not_round(mock_ee):
    image = MagicMock(name="image")
    io._scale_stack(image, io.build_band_entries(["NDVI"]), "float32")
    image.select.return_value.multiply.return_value.round.assert_not_called()


@pytest.fixture
def site_export(mock_ee, mock_storage):
    """Patch the EE / GCS touch points of export_composites_per_site."""
    mock_ee.Image.return_value.bandNames.return_value.getInfo.return_value = [
        "NDVI_2022",
        "clear_count_2022",
    ]
    with (
        patch.object(io, "_scale_stack", return_value="STACK") as scale_stack,
        patch.object(
            io, "_region_info", return_value=[("s01", RING), ("s02", RING)]
        ) as region_info,
        patch.object(io, "_existing_state", return_value=None) as existing,
        patch.object(io, "export_image_to_gcs") as export,
        patch.object(io, "write_export_manifest") as write,
    ):
        yield {
            "ee": mock_ee,
            "scale_stack": scale_stack,
            "region_info": region_info,
            "existing": existing,
            "export": export,
            "write": write,
        }


def _run_site_export(**kwargs):
    params = dict(
        product="s2",
        bucket="bkt",
        prefix="proj/stacks",
        crs="EPSG:32737",
        scale=10,
    )
    params.update(kwargs)
    return io.export_composites_per_site("IMG", "REGIONS", **params)


def test_export_composites_per_site_one_export_and_manifest_per_region(site_export):
    meta = {"NDVI_2022": {"metric": "NDVI", "period_type": "annual", "period": "2022"}}
    records = _run_site_export(band_meta=meta)

    assert [r.region_id for r in records] == ["s01", "s02"]
    assert [r.status for r in records] == ["started", "started"]
    assert records[0].object_name == "proj/stacks/s2/s01_s2.tif"
    assert records[0].manifest_name == "proj/stacks/s2/s01_s2.json"
    site_export["region_info"].assert_called_once_with(
        "REGIONS", "site_id", "EPSG:32737"
    )

    export = site_export["export"]
    assert export.call_count == 2
    args, kwargs = export.call_args_list[0]
    assert args[0] == "STACK"
    assert kwargs["file_name_prefix"] == "proj/stacks/s2/s01_s2"
    assert kwargs["crs_transform"] == [10.0, 0.0, 500010.0, 0.0, -10.0, 9602000.0]
    assert kwargs["dtype"] == "int16"
    assert kwargs["nodata"] == -32768
    assert kwargs["skip_existing"] is False
    assert kwargs["start"] is True
    assert records[0].task is export.return_value

    manifest = records[0].manifest
    assert manifest["file"] == "s01_s2.tif"
    assert [b["name"] for b in manifest["bands"]] == ["NDVI_2022", "clear_count_2022"]
    assert [b["scale_factor"] for b in manifest["bands"]] == [10000, 1]
    assert manifest["bands"][0]["period_type"] == "annual"
    assert site_export["write"].call_count == 2
    site_export["write"].assert_any_call("bkt", records[1])


def test_export_composites_per_site_start_false_writes_no_manifest(site_export):
    records = _run_site_export(start=False, skip_existing=False)
    assert [r.status for r in records] == ["unstarted", "unstarted"]
    site_export["write"].assert_not_called()
    site_export["existing"].assert_not_called()
    assert site_export["export"].call_args.kwargs["start"] is False


@pytest.mark.parametrize("state", ["skipped", "pending"])
def test_export_composites_per_site_skips_existing(site_export, state):
    site_export["existing"].return_value = state
    records = _run_site_export()
    assert [r.status for r in records] == [state, state]
    assert all(r.task is None for r in records)
    site_export["export"].assert_not_called()
    site_export["write"].assert_not_called()
    first_call = site_export["existing"].call_args_list[0].args
    assert first_call[:2] == ("bkt", "proj/stacks/s2/s01_s2")
    assert first_call[3] is False  # overwrite


@pytest.mark.parametrize("bad", ["site-01", "site 01", "", 1.5, None])
def test_export_composites_per_site_rejects_bad_region_ids(site_export, bad):
    site_export["region_info"].return_value = [(bad, RING)]
    with pytest.raises(ValueError, match=r"A-Za-z0-9_"):
        _run_site_export()
    site_export["export"].assert_not_called()


def test_export_composites_per_site_accepts_integer_ids(site_export):
    site_export["region_info"].return_value = [(7, RING)]
    records = _run_site_export()
    assert records[0].object_name == "proj/stacks/s2/7_s2.tif"


@pytest.mark.parametrize("product", ["s2-dry", "s2.dry", "s2/dry"])
def test_export_composites_per_site_rejects_bad_product(site_export, product):
    with pytest.raises(ValueError, match="product"):
        _run_site_export(product=product)
    site_export["region_info"].assert_not_called()


def test_export_composites_per_site_rejects_duplicates_and_bad_args(site_export):
    site_export["region_info"].return_value = [("s01", RING), ("s01", RING)]
    with pytest.raises(ValueError, match="Duplicate"):
        _run_site_export()
    with pytest.raises(ValueError, match="dtype"):
        _run_site_export(dtype="uint8")
    with pytest.raises(ValueError, match="scale"):
        _run_site_export(scale=0)
    site_export["export"].assert_not_called()


def test_export_composites_per_site_needs_storage(site_export):
    with patch.object(io, "storage", None):
        with pytest.raises(ImportError, match=r"eetools\[gcs\]"):
            _run_site_export()
    site_export["export"].assert_not_called()


def test_export_composites_per_site_float32_nodata(site_export):
    records = _run_site_export(dtype="float32")
    assert site_export["export"].call_args.kwargs["nodata"] == -9999
    assert records[0].manifest["nodata"] == -9999


def test_export_composites_per_site_size_warning(site_export):
    huge = [[0.0, 0.0], [400000.0, 0.0], [400000.0, 400000.0], [0.0, 400000.0]]
    site_export["region_info"].return_value = [("big", huge)]
    with pytest.warns(UserWarning, match="GB"):
        _run_site_export()


@pytest.fixture
def gcs_state():
    """Control which objects exist and what the old manifest holds."""
    with (
        patch.object(io, "gcs_object_exists") as exists,
        patch.object(io, "_read_manifest") as read,
        patch.object(io, "_export_objects") as objects,
    ):
        yield exists, read, objects


def test_existing_state_nothing_there(gcs_state):
    exists, read, _ = gcs_state
    exists.return_value = False
    read.return_value = None
    assert io._existing_state("b", "p/s01_s2", "H", overwrite=False) is None


def test_existing_state_matching_hash(gcs_state):
    exists, read, _ = gcs_state
    read.return_value = {"params_hash": "H"}
    exists.side_effect = lambda b, n: n == "p/s01_s2-0000000000-0000000000.tif"
    assert io._existing_state("b", "p/s01_s2", "H", overwrite=False) == "skipped"
    exists.side_effect = None
    exists.return_value = False
    assert io._existing_state("b", "p/s01_s2", "H", overwrite=False) == "pending"


@pytest.mark.parametrize(
    "old, tif, reason",
    [
        ({"params_hash": "OLD"}, True, "different parameters"),
        (None, True, "no manifest"),
    ],
)
def test_existing_state_mismatch_raises_without_overwrite(gcs_state, old, tif, reason):
    exists, read, objects = gcs_state
    exists.return_value = tif
    read.return_value = old
    with pytest.raises(ValueError, match=reason):
        io._existing_state("b", "p/s01_s2", "H", overwrite=False)
    objects.assert_not_called()


def test_existing_state_overwrite_deletes_stale_objects(gcs_state):
    exists, read, objects = gcs_state
    exists.return_value = True
    read.return_value = {"params_hash": "OLD"}
    stale = [MagicMock(name="tif"), MagicMock(name="json")]
    objects.return_value = stale
    assert io._existing_state("b", "p/s01_s2", "H", overwrite=True) is None
    objects.assert_called_once_with("b", "p/s01_s2")
    for blob in stale:
        blob.delete.assert_called_once_with()


def test_export_objects_matches_only_this_stem(mock_ee, mock_storage):
    names = [
        "p/s01_s2.tif",
        "p/s01_s2.json",
        "p/s01_s2-0000000000-0000065536.tif",
        "p/s01_s2b.tif",
        "p/s01_s2_old.tif",
    ]
    blobs = []
    for name in names:
        blob = MagicMock()
        blob.name = name
        blobs.append(blob)
    mock_storage.Client.return_value.list_blobs.return_value = blobs
    found = [b.name for b in io._export_objects("b", "p/s01_s2")]
    assert found == names[:3]


def test_write_and_read_manifest_round_trip(mock_ee, mock_storage):
    blob = mock_storage.Client.return_value.bucket.return_value.blob.return_value
    record = io.ExportRecord(
        "s01", "p/s01_s2.tif", "p/s01_s2.json", {"manifest_version": 1}, None, "started"
    )
    uri = io.write_export_manifest("b", record)
    assert uri == "gs://b/p/s01_s2.json"
    (payload,) = blob.upload_from_string.call_args.args
    assert blob.upload_from_string.call_args.kwargs == {
        "content_type": "application/json"
    }

    blob.exists.return_value = True
    blob.download_as_text.return_value = payload
    assert io._read_manifest("b", "p/s01_s2.json") == {"manifest_version": 1}
    blob.exists.return_value = False
    assert io._read_manifest("b", "p/s01_s2.json") is None


@pytest.mark.ee
def test_scale_stack_live_values_and_types(ee_session, first_value):
    image = ee_session.Image.constant([0.25, 3, -0.123456]).rename(
        ["NDVI", "clear_count", "NDMI"]
    )
    image = image.addBands(image.select("clear_count").toUint16(), overwrite=True)
    entries = io.build_band_entries(image.bandNames().getInfo())
    stack = io._scale_stack(image, entries, "int16").toInt16()

    assert stack.bandNames().getInfo() == ["NDVI", "clear_count", "NDMI"]
    assert first_value(stack, "NDVI") == 2500
    assert first_value(stack, "clear_count") == 3
    assert first_value(stack, "NDMI") == -1235  # rounded, not truncated


# Grid-aligned 2 km window as it comes back from EE after a round trip through
# EPSG:4326 (measured 2026-11-12): edges ~3e-10 m off the grid lines.
ALIGNED_RING = [
    [523229.9999999997, 9524000.0],
    [525230.0000000003, 9524000.0],
    [525230.0000000003, 9526000.0000000002],
    [523229.9999999997, 9526000.0000000002],
    [523229.9999999997, 9524000.0],
]


def test_transform_from_ring_tolerates_float_noise_on_grid_lines():
    transform = io._transform_from_ring(ALIGNED_RING, 10)
    assert transform == [10.0, 0.0, 523230.0, 0.0, -10.0, 9526000.0]
    assert io._grid_shape(ALIGNED_RING, transform) == (200, 200)


def test_transform_from_ring_still_snaps_real_offsets_outward():
    # 1 mm (1e-4 px) west of the grid line is a real offset, not noise.
    ring = [[523229.999, 9525995.0], [523250.0, 9525990.001]]
    transform = io._transform_from_ring(ring, 10)
    assert transform[2] == 523220.0
    assert transform[5] == 9526000.0
    assert io._grid_shape(ring, transform) == (3, 1)


def test_grid_region_is_inset_planar_rectangle(mock_ee):
    transform = [10.0, 0.0, 523230.0, 0.0, -10.0, 9526000.0]
    region = io._grid_region(transform, (200, 100), "EPSG:32737")
    mock_ee.Geometry.Rectangle.assert_called_once_with(
        [523232.5, 9525002.5, 525227.5, 9525997.5], "EPSG:32737", False
    )
    assert region is mock_ee.Geometry.Rectangle.return_value


def test_export_composites_per_site_exports_over_grid_region(site_export):
    site_export["region_info"].return_value = [("s01", ALIGNED_RING)]
    _run_site_export()
    rectangle = site_export["ee"].Geometry.Rectangle
    rectangle.assert_called_once_with(
        [523232.5, 9524002.5, 525227.5, 9525997.5], "EPSG:32737", False
    )
    args, kwargs = site_export["export"].call_args
    assert args[1] is rectangle.return_value
    assert kwargs["crs_transform"] == [10.0, 0.0, 523230.0, 0.0, -10.0, 9526000.0]
    site_export["ee"].FeatureCollection.return_value.filter.assert_not_called()


# --------------------------------------------------------------------------- #
# Stale-pending manifests: task_id lookup (0.7.2)                              #
# --------------------------------------------------------------------------- #
def test_write_manifest_adds_unhashed_task_id(mock_ee, mock_storage):
    blob = mock_storage.Client.return_value.bucket.return_value.blob.return_value
    manifest = _manifest()
    v070_hash = manifest["params_hash"]
    task = MagicMock(id="TASK123")
    record = io.ExportRecord(
        "s01", "p/s01_s2.tif", "p/s01_s2.json", manifest, task, "started"
    )
    io.write_export_manifest("b", record)

    import json

    (payload,) = blob.upload_from_string.call_args.args
    written = json.loads(payload)
    assert written["task_id"] == "TASK123"
    assert record.manifest["task_id"] == "TASK123"
    # Not hashed: same hash as a 0.7.0 manifest for the same parameters.
    assert io.manifest_params_hash(written) == v070_hash
    assert written["params_hash"] == v070_hash
    assert "task_id" not in io._HASHED_MANIFEST_KEYS


@pytest.fixture
def pending_manifest(gcs_state):
    """A matching manifest with a task_id and no tif."""
    exists, read, objects = gcs_state
    exists.return_value = False
    read.return_value = {"params_hash": "H", "task_id": "T1"}
    stale = [MagicMock(name="json")]
    objects.return_value = stale
    with patch.object(io, "_task_state") as state:
        yield state, objects, stale


@pytest.mark.parametrize("task_state", ["PENDING", "RUNNING", None, "WEIRD"])
def test_existing_state_live_or_unknown_task_is_pending(pending_manifest, task_state):
    state, objects, _ = pending_manifest
    state.return_value = task_state
    assert io._existing_state("b", "p/s01_s2", "H", overwrite=False) == "pending"
    state.assert_called_once_with("T1")
    objects.assert_not_called()


@pytest.mark.parametrize("task_state", ["FAILED", "CANCELLING", "CANCELLED"])
def test_existing_state_dead_task_deletes_manifest_for_resubmit(
    pending_manifest, task_state
):
    state, objects, stale = pending_manifest
    state.return_value = task_state
    assert io._existing_state("b", "p/s01_s2", "H", overwrite=False) is None
    objects.assert_called_once_with("b", "p/s01_s2")
    stale[0].delete.assert_called_once_with()


def test_existing_state_succeeded_without_tif_raises(pending_manifest):
    state, objects, _ = pending_manifest
    state.return_value = "SUCCEEDED"
    with pytest.raises(RuntimeError, match="succeeded but its tif is missing"):
        io._existing_state("b", "p/s01_s2", "H", overwrite=False)
    objects.assert_not_called()


def test_existing_state_manifest_without_task_id_stays_pending(gcs_state, caplog):
    exists, read, objects = gcs_state
    exists.return_value = False
    read.return_value = {"params_hash": "H"}
    with patch.object(io, "_task_state") as state, caplog.at_level("INFO"):
        assert io._existing_state("b", "p/s01_s2", "H", overwrite=False) == "pending"
    state.assert_not_called()
    objects.assert_not_called()
    assert "no task_id" in caplog.text


def test_task_state_reads_operation_and_handles_missing(mock_ee):
    import ee as real_ee

    mock_ee.EEException = real_ee.EEException
    mock_ee.data.getOperation.return_value = {"metadata": {"state": "FAILED"}}
    with patch.object(io, "get_project", return_value="proj"):
        assert io._task_state("T1") == "FAILED"
        mock_ee.data.getOperation.assert_called_once_with("projects/proj/operations/T1")
        mock_ee.data.getOperation.side_effect = real_ee.EEException("not found")
        assert io._task_state("T1") is None
    with patch.object(io, "get_project", return_value=None):
        assert io._task_state("T1") is None


def test_export_composites_per_site_resubmits_dead_task(mock_ee, mock_storage):
    """End to end through _existing_state: FAILED task -> manifest deleted, re-export."""
    mock_ee.Image.return_value.bandNames.return_value.getInfo.return_value = ["NDVI"]
    stale = MagicMock(name="json")
    with (
        patch.object(io, "_scale_stack", return_value="STACK"),
        patch.object(io, "_region_info", return_value=[("s01", RING)]),
        patch.object(io, "gcs_object_exists", return_value=False),
        patch.object(io, "_export_objects", return_value=[stale]),
        patch.object(io, "_task_state", return_value="FAILED"),
        patch.object(io, "export_image_to_gcs") as export,
        patch.object(io, "write_export_manifest") as write,
        patch.object(io, "_read_manifest", return_value=None) as read,
    ):
        params = dict(product="s2", bucket="b", prefix="p", crs="EPSG:32737", scale=10)
        first = io.export_composites_per_site("IMG", "REGIONS", **params)
        read.return_value = dict(first[0].manifest, task_id="DEAD")
        records = io.export_composites_per_site("IMG", "REGIONS", **params)
    assert records[0].status == "started"
    stale.delete.assert_called_once_with()
    assert export.call_count == 2
    assert write.call_count == 2
