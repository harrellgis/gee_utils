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
