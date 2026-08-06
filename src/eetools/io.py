import logging
from pathlib import Path
from typing import cast

import ee
from google.cloud import storage

from eetools._config import get_project

logger = logging.getLogger(__name__)

# Required and optional shapefile sidecar extensions, keyed off the .shp basename.
_SHAPEFILE_REQUIRED_SIDECARS = (".shx", ".dbf")
_SHAPEFILE_OPTIONAL_SIDECARS = (".prj", ".cpg", ".sbn", ".sbx")


def export_image_to_drive(
    image: ee.Image,
    aoi: ee.Geometry,
    description: str,
    folder: str,
    file_prefix: str,
    scale: int,
    crs: str = "EPSG:4326",
):
    """Export a single Earth Engine image to Google Drive, unmasking nodata to -9999.

    Args:
        image: ee.Image to export.
        aoi: Export region as ee.Geometry.
        description: Human-readable task description shown in the EE task manager.
        folder: Google Drive folder name to write the file into.
        file_prefix: Filename prefix (without extension) for the exported GeoTIFF.
        scale: Output pixel size in metres.
        crs: Coordinate reference system for the export (default EPSG:4326).

    Returns:
        ee.batch.Task. Starts the export task and returns the running task object.
    """
    no_data_value = -9999
    image = image.unmask(no_data_value, sameFootprint=False)

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=file_prefix,
        region=aoi,
        scale=scale,
        crs=crs,
        maxPixels=int(1e13),
        formatOptions={"noData": no_data_value},
    )

    task.start()
    return task


def export_image_list_to_drive(
    images: list[tuple[ee.Image, str, int]],
    aoi: ee.Geometry,
    folder: str,
    crs: str = "EPSG:4326",
) -> list:
    """Export a list of (image, name, scale) tuples to Google Drive, one task per image.

    A convenience wrapper over export_image_to_drive for the common case of exporting
    several distinct, individually-named images (e.g. a set of derived layers) to the same
    Drive folder and CRS in one call. Each tuple's name is used as both the task
    description and the output file prefix. Unlike export_image_collection_to_drive (which
    exports the images of a single ImageCollection), this takes arbitrary images that each
    carry their own filename and scale.

    Args:
        images: List of (image, export_name, scale) tuples — the ee.Image to export, the filename/description string, and the output pixel size in metres.
        aoi: Export region as ee.Geometry, shared by every image.
        folder: Google Drive folder name to write all files into.
        crs: Coordinate reference system applied to every export (default EPSG:4326).

    Returns:
        list of started ee.batch.Task objects, one per input image, in input order.
    """
    tasks = []
    for image, export_name, scale in images:
        task = export_image_to_drive(
            image=image,
            aoi=aoi,
            description=export_name,
            folder=folder,
            file_prefix=export_name,
            scale=scale,
            crs=crs,
        )
        tasks.append(task)

    return tasks


def _image_export_suffix(
    image: ee.Image,
    index: int,
    date_property: str = "date",
    fallback_property: str = "year",
) -> str:
    props = cast(dict, image.toDictionary([date_property, fallback_property]).getInfo())
    if props.get(date_property) is not None:
        return str(props[date_property]).replace(":", "-")
    if props.get(fallback_property) is not None:
        return str(props[fallback_property])
    return f"{index:03d}"


def export_image_collection_to_drive(
    collection: ee.ImageCollection,
    aoi: ee.Geometry,
    folder: str,
    file_prefix: str,
    scale: int,
    crs: str = "EPSG:4326",
    band_names: list[str] | None = None,
    sort_property: str = "system:time_start",
    description_prefix: str | None = None,
    date_property: str | None = None,
    fallback_property: str = "year",
    file_suffix: str | None = None,
) -> list:
    """Export every image in an ImageCollection to Google Drive as individual GeoTIFF
    tasks.

    Args:
        collection: ee.ImageCollection to export.
        aoi: Export region as ee.Geometry.
        folder: Google Drive folder name to write files into.
        file_prefix: Base filename prefix shared by all exported images.
        scale: Output pixel size in metres.
        crs: Coordinate reference system for the export (default EPSG:4326).
        band_names: Optional list of band names to select before exporting; all bands used if None.
        sort_property: Image property used to sort the collection before export (default system:time_start).
        description_prefix: Task description prefix; falls back to file_prefix if None.
        date_property: Image property used to build per-image filename suffixes; skipped if None.
        fallback_property: Property used when date_property is absent or None (default year).
        file_suffix: Optional string appended after the per-image suffix in the filename.

    Returns:
        list of ee.batch.Task objects, one per image, all started.
    """
    collection = ee.ImageCollection(collection).sort(sort_property)
    if band_names is not None:
        collection = collection.select(band_names)

    n = cast(int, collection.size().getInfo())
    images = collection.toList(n)
    tasks = []

    for i in range(n):
        image = ee.Image(images.get(i))

        props_to_get = (
            [fallback_property]
            if date_property is None
            else [date_property, fallback_property]
        )
        props = cast(dict, image.toDictionary(props_to_get).getInfo())

        if date_property is not None and props.get(date_property) is not None:
            suffix = str(props[date_property]).replace(":", "-")
        elif props.get(fallback_property) is not None:
            suffix = str(props[fallback_property])
        else:
            suffix = f"{i:03d}"

        description = f"{description_prefix or file_prefix}_{suffix}"
        prefix = (
            f"{file_prefix}_{suffix}"
            if file_suffix is None
            else f"{file_prefix}_{suffix}_{file_suffix}"
        )

        task = export_image_to_drive(
            image=image,
            aoi=aoi,
            description=description,
            folder=folder,
            file_prefix=prefix,
            scale=scale,
            crs=crs,
        )
        tasks.append(task)

    return tasks


def export_site_collections_to_drive(
    site_collections: dict[str, dict],
    folder: str,
    scale: int,
    crs: str = "EPSG:4326",
    band_names: list[str] | None = None,
    sort_property: str = "system:time_start",
    date_property: str = "date",
    fallback_property: str = "year",
    file_suffix: str = "dw_woody_cover",
) -> dict[str, list]:
    """Export one ImageCollection per site from a site dictionary to Google Drive.

    Args:
        site_collections: Dict mapping site_id strings to dicts with keys 'collection' (ee.ImageCollection) and 'aoi' (ee.Geometry).
        folder: Google Drive folder name to write all files into.
        scale: Output pixel size in metres.
        crs: Coordinate reference system for the export (default EPSG:4326).
        band_names: Optional list of band names to select before exporting; all bands used if None.
        sort_property: Image property used to sort each collection before export.
        date_property: Image property used to build per-image filename suffixes.
        fallback_property: Property used when date_property is absent (default year).
        file_suffix: String appended after the per-image suffix in every exported filename.

    Returns:
        dict mapping each site_id to its list of started ee.batch.Task objects.
    """
    tasks_by_site = {}

    for site_id, site_info in site_collections.items():
        tasks_by_site[site_id] = export_image_collection_to_drive(
            collection=site_info["collection"],
            aoi=site_info["aoi"],
            folder=folder,
            file_prefix=site_id,
            scale=scale,
            crs=crs,
            band_names=band_names,
            sort_property=sort_property,
            description_prefix=f"{site_id}_{file_suffix}",
            date_property=date_property,
            fallback_property=fallback_property,
            file_suffix=file_suffix,
        )

    return tasks_by_site


def export_image_to_asset(
    image: ee.Image,
    aoi: ee.Geometry,
    asset_id: str,
    description: str,
    scale: int = 10,
    crs: str | None = None,
    project_id: str | None = None,
) -> None:
    """Export an image to a Google Earth Engine asset folder.

    Args:
        image: ee.Image to export.
        aoi: Export region as ee.Geometry.
        asset_id: Full EE asset path for the output (e.g. projects/my-project/assets/foo).
        description: Human-readable task description shown in the EE task manager.
        scale: Output pixel size in metres (default 10).
        crs: Optional coordinate reference system; EE uses the image's native CRS if None.
        project_id: GEE project ID; falls back to the globally configured project if None.

    Returns:
        None. Starts an EE batch asset-export task and prints the initial task status.
    """
    if project_id is None:
        project_id = get_project()

    task = ee.batch.Export.image.toAsset(
        image=image,
        description=description,
        assetId=asset_id,
        region=aoi,
        scale=scale,
        maxPixels=int(1e13),
        crs=crs,
    )
    task.start()
    logger.info("Export started: %s", task.status())


def export_table_to_drive(
    collection: ee.FeatureCollection,
    description: str,
    folder: str,
    fileNamePrefix: str,
    fileFormat: str = "CSV",
) -> None:
    """Export an Earth Engine FeatureCollection to Google Drive as a tabular file.

    Args:
        collection: ee.FeatureCollection to export.
        description: Human-readable task description shown in the EE task manager.
        folder: Google Drive folder name to write the file into.
        fileNamePrefix: Filename prefix (without extension) for the exported file.
        fileFormat: Output file format, e.g. 'CSV', 'GeoJSON', 'SHP' (default CSV).

    Returns:
        None. Starts an EE batch table-export task and prints the initial task status.
    """
    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=description,
        folder=folder,
        fileNamePrefix=fileNamePrefix,
        fileFormat=fileFormat,
    )
    task.start()
    logger.info("Export started: %s", task.status())


def check_ee_task_status(task_id: str) -> dict | None:
    """Query an Earth Engine batch task by ID and log its current status fields.

    Args:
        task_id: The EE task ID string to look up.

    Returns:
        dict containing the task status fields (state, description, progress, error_message), or None if the task ID is not found.
    """
    tasks = ee.batch.Task.list()

    for task in tasks:
        if task.id == task_id:
            status = task.status()
            logger.info(
                "Task %s — state=%s, description=%s, progress=%s, error=%s",
                task_id,
                status.get("state"),
                status.get("description"),
                status.get("progress", "N/A"),
                status.get("error_message", "None"),
            )
            return status

    logger.warning("Task ID not found: %s", task_id)
    return None


def _upload_file_to_gcs(
    local_path: Path, bucket: str, blob_name: str, project_id: str | None
) -> str:
    """Upload a single local file to a GCS bucket, reusing EE's own credentials.

    Args:
        local_path: Local file to upload.
        bucket: Destination GCS bucket name.
        blob_name: Destination object name (path) within the bucket.
        project_id: GCP project to bill/authorize the upload against; falls back to credential-derived project if None.

    Returns:
        The uploaded object's gs:// URI.
    """
    client = storage.Client(
        project=project_id, credentials=ee.data.get_persistent_credentials()
    )
    client.bucket(bucket).blob(blob_name).upload_from_filename(str(local_path))
    return f"gs://{bucket}/{blob_name}"


def upload_geotiff_to_asset(
    local_path: str,
    asset_id: str,
    bucket: str,
    gcs_folder: str = "eetools_uploads",
    project_id: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Upload a local GeoTIFF and ingest it as an Earth Engine image asset.

    Earth Engine can only ingest assets from Cloud Storage, not local disk, so the
    file is first staged in `bucket` (via the same persistent credentials EE
    authorized with — no separate GCS auth needed) and then ingested from there
    with `ee.data.startIngestion`.

    Args:
        local_path: Path to the local .tif/.tiff file to upload.
        asset_id: Full EE asset path for the output (e.g. projects/my-project/assets/foo).
        bucket: Name of the GCS bucket to stage the file in; must already exist and be writable by the EE-authorized account.
        gcs_folder: GCS folder (blob prefix) within the bucket to stage the upload under.
        project_id: GEE/GCS project ID; falls back to the globally configured project if None.
        overwrite: Whether the ingestion may overwrite an existing asset at asset_id.

    Returns:
        dict with the started ingestion task's info, including the EE task id under 'id'.

    Raises:
        ValueError: If local_path does not have a .tif/.tiff extension.
    """
    path = Path(local_path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError(f"Expected a .tif/.tiff file, got: {local_path}")

    if project_id is None:
        project_id = get_project()

    gcs_uri = _upload_file_to_gcs(
        path,
        bucket=bucket,
        blob_name=f"{gcs_folder}/{path.name}",
        project_id=project_id,
    )

    request_id = ee.data.newTaskId()[0]
    manifest = {"name": asset_id, "tilesets": [{"sources": [{"uris": [gcs_uri]}]}]}
    result = ee.data.startIngestion(request_id, manifest, allow_overwrite=overwrite)
    logger.info("Image ingestion started: %s", result)
    return result


def upload_shapefile_to_asset(
    local_path: str,
    asset_id: str,
    bucket: str,
    gcs_folder: str = "eetools_uploads",
    project_id: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Upload a local shapefile (and its sidecar files) and ingest it as an Earth
    Engine table asset.

    Earth Engine can only ingest assets from Cloud Storage, not local disk, so the
    .shp and its sidecar files (.shx, .dbf, and any of .prj/.cpg/.sbn/.sbx present)
    are staged together in `bucket` (via the same persistent credentials EE
    authorized with — no separate GCS auth needed); EE resolves the sidecar files
    automatically from the .shp object's GCS location, so only the .shp URI is
    passed to `ee.data.startTableIngestion`.

    Args:
        local_path: Path to the local .shp file; sibling sidecar files sharing its basename are uploaded alongside it.
        asset_id: Full EE asset path for the output (e.g. projects/my-project/assets/foo).
        bucket: Name of the GCS bucket to stage the files in; must already exist and be writable by the EE-authorized account.
        gcs_folder: GCS folder (blob prefix) within the bucket to stage the upload under.
        project_id: GEE/GCS project ID; falls back to the globally configured project if None.
        overwrite: Whether the ingestion may overwrite an existing asset at asset_id.

    Returns:
        dict with the started ingestion task's info, including the EE task id under 'id'.

    Raises:
        ValueError: If local_path does not have a .shp extension, or a required .shx/.dbf sidecar file is missing next to it.
    """
    path = Path(local_path)
    if path.suffix.lower() != ".shp":
        raise ValueError(f"Expected a .shp file, got: {local_path}")

    missing = [
        ext
        for ext in _SHAPEFILE_REQUIRED_SIDECARS
        if not path.with_suffix(ext).exists()
    ]
    if missing:
        raise ValueError(
            f"Missing required shapefile sidecar file(s) {missing} next to {local_path}"
        )

    sidecar_paths = [
        sidecar_path
        for ext in _SHAPEFILE_REQUIRED_SIDECARS + _SHAPEFILE_OPTIONAL_SIDECARS
        if (sidecar_path := path.with_suffix(ext)).exists()
    ]

    if project_id is None:
        project_id = get_project()

    gcs_shp_uri = _upload_file_to_gcs(
        path,
        bucket=bucket,
        blob_name=f"{gcs_folder}/{path.name}",
        project_id=project_id,
    )
    for sidecar_path in sidecar_paths:
        _upload_file_to_gcs(
            sidecar_path,
            bucket=bucket,
            blob_name=f"{gcs_folder}/{sidecar_path.name}",
            project_id=project_id,
        )

    request_id = ee.data.newTaskId()[0]
    manifest = {"name": asset_id, "sources": [{"uris": [gcs_shp_uri]}]}
    result = ee.data.startTableIngestion(
        request_id, manifest, allow_overwrite=overwrite
    )
    logger.info("Table ingestion started: %s", result)
    return result
