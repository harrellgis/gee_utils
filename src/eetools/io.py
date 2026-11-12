import hashlib
import logging
import math
import re
from pathlib import Path
from typing import Any, Literal, cast

import ee

from eetools._config import get_project
from eetools.constants import (
    EE_DESCRIPTION_ALLOWED_CHARS,
    EE_DESCRIPTION_MAX_LENGTH,
    GCS_EXPORT_DTYPES,
    GCS_EXPORT_MAX_PIXELS,
    GCS_EXPORT_NODATA,
    GCS_FIRST_SHARD_SUFFIX,
    GCS_PREFIX_ALLOWED_CHARS,
    GCS_TABLE_FORMATS,
)

try:  # google-cloud-storage is needed for skip-if-exists checks and uploads only.
    from google.cloud import storage
except ImportError:  # pragma: no cover - exercised by patching in tests
    storage = None

logger = logging.getLogger(__name__)

_GCS_IMPORT_ERROR = (
    "google-cloud-storage is required for this operation (skip_existing checks, "
    "manifests and uploads). Install it with `uv add eetools[gcs]` or "
    "`pip install google-cloud-storage`, or pass skip_existing=False."
)

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


def _gcs_client(project_id: str | None) -> Any:
    """Build a Cloud Storage client on EE's own persistent credentials.

    Raises:
        ImportError: If google-cloud-storage is not installed.
    """
    if storage is None:
        raise ImportError(_GCS_IMPORT_ERROR)
    return storage.Client(
        project=project_id, credentials=ee.data.get_persistent_credentials()
    )


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
    client = _gcs_client(project_id)
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


# --------------------------------------------------------------------------- #
# Cloud Storage exports                                                        #
# --------------------------------------------------------------------------- #
# EE batch exports write to the bucket with the credentials of the account that
# authenticated `ee`. That account needs storage.objects.create on the bucket (a
# project owner has it; a service account needs roles/storage.objectAdmin).

_DESCRIPTION_DISALLOWED = re.compile(f"[^{EE_DESCRIPTION_ALLOWED_CHARS}]")
_PREFIX_DISALLOWED = re.compile(f"[^{GCS_PREFIX_ALLOWED_CHARS}]")
_FETCH_BACKENDS = ("gcs_batch",)


def sanitize_file_name_prefix(prefix: str) -> str:
    """Make a Cloud Storage object prefix safe and deterministic.

    Characters outside ``A-Za-z0-9._-/`` become ``_`` and repeated or
    leading/trailing slashes are removed. EE appends the extension itself.

    Args:
        prefix: Raw object prefix, e.g. ``"tgbs/ndvi/site 01_ndvi"``.

    Returns:
        str. The sanitised prefix, e.g. ``"tgbs/ndvi/site_01_ndvi"``.

    Raises:
        ValueError: If nothing usable is left after sanitising.
    """
    cleaned = _PREFIX_DISALLOWED.sub("_", prefix)
    cleaned = re.sub(r"/+", "/", cleaned).strip("/")
    if not cleaned:
        raise ValueError(f"file_name_prefix {prefix!r} is empty after sanitising.")
    return cleaned


def sanitize_task_description(description: str) -> str:
    """Make an EE batch task description valid and deterministic.

    EE allows at most 100 characters from ``A-Za-z0-9.,:;_-``. Other characters
    (including ``/``) become ``_``. A description that is still too long is cut and
    suffixed with 8 hex characters of its SHA-1, so two long names sharing a
    beginning still get distinct, reproducible descriptions (no timestamps).

    Args:
        description: Raw description, typically the object prefix.

    Returns:
        str. A valid description of at most 100 characters.
    """
    cleaned = _DESCRIPTION_DISALLOWED.sub("_", description).strip("_") or "export"
    if len(cleaned) <= EE_DESCRIPTION_MAX_LENGTH:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[: EE_DESCRIPTION_MAX_LENGTH - 9]}_{digest}"


def snapped_transform(
    region: ee.Geometry, crs: str, scale: float, max_error: float = 1
) -> list[float]:
    """Return a north-up affine transform whose origin is snapped to ``scale``.

    The origin is the region's top-left bounds corner in ``crs`` rounded outward to
    a multiple of ``scale`` (``x0 = floor(xmin / scale) * scale``,
    ``y0 = ceil(ymax / scale) * scale``), so every export of a project in the same
    ``crs`` and ``scale`` lands on one shared pixel grid, and local exactextract
    weights match EE ``reduceRegions`` on that grid.

    Makes one small ``getInfo`` call (the region bounds); never call it inside a
    ``.map()``.

    Args:
        region: Export region; only its bounds in ``crs`` are used.
        crs: Target CRS code, e.g. ``"EPSG:32737"``; ``scale`` is in its units.
        scale: Pixel size in ``crs`` units (metres for UTM). Must be > 0.
        max_error: Error margin (in ``crs`` units) for reprojecting the bounds.

    Returns:
        list[float]. ``[scale, 0, x0, 0, -scale, y0]``, for ``crs_transform``.

    Raises:
        ValueError: If ``scale`` is not positive.
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}.")
    coords = ee.Geometry(region).bounds(max_error, ee.Projection(crs)).coordinates()
    ring = cast(list, coords.getInfo())[0]  # bounds() is one non-empty ring
    xs = [pt[0] for pt in ring]
    ys = [pt[1] for pt in ring]
    x0 = round(math.floor(min(xs) / scale) * scale, 10)
    y0 = round(math.ceil(max(ys) / scale) * scale, 10)
    return [float(scale), 0.0, float(x0), 0.0, -float(scale), float(y0)]


def gcs_object_exists(bucket: str, name: str, project_id: str | None = None) -> bool:
    """Check whether an object exists in a Cloud Storage bucket.

    Uses EE's persistent credentials (the same account the exports write with).

    Args:
        bucket: Bucket name, without ``gs://``.
        name: Full object name within the bucket, e.g. ``"tgbs/ndvi/s01_ndvi.tif"``.
        project_id: GCP project for the client; falls back to the configured project.

    Returns:
        bool. True if the object exists.

    Raises:
        ImportError: If google-cloud-storage is not installed (``eetools[gcs]``).
    """
    if project_id is None:
        project_id = get_project()
    client = _gcs_client(project_id)
    return bool(client.bucket(bucket).blob(name).exists())


def _resolve_nodata(dtype: str, nodata: float | None) -> float:
    if dtype not in GCS_EXPORT_DTYPES:
        raise ValueError(f"dtype must be one of {GCS_EXPORT_DTYPES}, got {dtype!r}.")
    return GCS_EXPORT_NODATA[dtype] if nodata is None else nodata


def _cast_image(image: ee.Image, dtype: str) -> ee.Image:
    if dtype == "int16":
        return image.toInt16()
    if dtype == "uint8":
        return image.toUint8()
    if dtype == "uint16":
        return image.toUint16()
    return image.toFloat()


def export_image_to_gcs(
    image: ee.Image,
    region: ee.Geometry,
    *,
    bucket: str,
    file_name_prefix: str,
    description: str | None = None,
    scale: float | None = None,
    crs: str = "EPSG:4326",
    crs_transform: list[float] | None = None,
    dtype: Literal["int16", "uint8", "uint16", "float32"] = "float32",
    nodata: float | None = None,
    cloud_optimized: bool = True,
    max_pixels: float = GCS_EXPORT_MAX_PIXELS,
    skip_existing: bool = True,
    start: bool = True,
) -> ee.batch.Task | None:
    """Export an image to Cloud Storage as a (cloud-optimised) GeoTIFF.

    Masked pixels are filled with ``nodata`` (``unmask(..., sameFootprint=False)``,
    so the fringe between a clipped polygon and the export bbox is no-data too, not
    EE's literal 0), then the image is cast to ``dtype``. No value scaling is
    applied: scale reflectance/indices to int16 yourself, or use
    ``export_composites_per_site``.

    Large exports are split by EE into shards named ``<prefix>-<row>-<col>.tif``
    (10-digit pixel offsets; first shard ``<prefix>-0000000000-0000000000.tif``); a
    small export is a single ``<prefix>.tif``. ``skip_existing`` treats either as
    already exported; reassembling shards is the downloader's job.

    Permissions: EE writes with the credentials that authenticated ``ee``; that
    account needs ``storage.objects.create`` on the bucket (a project owner has it;
    a service account needs ``roles/storage.objectAdmin``).

    Args:
        image: Image to export.
        region: Export region.
        bucket: Destination bucket name (no ``gs://``); always caller-supplied.
        file_name_prefix: Object prefix without extension, e.g. ``"proj/ndvi/s01_ndvi"``; sanitised.
        description: Task description; defaults to the prefix. Sanitised to EE's rules (≤ 100 chars).
        scale: Pixel size in ``crs`` units. Pass exactly one of ``scale`` / ``crs_transform``.
        crs: Output CRS (default EPSG:4326).
        crs_transform: 6-element affine ``[xScale, 0, x0, 0, -yScale, y0]``; preferred over ``scale`` for a fixed project grid (see ``snapped_transform``).
        dtype: Output pixel type: ``"int16"``, ``"uint8"``, ``"uint16"`` or ``"float32"``.
        nodata: Fill value; defaults per dtype (``GCS_EXPORT_NODATA``: -32768 int16, 0 uint8, 65535 uint16, -9999 float32).
        cloud_optimized: Write a Cloud-Optimised GeoTIFF (default True).
        max_pixels: EE ``maxPixels`` cap (default 1e10).
        skip_existing: Return None without exporting when the tif (or its first shard) already exists. Needs google-cloud-storage.
        start: Start the task (default True); False returns the unstarted task.

    Returns:
        ee.batch.Task | None. The (started) export task, or None when skipped.

    Raises:
        ValueError: If both or neither of ``scale`` / ``crs_transform`` are given, ``crs_transform`` is not 6 numbers, or ``dtype`` is unknown.
        ImportError: If ``skip_existing`` is True and google-cloud-storage is missing.
    """
    if (scale is None) == (crs_transform is None):
        raise ValueError("Pass exactly one of scale or crs_transform.")
    if crs_transform is not None and len(crs_transform) != 6:
        raise ValueError(
            f"crs_transform must have 6 elements, got {len(crs_transform)}."
        )
    fill = _resolve_nodata(dtype, nodata)
    prefix = sanitize_file_name_prefix(file_name_prefix)
    task_description = sanitize_task_description(description or prefix)

    if skip_existing:
        candidates = (f"{prefix}.tif", f"{prefix}{GCS_FIRST_SHARD_SUFFIX}.tif")
        existing = next((n for n in candidates if gcs_object_exists(bucket, n)), None)
        if existing is not None:
            logger.info("Skipping export: gs://%s/%s exists", bucket, existing)
            return None

    out = _cast_image(ee.Image(image).unmask(fill, sameFootprint=False), dtype)
    grid: dict[str, Any] = (
        {"scale": scale}
        if crs_transform is None
        else {"crsTransform": [float(v) for v in crs_transform]}
    )

    task = ee.batch.Export.image.toCloudStorage(
        image=out,
        description=task_description,
        bucket=bucket,
        fileNamePrefix=prefix,
        region=region,
        crs=crs,
        maxPixels=int(max_pixels),
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": cloud_optimized, "noData": fill},
        **grid,
    )
    if start:
        task.start()
        logger.info("Export started: gs://%s/%s.tif", bucket, prefix)
    return task


def export_table_to_gcs(
    collection: ee.FeatureCollection,
    *,
    bucket: str,
    file_name_prefix: str,
    description: str | None = None,
    file_format: Literal["CSV", "GeoJSON", "SHP", "KML"] = "CSV",
    selectors: list[str] | None = None,
    skip_existing: bool = True,
    start: bool = True,
) -> ee.batch.Task | None:
    """Export a FeatureCollection to Cloud Storage as a table file.

    Args:
        collection: FeatureCollection to export.
        bucket: Destination bucket name (no ``gs://``); always caller-supplied.
        file_name_prefix: Object prefix without extension; sanitised.
        description: Task description; defaults to the prefix. Sanitised to EE's rules (≤ 100 chars).
        file_format: ``"CSV"``, ``"GeoJSON"``, ``"SHP"`` or ``"KML"``.
        selectors: Properties to write, in order; all properties if None.
        skip_existing: Return None without exporting when ``<prefix>.<ext>`` already exists. Needs google-cloud-storage.
        start: Start the task (default True); False returns the unstarted task.

    Returns:
        ee.batch.Task | None. The (started) export task, or None when skipped.

    Raises:
        ValueError: If ``file_format`` is not supported.
        ImportError: If ``skip_existing`` is True and google-cloud-storage is missing.
    """
    if file_format not in GCS_TABLE_FORMATS:
        raise ValueError(
            f"file_format must be one of {tuple(GCS_TABLE_FORMATS)}, "
            f"got {file_format!r}."
        )
    prefix = sanitize_file_name_prefix(file_name_prefix)
    task_description = sanitize_task_description(description or prefix)

    name = f"{prefix}.{GCS_TABLE_FORMATS[file_format]}"
    if skip_existing and gcs_object_exists(bucket, name):
        logger.info("Skipping export: gs://%s/%s exists", bucket, name)
        return None

    extra: dict[str, Any] = {}
    if selectors is not None:
        extra["selectors"] = list(selectors)
    task = ee.batch.Export.table.toCloudStorage(
        collection=collection,
        description=task_description,
        bucket=bucket,
        fileNamePrefix=prefix,
        fileFormat=file_format,
        **extra,
    )
    if start:
        task.start()
        logger.info("Export started: gs://%s/%s", bucket, name)
    return task


def fetch_stack(
    region: ee.Geometry,
    image: ee.Image,
    *,
    backend: Literal["gcs_batch"] = "gcs_batch",
    **kwargs: Any,
) -> ee.batch.Task | None:
    """Materialise an image stack over a region through a pluggable back-end.

    A thin dispatcher, so callers don't change when another back-end is added.
    Today the only back-end is ``"gcs_batch"`` (``export_image_to_gcs``). A
    ``computePixels`` back-end is deferred; when built it must respect the request
    limits (48 MB, 32k px per side, 1,024 bands), retry with backoff and keep
    fewer than 40 concurrent requests.

    Args:
        region: Region to materialise.
        image: Image stack to materialise.
        backend: Back-end name; only ``"gcs_batch"`` is supported.
        **kwargs: Keyword arguments for the back-end (``gcs_batch``: those of ``export_image_to_gcs``, e.g. ``bucket``, ``file_name_prefix``, ``crs``, ``crs_transform``).

    Returns:
        ee.batch.Task | None. The back-end's result (``gcs_batch``: the task, or None when skipped).

    Raises:
        ValueError: If ``backend`` is not supported.
    """
    if backend == "gcs_batch":
        return export_image_to_gcs(image, region, **kwargs)
    raise ValueError(f"backend must be one of {_FETCH_BACKENDS}, got {backend!r}.")
