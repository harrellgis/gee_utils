import fnmatch
import hashlib
import json
import logging
import math
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
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
    MANIFEST_RESERVED_BAND_KEYS,
    MANIFEST_VERSION,
    SITE_EXPORT_DTYPE_BYTES,
    SITE_EXPORT_DTYPES,
    SITE_EXPORT_NAME_PATTERN,
    SITE_EXPORT_SCALE_FACTOR,
    SITE_EXPORT_UNSCALED_BANDS,
    SITE_EXPORT_WARN_BYTES,
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
        prefix: Raw object prefix, e.g. ``"proj/ndvi/site 01_ndvi"``.

    Returns:
        str. The sanitised prefix, e.g. ``"proj/ndvi/site_01_ndvi"``.

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
    return _transform_from_ring(ring, scale)


def _transform_from_ring(ring: list[list[float]], scale: float) -> list[float]:
    xs = [pt[0] for pt in ring]
    ys = [pt[1] for pt in ring]
    x0 = round(math.floor(min(xs) / scale) * scale, 10)
    y0 = round(math.ceil(max(ys) / scale) * scale, 10)
    return [float(scale), 0.0, float(x0), 0.0, -float(scale), float(y0)]


def _ring_pixel_count(ring: list[list[float]], transform: list[float]) -> int:
    scale = transform[0]
    width = math.ceil((max(pt[0] for pt in ring) - transform[2]) / scale)
    height = math.ceil((transform[5] - min(pt[1] for pt in ring)) / scale)
    return max(width, 0) * max(height, 0)


def gcs_object_exists(bucket: str, name: str, project_id: str | None = None) -> bool:
    """Check whether an object exists in a Cloud Storage bucket.

    Uses EE's persistent credentials (the same account the exports write with).

    Args:
        bucket: Bucket name, without ``gs://``.
        name: Full object name within the bucket, e.g. ``"proj/ndvi/s01_ndvi.tif"``.
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


# --------------------------------------------------------------------------- #
# Per-site composite stacks + band manifest                                    #
# --------------------------------------------------------------------------- #
_SITE_NAME_RE = re.compile(f"^{SITE_EXPORT_NAME_PATTERN}$")
_HASHED_MANIFEST_KEYS = (
    "manifest_version",
    "file",
    "region_id",
    "product",
    "crs",
    "crs_transform",
    "dtype",
    "nodata",
    "bands",
)


@dataclass(frozen=True)
class ExportRecord:
    """Outcome of one per-site export from ``export_composites_per_site``.

    Attributes:
        region_id: The region's ``id_property`` value.
        object_name: Object name of the (unsharded) tif in the bucket.
        manifest_name: Object name of the JSON band manifest next to it.
        manifest: The manifest dict (written to ``manifest_name`` when the task starts).
        task: The export task, or None when skipped / pending.
        status: ``"started"``, ``"unstarted"`` (``start=False``; call ``write_export_manifest`` after starting it), ``"skipped"`` (tif + matching manifest exist) or ``"pending"`` (matching manifest without its tif: an earlier export is still running or failed).
    """

    region_id: str
    object_name: str
    manifest_name: str
    manifest: dict[str, Any]
    task: ee.batch.Task | None
    status: Literal["started", "unstarted", "skipped", "pending"]


def _validate_site_name(value: Any, what: str) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or not _SITE_NAME_RE.match(value):
        raise ValueError(
            f"{what} {value!r} must match {SITE_EXPORT_NAME_PATTERN} (letters, digits, "
            "underscores; no hyphens, which clash with EE's shard suffix)."
        )
    return value


def _plain_number(value: float) -> float | int:
    return int(value) if float(value).is_integer() else float(value)


def build_band_entries(
    bands: Sequence[str],
    *,
    scale_factor: float = SITE_EXPORT_SCALE_FACTOR,
    unscaled_bands: Sequence[str] = SITE_EXPORT_UNSCALED_BANDS,
    band_meta: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build the manifest band entries for a stack, in band order.

    Each entry has ``index`` (1-based, following ``bands``), ``name``,
    ``scale_factor`` (1 for ``unscaled_bands``) and ``offset`` (0), then every key of
    ``band_meta[name]`` copied verbatim. eetools does not interpret those keys.

    Args:
        bands: Band names in stack order (``image.bandNames()``).
        scale_factor: Factor applied to every band not in ``unscaled_bands``.
        unscaled_bands: Bands stored as-is (``scale_factor`` 1), as exact names or case-sensitive fnmatch patterns; non-matching entries are ignored.
        band_meta: Optional ``{band name: {key: value}}`` copied into the entries.

    Returns:
        list[dict]. One manifest band entry per band.

    Raises:
        ValueError: If ``band_meta`` names a band not in ``bands``, or sets a reserved key (``index``, ``name``, ``scale_factor``, ``offset``).
    """
    meta = dict(band_meta or {})
    unknown = sorted(set(meta) - set(bands))
    if unknown:
        raise ValueError(f"band_meta names bands not in the image: {unknown}.")
    unscaled = {
        name
        for name in bands
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in unscaled_bands)
    }
    entries: list[dict[str, Any]] = []
    for index, name in enumerate(bands, start=1):
        extra = dict(meta.get(name, {}))
        clash = sorted(set(extra) & set(MANIFEST_RESERVED_BAND_KEYS))
        if clash:
            raise ValueError(
                f"band_meta[{name!r}] may not set reserved keys {clash}; "
                "they are derived from the export parameters."
            )
        factor = 1 if name in unscaled else _plain_number(scale_factor)
        entry: dict[str, Any] = {
            "index": index,
            "name": name,
            "scale_factor": factor,
            "offset": 0,
        }
        entry.update(extra)
        entries.append(entry)
    return entries


def manifest_params_hash(manifest: Mapping[str, Any]) -> str:
    """SHA-256 of a manifest's export parameters and band list.

    Hashes ``manifest_version``, ``file``, ``region_id``, ``product``, ``crs``,
    ``crs_transform``, ``dtype``, ``nodata`` and ``bands`` (with all band
    metadata), as canonical JSON. ``created``, ``eetools_version`` and the hash itself
    are excluded, so re-running with the same parameters gives the same hash. It
    says nothing about file content; integrity is the GCS MD5.

    Args:
        manifest: A manifest dict as built by ``build_site_manifest``.

    Returns:
        str. Hex SHA-256 digest.
    """
    payload = {key: manifest[key] for key in _HASHED_MANIFEST_KEYS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_site_manifest(
    *,
    region_id: str,
    product: str,
    file: str,
    crs: str,
    crs_transform: Sequence[float],
    dtype: str,
    nodata: float,
    bands: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the JSON band manifest for one per-site stack.

    Args:
        region_id: Region id the stack covers.
        product: Product name.
        file: Basename of the unsharded tif, e.g. ``"s01_s2.tif"``.
        crs: CRS code of the stack.
        crs_transform: EE-order affine ``[xScale, xShear, xTranslation, yShear, yScale, yTranslation]``.
        dtype: Stored pixel type.
        nodata: No-data value (of the stored values).
        bands: Band entries from ``build_band_entries``.

    Returns:
        dict. The manifest, including ``manifest_version``, ``eetools_version``, ``params_hash`` and a UTC ``created`` timestamp.
    """
    from eetools import __version__

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "file": file,
        "region_id": region_id,
        "product": product,
        "crs": crs,
        "crs_transform": [_plain_number(v) for v in crs_transform],
        "dtype": dtype,
        "nodata": _plain_number(nodata),
        "bands": bands,
        "eetools_version": __version__,
    }
    manifest["params_hash"] = manifest_params_hash(manifest)
    manifest["created"] = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    return manifest


def _scale_stack(
    image: ee.Image, entries: list[dict[str, Any]], dtype: str
) -> ee.Image:
    parts = []
    for entry in entries:
        band = image.select(entry["name"])
        if entry["scale_factor"] != 1:
            band = band.multiply(entry["scale_factor"])
            if dtype == "int16":
                band = band.round()  # the int16 cast would otherwise truncate
        parts.append(band)
    return ee.Image.cat(parts).rename([entry["name"] for entry in entries])


def _region_info(
    regions: ee.FeatureCollection, id_property: str, crs: str
) -> list[tuple[Any, list[list[float]]]]:
    proj = ee.Projection(crs)

    def _info(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        ring = feature.geometry().bounds(1, proj).coordinates().get(0)
        return ee.Feature(None, {"region_id": feature.get(id_property), "ring": ring})

    # One request for every region's id and bounds, instead of one per region.
    info = cast(dict, ee.FeatureCollection(regions).map(_info).getInfo())
    props = [f["properties"] for f in info["features"]]
    return [(p.get("region_id"), p["ring"]) for p in props]


def _read_manifest(bucket: str, name: str) -> dict[str, Any] | None:
    blob = _gcs_client(get_project()).bucket(bucket).blob(name)
    if not blob.exists():
        return None
    return cast(dict, json.loads(blob.download_as_text()))


def _export_objects(bucket: str, stem: str) -> list[Any]:
    pattern = re.compile(re.escape(stem) + r"(\.tif|\.json|-\d{10}-\d{10}\.tif)$")
    blobs = _gcs_client(get_project()).list_blobs(bucket, prefix=stem)
    return [blob for blob in blobs if pattern.fullmatch(blob.name)]


def write_export_manifest(bucket: str, record: ExportRecord) -> str:
    """Write an export's JSON band manifest next to its tif.

    ``export_composites_per_site`` calls this itself when it starts a task; call
    it yourself after starting a task you created with ``start=False``.

    Args:
        bucket: Bucket the stack is exported to.
        record: The record returned for that region.

    Returns:
        str. The manifest's ``gs://`` URI.

    Raises:
        ImportError: If google-cloud-storage is not installed (``eetools[gcs]``).
    """
    blob = _gcs_client(get_project()).bucket(bucket).blob(record.manifest_name)
    blob.upload_from_string(
        json.dumps(record.manifest, indent=2), content_type="application/json"
    )
    return f"gs://{bucket}/{record.manifest_name}"


def _existing_state(
    bucket: str, stem: str, params_hash: str, overwrite: bool
) -> Literal["skipped", "pending"] | None:
    tif_exists = any(
        gcs_object_exists(bucket, name)
        for name in (f"{stem}.tif", f"{stem}{GCS_FIRST_SHARD_SUFFIX}.tif")
    )
    old = _read_manifest(bucket, f"{stem}.json")
    if old is None and not tif_exists:
        return None
    if old is not None and old.get("params_hash") == params_hash:
        return "skipped" if tif_exists else "pending"
    if not overwrite:
        reason = "has no manifest" if old is None else "used different parameters"
        raise ValueError(
            f"gs://{bucket}/{stem}: the existing export {reason}. Pass "
            "overwrite=True to replace it (its tif, shards and manifest are deleted)."
        )
    for blob in _export_objects(bucket, stem):
        logger.info("Deleting stale gs://%s/%s", bucket, blob.name)
        blob.delete()
    return None


def export_composites_per_site(
    image: ee.Image,
    regions: ee.FeatureCollection,
    *,
    id_property: str = "site_id",
    product: str,
    bucket: str,
    prefix: str,
    crs: str,
    scale: float,
    dtype: Literal["int16", "float32"] = "int16",
    scale_factor: float = SITE_EXPORT_SCALE_FACTOR,
    unscaled_bands: Sequence[str] = SITE_EXPORT_UNSCALED_BANDS,
    nodata: float | None = None,
    band_meta: Mapping[str, Mapping[str, Any]] | None = None,
    skip_existing: bool = True,
    overwrite: bool = False,
    start: bool = True,
) -> list[ExportRecord]:
    """Export one multiband stack per region to Cloud Storage, each with a band manifest.

    For each region (window polygon) the whole ``image`` is exported to
    ``gs://<bucket>/<prefix>/<product>/<region_id>_<product>.tif`` on
    ``snapped_transform(region, crs, scale)``, so all products of a project share
    one grid. A JSON manifest ``<region_id>_<product>.json`` is written next to it.
    Bands not in ``unscaled_bands`` are multiplied by ``scale_factor`` (and rounded
    for int16) before the cast. eetools does not build windows or stack periods:
    pass one image (``toBands()`` or explicit ``<metric>_<period>`` renames) and the
    window polygons.

    Manifest guarantees (consumers rely on them):

    - band ``index`` is 1-based and follows ``image.bandNames()`` order;
    - ``crs_transform`` is in EE order ``[xScale, xShear, xTranslation, yShear,
      yScale, yTranslation]``, i.e. ``rasterio.Affine(*t)``, **not** the GDAL
      geotransform order;
    - decoding is ``physical = stored / scale_factor + offset``; ``offset`` is 0 in
      v0.7; ``unscaled_bands`` have ``scale_factor`` 1; ``nodata`` applies to the
      stored value;
    - every ``band_meta[<band>]`` key is copied verbatim into that band's entry
      (eetools interprets none of them); read bands by those fields, never by
      parsing names (``clear_count`` contains an underscore);
    - ``manifest_version`` (currently 1) is bumped on any breaking change;
    - the manifest is written when the task starts, so it can exist before its tif
      (treat that as pending); ``file`` is always the unsharded basename;
    - ``params_hash`` covers the export parameters and band list
      (``manifest_params_hash``), not the file content (that is the GCS MD5).

    Skip logic (``skip_existing=True``): tif (or first shard) + manifest with the same
    ``params_hash`` -> skipped; matching manifest without a tif -> pending (not
    resubmitted); anything else that exists -> ValueError, unless ``overwrite=True``,
    which deletes that region's tif, shards and manifest and exports again.
    ``skip_existing=False`` exports without any check or deletion.

    Makes two small ``getInfo`` calls (band names; all region ids and bounds).

    Args:
        image: Multiband image to export (one stack, all periods/indices).
        regions: Window polygons, one export each; ``id_property`` must be unique.
        id_property: Region property used as ``region_id`` in names and the manifest; values must match ``[A-Za-z0-9_]+``.
        product: Product name, ``[A-Za-z0-9_]+``.
        bucket: Destination bucket (no ``gs://``); always caller-supplied.
        prefix: Object prefix above ``<product>/`` (sanitised like ``sanitize_file_name_prefix``).
        crs: Output CRS, e.g. the project UTM zone.
        scale: Pixel size in ``crs`` units.
        dtype: ``"int16"`` (default) or ``"float32"``.
        scale_factor: Stored = physical x scale_factor for scaled bands (default 10000).
        unscaled_bands: Bands stored as-is, as exact names or fnmatch patterns. The default (``SITE_EXPORT_UNSCALED_BANDS``) matches ``clear_count``, ``clear_count_*`` and ``*_clear_count``; non-matching entries are ignored.
        nodata: Stored no-data value; defaults to -32768 (int16) / -9999 (float32).
        band_meta: Optional ``{band: {key: value}}`` copied into each band entry (e.g. ``metric``, ``period_type``, ``period``, ``unit``, ``source``).
        skip_existing: Apply the skip logic above (default True).
        overwrite: Replace an existing export whose parameters differ.
        start: Start the tasks and write the manifests (default True). With False, nothing is written; start each task and then call ``write_export_manifest``.

    Returns:
        list[ExportRecord]. One record per region, in collection order.

    Raises:
        ValueError: For an invalid ``product``/``region_id`` name, duplicate region ids, unknown ``dtype``, non-positive ``scale``/``scale_factor``, bad ``band_meta``, or an existing export with other parameters (without ``overwrite``).
        ImportError: If google-cloud-storage is missing and ``skip_existing`` or ``start`` needs it.
    """
    _validate_site_name(product, "product")
    if dtype not in SITE_EXPORT_DTYPES:
        raise ValueError(f"dtype must be one of {SITE_EXPORT_DTYPES}, got {dtype!r}.")
    if scale <= 0 or scale_factor <= 0:
        raise ValueError("scale and scale_factor must be > 0.")
    if (skip_existing or start) and storage is None:
        raise ImportError(_GCS_IMPORT_ERROR)
    fill = _resolve_nodata(dtype, nodata)
    base = sanitize_file_name_prefix(f"{prefix}/{product}")

    image = ee.Image(image)
    bands = cast(list, image.bandNames().getInfo())
    entries = build_band_entries(
        bands,
        scale_factor=scale_factor,
        unscaled_bands=unscaled_bands,
        band_meta=band_meta,
    )
    stack = _scale_stack(image, entries, dtype)

    infos = _region_info(regions, id_property, crs)
    region_ids = [_validate_site_name(raw, id_property) for raw, _ in infos]
    duplicates = sorted({rid for rid in region_ids if region_ids.count(rid) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {id_property} values: {duplicates}.")

    records: list[ExportRecord] = []
    for (raw_id, ring), region_id in zip(infos, region_ids):
        transform = _transform_from_ring(ring, scale)
        basename = f"{region_id}_{product}"
        stem = f"{base}/{basename}"
        manifest = build_site_manifest(
            region_id=region_id,
            product=product,
            file=f"{basename}.tif",
            crs=crs,
            crs_transform=transform,
            dtype=dtype,
            nodata=fill,
            bands=entries,
        )
        size = len(bands) * _ring_pixel_count(ring, transform)
        size *= SITE_EXPORT_DTYPE_BYTES[dtype]
        if size > SITE_EXPORT_WARN_BYTES:
            warnings.warn(
                f"{basename}: ~{size / 1e9:.1f} GB uncompressed "
                f"({len(bands)} bands); consider splitting the product.",
                stacklevel=2,
            )

        state = (
            _existing_state(bucket, stem, manifest["params_hash"], overwrite)
            if skip_existing
            else None
        )
        if state is not None:
            logger.info("%s: %s", stem, state)
            records.append(
                ExportRecord(
                    region_id, f"{stem}.tif", f"{stem}.json", manifest, None, state
                )
            )
            continue

        region = (
            ee.FeatureCollection(regions)
            .filter(ee.Filter.eq(id_property, raw_id))
            .geometry()
        )
        task = export_image_to_gcs(
            stack,
            region,
            bucket=bucket,
            file_name_prefix=stem,
            crs=crs,
            crs_transform=transform,
            dtype=dtype,
            nodata=fill,
            skip_existing=False,
            start=start,
        )
        record = ExportRecord(
            region_id,
            f"{stem}.tif",
            f"{stem}.json",
            manifest,
            task,
            "started" if start else "unstarted",
        )
        if start:
            write_export_manifest(bucket, record)
        records.append(record)
    return records
