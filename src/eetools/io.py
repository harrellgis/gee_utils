import ee

from eetools._config import get_project


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
    image = image.unmask(no_data_value)

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=file_prefix,
        region=aoi,
        scale=scale,
        crs=crs,
        maxPixels=1e13,
        formatOptions={"noData": no_data_value},
    )

    task.start()
    return task


def _image_export_suffix(
    image: ee.Image,
    index: int,
    date_property: str = "date",
    fallback_property: str = "year",
) -> str:
    props = image.toDictionary([date_property, fallback_property]).getInfo()
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
    """Export every image in an ImageCollection to Google Drive as individual GeoTIFF tasks.

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

    n = collection.size().getInfo()
    images = collection.toList(n)
    tasks = []

    for i in range(n):
        image = ee.Image(images.get(i))

        props_to_get = (
            [fallback_property]
            if date_property is None
            else [date_property, fallback_property]
        )
        props = image.toDictionary(props_to_get).getInfo()

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
        maxPixels=1e13,
        crs=crs,
    )
    task.start()
    print("Export started:", task.status())


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
    print("Export started:", task.status())


def check_ee_task_status(task_id: str) -> dict | None:
    """Query an Earth Engine batch task by ID and print its current status fields.

    Args:
        task_id: The EE task ID string to look up.

    Returns:
        dict containing the task status fields (state, description, progress, error_message), or None if the task ID is not found.
    """
    tasks = ee.batch.Task.list()

    for task in tasks:
        if task.id == task_id:
            status = task.status()
            print("Task ID:", task_id)
            print("State:", status.get("state"))
            print("Description:", status.get("description"))
            print("Progress:", status.get("progress", "N/A"))
            print("Error Message:", status.get("error_message", "None"))
            return status

    print("Task ID not found.")
    return None
