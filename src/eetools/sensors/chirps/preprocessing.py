from collections.abc import Sequence
from typing import Literal

import ee

from eetools.compositing import build_period_composites
from eetools.constants import (
    CHIRPS_CADENCE_COLLECTIONS,
    CHIRPS_COLLECTION,
    CHIRPS_DAILY_TABLE_COLUMNS,
    CHIRPS_DAILY_VERSIONS,
    CHIRPS_NATIVE_SCALE_M,
    CHIRPS_PRECIP_BAND,
    CLIMATOLOGY_END_YEAR,
    CLIMATOLOGY_START_YEAR,
)
from eetools.io import (
    export_table_to_drive,
    export_table_to_gcs,
    sanitize_file_name_prefix,
    sanitize_task_description,
)
from eetools.utils import validate_collection_date_range
from eetools.zonal import collection_to_region_timeseries


def _chirps_collection_id(cadence: str, version: str | None = None) -> str:
    if cadence not in CHIRPS_CADENCE_COLLECTIONS:
        raise ValueError(
            f"cadence must be one of {sorted(CHIRPS_CADENCE_COLLECTIONS)}, "
            f"got '{cadence}'."
        )
    if version is None:
        return CHIRPS_CADENCE_COLLECTIONS[cadence]
    if cadence != "daily":
        raise ValueError(
            f"version applies to daily CHIRPS only (got cadence='{cadence}'); "
            "the pentad collection exists for v3 only."
        )
    if version not in CHIRPS_DAILY_VERSIONS:
        raise ValueError(
            f"version must be one of {sorted(CHIRPS_DAILY_VERSIONS)}, got '{version}'."
        )
    return CHIRPS_DAILY_VERSIONS[version]


def validate_chirps_date_range(
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    collection_id: str = CHIRPS_COLLECTION,
) -> None:
    """Validate that the requested date window overlaps available CHIRPS v3 imagery
    (the daily reanalysis by default, or the pentad collection) over the AOI.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Requested start date as ee.Date.
        end_date: Requested end date as ee.Date.
        collection_id: CHIRPS collection to validate against (default the daily product).

    Returns:
        None. Raises ValueError if the date range falls outside the available CHIRPS data extent.
    """
    validate_collection_date_range(
        collection_id=collection_id,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        sensor_label="CHIRPS imagery",
    )


def process_chirps_image(image: ee.Image) -> ee.Image:
    """Select the CHIRPS precipitation band and preserve temporal properties.

    Works for both cadences: daily images carry ``day`` (mm/day), pentad images carry
    ``pentad`` (mm per pentad). Sums over a year or more agree between cadences; daily
    means do not.

    Args:
        image: Raw CHIRPS ee.Image with the precipitation band and time properties.

    Returns:
        ee.Image containing only the precipitation band (mm/day for daily images, mm per pentad for pentad images), with system:time_start, year, month, day and pentad properties preserved where present.
    """
    source = ee.Image(image)
    image = source.select([CHIRPS_PRECIP_BAND])
    return image.copyProperties(
        source, ["system:time_start", "year", "month", "day", "pentad"]
    )


def get_chirps_collection(
    aoi: ee.Geometry,
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    cadence: Literal["daily", "pentad"] = "daily",
    version: Literal["v2", "v3_rnl", "v3_sat"] | None = None,
) -> ee.ImageCollection:
    """Build a CHIRPS precipitation collection for the AOI and date range.

    ``cadence="daily"`` (default) uses the Daily Reanalysis (mm/day);
    ``cadence="pentad"`` uses the pentad collection (mm per pentad, 72 per year).
    Annual or multi-year totals agree between the two to < 0.0001 mm, so request
    ``"pentad"`` for whole-year sums (~5x fewer image reads) and ``"daily"`` for short
    windows (pentad edges can shift a window by up to 5 days).

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as a string (e.g. '2020-01-01') or ee.Date.
        end_date: Collection end date as a string or ee.Date.
        cadence: 'daily' (UCSB-CHC/CHIRPS/V3/DAILY_RNL, default) or 'pentad' (UCSB-CHC/CHIRPS/V3/PENTAD).
        version: Daily product version (``CHIRPS_DAILY_VERSIONS``): 'v2' (UCSB-CHG/CHIRPS/DAILY), 'v3_rnl' (the default daily product) or 'v3_sat' (UCSB-CHC/CHIRPS/V3/DAILY_SAT). None (default) keeps the cadence default; daily cadence only.

    Returns:
        ee.ImageCollection of precipitation images (band 'precipitation').

    Raises:
        ValueError: If cadence or version is unknown, a version is given with cadence='pentad', or the date range falls outside the available data.
    """
    collection_id = _chirps_collection_id(cadence, version)
    start = ee.Date(start_date)
    end = ee.Date(end_date)

    validate_chirps_date_range(aoi, start, end, collection_id=collection_id)

    return (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi)
        .filterDate(start, end)
        .map(process_chirps_image)
    )


def get_chirps_climatology(
    aoi: ee.Geometry,
    start_year: int = CLIMATOLOGY_START_YEAR,
    end_year: int = CLIMATOLOGY_END_YEAR,
) -> ee.Image:
    """Mean annual CHIRPS rainfall over a fixed window, built once as one image.

    Pentads are summed over the whole window and divided by the number of years
    (identical to averaging per-year sums), at native ~5.5 km.

    Args:
        aoi: Area of interest as ee.Geometry (used for bounds and date validation).
        start_year: First year (inclusive; default 1991).
        end_year: Last year (inclusive; default 2020).

    Returns:
        ee.Image with a single 'rain_mean_annual' band (mm/yr), native projection.

    Raises:
        ValueError: If end_year < start_year or the window is outside the data.
    """
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) must be >= start_year ({start_year}).")
    col = get_chirps_collection(
        aoi, f"{start_year}-01-01", f"{end_year + 1}-01-01", cadence="pentad"
    )
    n_years = end_year - start_year + 1
    projection = ee.Image(col.first()).projection()
    return (
        col.select(CHIRPS_PRECIP_BAND)
        .sum()
        .divide(n_years)
        .rename("rain_mean_annual")
        .toFloat()
        .setDefaultProjection(projection)
    )


def get_chirps_monthly_climatology(
    aoi: ee.Geometry,
    start_year: int = CLIMATOLOGY_START_YEAR,
    end_year: int = CLIMATOLOGY_END_YEAR,
) -> ee.Image:
    """Mean monthly CHIRPS rainfall over a fixed window: 12 bands ``m01``..``m12``.

    For each calendar month the pentads of that month are summed over the whole
    window and divided by the number of years, so the 12 bands add up to
    ``get_chirps_climatology`` for the same window. Native ~5.5 km.

    Args:
        aoi: Area of interest as ee.Geometry (used for bounds and date validation).
        start_year: First year (inclusive; default 1991).
        end_year: Last year (inclusive; default 2020).

    Returns:
        ee.Image with float bands 'm01'..'m12' (mean mm/month), native projection.

    Raises:
        ValueError: If end_year < start_year or the window is outside the data.
    """
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) must be >= start_year ({start_year}).")
    col = get_chirps_collection(
        aoi, f"{start_year}-01-01", f"{end_year + 1}-01-01", cadence="pentad"
    ).select(CHIRPS_PRECIP_BAND)
    n_years = end_year - start_year + 1
    projection = ee.Image(col.first()).projection()
    months = [
        col.filter(ee.Filter.calendarRange(m, m, "month"))
        .sum()
        .divide(n_years)
        .rename(f"m{m:02d}")
        for m in range(1, 13)
    ]
    return ee.Image.cat(months).toFloat().setDefaultProjection(projection)


def get_chirps_annual_totals(
    aoi: ee.Geometry,
    start_year: int,
    end_year: int,
) -> ee.ImageCollection:
    """Annual CHIRPS rainfall totals (pentad cadence), one image per calendar year.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_year: First year (inclusive).
        end_year: Last year (inclusive).

    Returns:
        ee.ImageCollection of single-band 'precipitation' images (mm/yr), each with
        'year' and 'system:time_start' (Jan 1) properties.

    Raises:
        ValueError: If end_year < start_year or the window is outside the data.
    """
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) must be >= start_year ({start_year}).")
    col = get_chirps_collection(
        aoi, f"{start_year}-01-01", f"{end_year + 1}-01-01", cadence="pentad"
    )

    def _year_total(year: ee.Number) -> ee.Image:
        start = ee.Date.fromYMD(year, 1, 1)
        return (
            col.filterDate(start, start.advance(1, "year"))
            .select(CHIRPS_PRECIP_BAND)
            .sum()
            .rename(CHIRPS_PRECIP_BAND)
            .set({"year": year, "system:time_start": start.millis()})
        )

    return ee.ImageCollection(ee.List.sequence(start_year, end_year).map(_year_total))


def export_total_rainfall_table(
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    aoi: ee.Geometry,
    temporal_scale: str,
    export_folder: str,
    reducer: ee.Reducer | None = None,
    scale: int = 5566,
    file_prefix: str | None = None,
) -> ee.FeatureCollection:
    """Compute total CHIRPS rainfall per period over an AOI and export it as a CSV table.

    Runs the full workflow end to end: build the daily CHIRPS collection for the window,
    sum daily precipitation into monthly or annual totals (one image per period), reduce
    each period's total over the AOI (default: spatial mean depth in mm), and start a
    Google Drive table-export task with the resulting long-format timeseries.

    Args:
        start_date: Window start as a string (e.g. '2020-01-01') or ee.Date.
        end_date: Window end (exclusive) as a string or ee.Date.
        aoi: Area of interest as ee.Geometry; bounds both the collection and the reduction.
        temporal_scale: Aggregation period; either 'monthly' or 'annual'.
        export_folder: Google Drive folder name to write the CSV into.
        reducer: Spatial ee.Reducer applied per period over the AOI; defaults to ee.Reducer.mean() (AOI-mean rainfall depth in mm).
        scale: Pixel scale in metres for the reduction (default 5566, CHIRPS native resolution).
        file_prefix: Filename prefix (without extension) and task description; defaults to 'chirps_{temporal_scale}_total_rainfall'.

    Returns:
        ee.FeatureCollection of the per-period rainfall timeseries (one feature per period, carrying date/year/month metadata and the per-band reduction result). A batch table-export task to Drive is started as a side effect.

    Raises:
        ValueError: If temporal_scale is not 'monthly' or 'annual', or if the date range falls outside the available CHIRPS data extent over the AOI.
    """
    valid_scales = {"monthly", "annual"}
    if temporal_scale not in valid_scales:
        raise ValueError(f"temporal_scale must be one of {sorted(valid_scales)}")

    if file_prefix is None:
        file_prefix = f"chirps_{temporal_scale}_total_rainfall"

    daily = get_chirps_collection(aoi, start_date, end_date)

    period_totals = build_period_composites(
        daily,
        bands=[CHIRPS_PRECIP_BAND],
        start_date=start_date,
        end_date=end_date,
        temporal_scale=temporal_scale,
        composite_stat="sum",
    )

    rainfall_ts = collection_to_region_timeseries(
        period_totals,
        region=aoi,
        bands=[CHIRPS_PRECIP_BAND],
        reducer=reducer,
        scale=scale,
        # 5566 m is CHIRPS' native resolution: no finer scale exists to suggest.
        min_pixels_per_region=None,
    )

    export_table_to_drive(
        collection=rainfall_ts,
        description=file_prefix,
        folder=export_folder,
        fileNamePrefix=file_prefix,
        fileFormat="CSV",
    )

    return rainfall_ts


def _year_chunks(start_date: str, end_date: str) -> list[tuple[int, str, str]]:
    """Split ``[start_date, end_date)`` (ISO dates) into calendar-year pieces."""
    start_year, end_year = int(start_date[:4]), int(end_date[:4])
    chunks = []
    for year in range(start_year, end_year + 1):
        lo = max(start_date, f"{year}-01-01")
        hi = min(end_date, f"{year + 1}-01-01")
        if lo < hi:
            chunks.append((year, lo, hi))
    return chunks


def _daily_rows(
    collection: ee.ImageCollection,
    regions: ee.FeatureCollection,
    id_property: str,
    version: str,
    scale: float,
) -> ee.FeatureCollection:
    """One row per image x region: date, region_id, ppt_mm, chirps_version."""

    def _per_image(image: ee.Image) -> ee.FeatureCollection:
        image = ee.Image(image)
        date = image.date().format("YYYY-MM-dd")
        reduced = image.reduceRegions(
            collection=regions, reducer=ee.Reducer.mean(), scale=scale
        )
        return reduced.map(
            lambda f: ee.Feature(
                None,
                {
                    "date": date,
                    "region_id": f.get(id_property),
                    "ppt_mm": f.get("mean"),
                    "chirps_version": version,
                },
            )
        )

    return ee.FeatureCollection(collection.map(_per_image)).flatten()


def export_daily_rainfall_table(
    regions: ee.FeatureCollection,
    start_date: str,
    end_date: str,
    *,
    id_property: str = "region_id",
    versions: Sequence[Literal["v2", "v3_rnl", "v3_sat"]] = ("v2", "v3_sat"),
    scale: float = CHIRPS_NATIVE_SCALE_M,
    dest: Literal["drive", "gcs"],
    bucket: str | None = None,
    prefix: str,
    chunk: Literal["year"] = "year",
    skip_existing: bool = True,
    start: bool = True,
) -> list[ee.batch.Task | None]:
    """Export daily CHIRPS region means as CSV tables, one per version and year.

    Each daily image of each version is reduced over every region
    (``reduceRegions`` mean at ``scale``) into rows with columns ``date``
    (YYYY-MM-DD), ``region_id`` (the ``id_property`` value), ``ppt_mm`` (mm/day; empty
    where a region has no data) and ``chirps_version``. Versions are never mixed: each
    table holds one version, named ``chirps_<version>_daily_<year>``. With
    ``dest="gcs"`` it is written to ``gs://<bucket>/<prefix>/<name>.csv``; with
    ``dest="drive"`` to the Drive folder ``prefix``. The date range is checked once
    per version against that collection's extent over the regions.

    Args:
        regions: Polygons to reduce over (e.g. a regional bbox plus each site).
        start_date: First date (inclusive), 'YYYY-MM-DD'.
        end_date: Last date (exclusive), 'YYYY-MM-DD'.
        id_property: Region property written as ``region_id``.
        versions: Keys of ``CHIRPS_DAILY_VERSIONS`` (default v2 and v3_sat side by side).
        scale: Reduction scale in metres (default CHIRPS native 5566 m).
        dest: 'drive' or 'gcs'.
        bucket: Destination bucket (required for 'gcs'; no ``gs://``).
        prefix: GCS object prefix, or the Drive folder name.
        chunk: Task chunking; only 'year' (one task per version x calendar year).
        skip_existing: GCS only: skip a table whose CSV already exists (needs ``eetools[gcs]``).
        start: Start the tasks (default True); False returns them unstarted.

    Returns:
        list[ee.batch.Task | None]. One entry per version x year, versions in the given order; None for a skipped GCS table.

    Raises:
        ValueError: For an unknown version, dest or chunk, a missing bucket with 'gcs', end_date <= start_date, or a date range outside a version's data.
    """
    if chunk != "year":
        raise ValueError(f"chunk must be 'year', got {chunk!r}.")
    if dest not in ("drive", "gcs"):
        raise ValueError(f"dest must be 'drive' or 'gcs', got {dest!r}.")
    if dest == "gcs" and not bucket:
        raise ValueError("bucket is required when dest='gcs'.")
    unknown = sorted(set(versions) - set(CHIRPS_DAILY_VERSIONS))
    if unknown or not versions:
        raise ValueError(
            f"versions must be a non-empty subset of {sorted(CHIRPS_DAILY_VERSIONS)}, "
            f"got {list(versions)}."
        )
    chunks = _year_chunks(start_date, end_date)
    if not chunks:
        raise ValueError(
            f"end_date ({end_date}) must be after start_date ({start_date})."
        )

    regions = ee.FeatureCollection(regions)
    bounds = regions.geometry().bounds()
    tasks: list[ee.batch.Task | None] = []
    for version in versions:
        collection_id = CHIRPS_DAILY_VERSIONS[version]
        validate_chirps_date_range(
            bounds, ee.Date(start_date), ee.Date(end_date), collection_id=collection_id
        )
        source = (
            ee.ImageCollection(collection_id)
            .filterBounds(bounds)
            .select(CHIRPS_PRECIP_BAND)
        )
        for year, lo, hi in chunks:
            rows = _daily_rows(
                source.filterDate(lo, hi), regions, id_property, version, scale
            )
            name = f"chirps_{version}_daily_{year}"
            selectors = list(CHIRPS_DAILY_TABLE_COLUMNS)
            if dest == "gcs":
                tasks.append(
                    export_table_to_gcs(
                        rows,
                        bucket=str(bucket),
                        file_name_prefix=f"{prefix}/{name}",
                        selectors=selectors,
                        skip_existing=skip_existing,
                        start=start,
                    )
                )
                continue
            task = ee.batch.Export.table.toDrive(
                collection=rows,
                description=sanitize_task_description(f"{prefix}_{name}"),
                folder=prefix,
                fileNamePrefix=sanitize_file_name_prefix(name),
                fileFormat="CSV",
                selectors=selectors,
            )
            if start:
                task.start()
            tasks.append(task)
    return tasks
