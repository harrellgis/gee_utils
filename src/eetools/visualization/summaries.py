from typing import cast

import ee


def _validate_composite_stat(composite_stat: str) -> None:
    valid = {"mean", "median", "sum"}
    if composite_stat not in valid:
        raise ValueError(f"composite_stat must be one of {sorted(valid)}")


def _apply_stat(collection: ee.ImageCollection, composite_stat: str) -> ee.Image:
    if composite_stat == "mean":
        return collection.mean()
    if composite_stat == "median":
        return collection.median()
    if composite_stat == "sum":
        return collection.sum()
    raise ValueError(f"Unsupported composite_stat: {composite_stat}")


def _time_windows(
    start_date: ee.Date, end_date: ee.Date, temporal_scale: str
) -> ee.FeatureCollection:
    start_date = ee.Date(start_date)
    end_date = ee.Date(end_date)

    if temporal_scale == "annual":
        start_year = ee.Number.parse(start_date.format("YYYY"))
        end_year = ee.Number.parse(end_date.advance(-1, "day").format("YYYY"))
        offsets = ee.List.sequence(0, end_year.subtract(start_year))

        def make_window(offset):
            offset = ee.Number(offset)
            window_start = ee.Date.fromYMD(start_year.add(offset), 1, 1)
            return ee.Feature(
                None,
                {
                    "window_start": window_start.millis(),
                    "date": window_start.format("YYYY-MM-dd"),
                    "year": window_start.get("year"),
                    "month": None,
                    "day": None,
                },
            )

    elif temporal_scale == "monthly":
        # Count whole calendar months in [start, end) rather than relying on
        # ee.Date.difference(unit="month"), which uses an average month length
        # (~30.44 days) and truncates — dropping the final window whenever the
        # span includes 31-day months. advance(-1, "day") makes the end
        # exclusive, mirroring the annual branch.
        last_day = end_date.advance(-1, "day")
        start_months = (
            ee.Number(start_date.get("year")).multiply(12).add(start_date.get("month"))
        )
        end_months = (
            ee.Number(last_day.get("year")).multiply(12).add(last_day.get("month"))
        )
        n_months = end_months.subtract(start_months).add(1)
        offsets = ee.List.sequence(0, n_months.subtract(1))

        def make_window(offset):
            offset = ee.Number(offset)
            window_start = start_date.advance(offset, "month")
            return ee.Feature(
                None,
                {
                    "window_start": window_start.millis(),
                    "date": window_start.format("YYYY-MM-dd"),
                    "year": window_start.get("year"),
                    "month": window_start.get("month"),
                    "day": None,
                },
            )

    else:
        raise ValueError("temporal_scale must be either 'annual' or 'monthly'")

    return ee.FeatureCollection(offsets.map(make_window))


def summarize_collection_histograms(
    image_collection: ee.ImageCollection,
    band_name: str,
    min_value: float,
    max_value: float,
    steps: int,
    scale: float,
    max_pixels: float = 1e13,
) -> list[dict]:
    """Compute a fixed-bin histogram of pixel values for each image in an
    ee.ImageCollection.

    Args:
        image_collection: ee.ImageCollection to summarize; each image is expected to have a 'site_name' property.
        band_name: Name of the band to histogram.
        min_value: Lower bound of the histogram range.
        max_value: Upper bound of the histogram range.
        steps: Number of histogram bins.
        scale: Pixel scale in metres for the reduceRegion call.
        max_pixels: Maximum number of pixels to sample per image (default 1e13).

    Returns:
        list of dicts, one per image, each with keys 'site_name' and 'histogram' (list of [bin_center, count] pairs).
    """

    # Build one feature per image carrying its site_name and histogram array
    # server-side, then pull the whole collection back in a single getInfo().
    def _histogram_feature(image: ee.Image) -> ee.Feature:
        image = ee.Image(image)
        histogram = image.select(band_name).reduceRegion(
            reducer=ee.Reducer.fixedHistogram(min_value, max_value, steps),
            geometry=image.geometry(),
            scale=scale,
            maxPixels=int(max_pixels),
        )
        return ee.Feature(
            None,
            {
                "site_name": image.get("site_name"),
                "histogram": histogram.get(band_name),
            },
        )

    fc = ee.FeatureCollection(image_collection.map(_histogram_feature))
    info = cast(dict, fc.getInfo())
    return [
        {
            "site_name": feature["properties"].get("site_name"),
            "histogram": feature["properties"].get("histogram"),
        }
        for feature in info["features"]
    ]


def image_collection_to_sample_fc(
    collection: ee.ImageCollection,
    regions_fc: ee.FeatureCollection,
    band_names: list[str],
    scale: int,
    region_properties: list[str] | None = None,
    image_properties: list[str] | None = None,
    tile_scale: int = 4,
    geometries: bool = False,
) -> ee.FeatureCollection:
    """Sample every image in an ImageCollection over polygon regions and return a flat
    FeatureCollection of pixel samples.

    Args:
        collection: ee.ImageCollection to sample.
        regions_fc: ee.FeatureCollection of polygon regions to sample over.
        band_names: List of band names to include in the samples.
        scale: Pixel scale in metres for the sampleRegions call.
        region_properties: List of region feature properties to copy onto each sample (default ['site_name']).
        image_properties: List of image properties to copy onto each sample (default ['year', 'product', 'param_set']).
        tile_scale: EE tileScale parameter to avoid memory limits (default 4).
        geometries: If True, include pixel geometries in the output features (default False).

    Returns:
        ee.FeatureCollection of pixel-level samples, one feature per pixel per image per region, with both region and image properties attached.
    """
    if region_properties is None:
        region_properties = ["site_name"]
    if image_properties is None:
        image_properties = ["year", "product", "param_set"]

    def _sample_image(image):
        image = ee.Image(image)
        props = image.toDictionary(image_properties)
        samples = image.select(band_names).sampleRegions(
            collection=regions_fc,
            properties=region_properties,
            scale=scale,
            geometries=geometries,
            tileScale=tile_scale,
        )
        return samples.map(lambda f: ee.Feature(f).setMulti(props))

    return ee.FeatureCollection(collection.map(_sample_image).flatten())


def image_collection_to_region_stats_fc(
    collection: ee.ImageCollection,
    regions_fc: ee.FeatureCollection,
    band_names: list[str],
    scale: int,
    reducers: ee.Reducer | None = None,
    image_properties: list[str] | None = None,
    tile_scale: int = 4,
) -> ee.FeatureCollection:
    """Reduce each image in a collection over polygon regions and return a flat
    FeatureCollection of per-region statistics.

    Args:
        collection: ee.ImageCollection to reduce.
        regions_fc: ee.FeatureCollection of polygon regions to reduce over.
        band_names: List of band names to include in the reduction.
        scale: Pixel scale in metres for the reduceRegions call.
        reducers: ee.Reducer to apply; defaults to mean combined with stdDev and minMax.
        image_properties: List of image properties to copy onto each output feature (default ['year', 'product', 'param_set']).
        tile_scale: EE tileScale parameter to avoid memory limits (default 4).

    Returns:
        ee.FeatureCollection of region-level statistics, one feature per region per image, with image properties attached.
    """
    if reducers is None:
        reducers = (
            ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
            .combine(ee.Reducer.minMax(), sharedInputs=True)
        )
    if image_properties is None:
        image_properties = ["year", "product", "param_set"]

    def _reduce_image(image):
        image = ee.Image(image)
        props = image.toDictionary(image_properties)
        fc = image.select(band_names).reduceRegions(
            collection=regions_fc,
            reducer=reducers,
            scale=scale,
            tileScale=tile_scale,
        )
        return fc.map(lambda f: ee.Feature(f).setMulti(props))

    return ee.FeatureCollection(collection.map(_reduce_image).flatten())


def build_period_composites(
    collection: ee.ImageCollection,
    bands: list[str],
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    temporal_scale: str = "annual",
    composite_stat: str = "median",
) -> ee.ImageCollection:
    """Build annual or monthly composites from an ImageCollection, producing one
    composite per period with data.

    Args:
        collection: ee.ImageCollection to composite.
        bands: List of band names to select before compositing.
        start_date: Compositing start date as a string or ee.Date.
        end_date: Compositing end date as a string or ee.Date.
        temporal_scale: Temporal aggregation window; either 'annual' or 'monthly' (default 'annual').
        composite_stat: Statistic used to combine images within each window; one of 'mean', 'median', or 'sum' (default 'median').

    Returns:
        ee.ImageCollection of composite images, one per period, each with date, year, month, day, image_count, temporal_scale, and composite_stat properties.
    """
    _validate_composite_stat(composite_stat)

    collection = ee.ImageCollection(collection).select(bands)
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    windows = _time_windows(start, end, temporal_scale)

    def make_composite(feature):
        feature = ee.Feature(feature)
        window_start = ee.Date(feature.get("window_start"))
        window_end = (
            window_start.advance(1, "year")
            if temporal_scale == "annual"
            else window_start.advance(1, "month")
        )
        subset = collection.filterDate(window_start, window_end)
        image_count = subset.size()
        composite = _apply_stat(subset, composite_stat)

        return ee.Image(
            ee.Algorithms.If(
                image_count.gt(0),
                composite.set(
                    {
                        "system:time_start": window_start.millis(),
                        "date": feature.get("date"),
                        "year": feature.get("year"),
                        "month": feature.get("month"),
                        "day": feature.get("day"),
                        "image_count": image_count,
                        "temporal_scale": temporal_scale,
                        "composite_stat": composite_stat,
                    }
                ),
                None,
            )
        )

    images_list = windows.toList(windows.size()).map(make_composite)
    return ee.ImageCollection.fromImages(images_list)


def build_seasonal_composites(
    collection: ee.ImageCollection,
    bands: list[str],
    start_year: int,
    end_year: int,
    season_months: tuple[int, int],
    season_name: str,
    composite_stat: str = "median",
) -> ee.ImageCollection:
    """Build per-year seasonal composites by filtering each year's collection to a
    fixed window of consecutive months.

    For each year in [start_year, end_year] the collection is filtered to the
    months [season_months[0], season_months[1]] (inclusive) and reduced with
    composite_stat. Years that contain no images in the season window are
    excluded from the output (the ImageCollection contains no null entries).

    Args:
        collection: ee.ImageCollection to composite.
        bands: List of band names to select before compositing.
        start_year: First year to process, inclusive.
        end_year: Last year to process, inclusive.
        season_months: (start_month, end_month) as 1-based integers (e.g. (3, 5)
            for March through May, inclusive).  Both months must fall within the
            same calendar year; cross-year seasons (e.g. Nov–Jan) are not
            supported.
        season_name: Label stored as the 'season' property on each output image
            (e.g. 'wet', 'dry').
        composite_stat: Statistic used to combine images; one of 'mean',
            'median', or 'sum' (default 'median').

    Returns:
        ee.ImageCollection of per-year composites, one per year with imagery in
        the season window, each carrying system:time_start, year, season,
        season_months, image_count, and composite_stat properties.

    Raises:
        ValueError: If composite_stat is not 'mean', 'median', or 'sum', or if
            season_months is not a valid (start, end) pair with 1 ≤ start ≤ end ≤ 12.
    """
    _validate_composite_stat(composite_stat)
    start_month, end_month = season_months
    if not (1 <= start_month <= end_month <= 12):
        raise ValueError(
            f"season_months must satisfy 1 <= start <= end <= 12; got {season_months}"
        )

    season_label = f"{start_month}-{end_month}"
    col = ee.ImageCollection(collection).select(bands)
    years = ee.List.sequence(start_year, end_year)

    def _make_composite(year: ee.Number) -> ee.Image:
        year = ee.Number(year)
        window_start = ee.Date.fromYMD(year, start_month, 1)
        window_end = ee.Date.fromYMD(year, end_month, 1).advance(1, "month")
        subset = col.filterDate(window_start, window_end)
        image_count = subset.size()
        composite = _apply_stat(subset, composite_stat)
        return ee.Image(
            ee.Algorithms.If(
                image_count.gt(0),
                composite.set(
                    {
                        "system:time_start": window_start.millis(),
                        "year": year,
                        "season": season_name,
                        "season_months": season_label,
                        "image_count": image_count,
                        "composite_stat": composite_stat,
                    }
                ),
                None,
            )
        )

    return ee.ImageCollection.fromImages(years.map(_make_composite))


def reduce_image_over_region(
    image: ee.Image,
    region: ee.Geometry,
    bands: list[str],
    reducer: ee.Reducer | None = None,
    scale: int = 5566,
    tile_scale: int = 4,
) -> ee.Feature:
    """Reduce a single image over a region geometry and return a Feature with band
    statistics and image properties.

    Args:
        image: ee.Image to reduce; should carry date, year, month, day, image_count, temporal_scale, and composite_stat properties.
        region: Region geometry as ee.Geometry.
        bands: List of band names to include in the reduction.
        reducer: ee.Reducer to apply; defaults to ee.Reducer.mean().
        scale: Pixel scale in metres for the reduceRegion call (default 5566).
        tile_scale: EE tileScale parameter to avoid memory limits (default 4).

    Returns:
        ee.Feature with properties combining the image's temporal metadata and the per-band reduction results.
    """
    image = ee.Image(image).select(bands)
    reducer = reducer or ee.Reducer.mean()

    stats = image.reduceRegion(
        reducer=reducer,
        geometry=region,
        scale=scale,
        maxPixels=int(1e13),
        tileScale=tile_scale,
    )

    props = ee.Dictionary(
        {
            "date": image.get("date"),
            "year": image.get("year"),
            "month": image.get("month"),
            "day": image.get("day"),
            "image_count": image.get("image_count"),
            "temporal_scale": image.get("temporal_scale"),
            "composite_stat": image.get("composite_stat"),
            "system_time_start": image.get("system:time_start"),
        }
    ).combine(stats, overwrite=True)

    return ee.Feature(None, props)


def collection_to_region_timeseries(
    collection: ee.ImageCollection,
    region: ee.Geometry,
    bands: list[str],
    reducer: ee.Reducer | None = None,
    scale: int = 5566,
    tile_scale: int = 4,
) -> ee.FeatureCollection:
    """Reduce every image in a collection over a region and return a long-format
    FeatureCollection timeseries.

    Args:
        collection: ee.ImageCollection to reduce; images should carry temporal metadata properties.
        region: Region geometry as ee.Geometry.
        bands: List of band names to include in each reduction.
        reducer: ee.Reducer to apply per image; defaults to ee.Reducer.mean().
        scale: Pixel scale in metres for each reduceRegion call (default 5566).
        tile_scale: EE tileScale parameter to avoid memory limits (default 4).

    Returns:
        ee.FeatureCollection with one feature per image, each containing temporal metadata and per-band reduction results.
    """
    collection = ee.ImageCollection(collection)
    reducer = reducer or ee.Reducer.mean()

    fc = collection.map(
        lambda image: reduce_image_over_region(
            image=ee.Image(image),
            region=region,
            bands=bands,
            reducer=reducer,
            scale=scale,
            tile_scale=tile_scale,
        )
    )

    return ee.FeatureCollection(fc)
