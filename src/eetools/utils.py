from typing import cast

import ee

# --------------------------------------------------------------------------- #
# Private helpers                                                              #
# --------------------------------------------------------------------------- #


def _clip_and_mask_image(image: ee.Image, geometry: ee.Geometry) -> ee.Image:
    aoi_mask = ee.Image.constant(1).clip(geometry).mask()
    return image.updateMask(aoi_mask).clip(geometry)


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #


def validate_collection_date_range(
    collection_id: str | list[str],
    aoi: ee.Geometry,
    start_date: ee.Date,
    end_date: ee.Date,
    sensor_label: str = "imagery",
) -> None:
    """Validate that a requested date window overlaps the available imagery over the
    AOI.

    Args:
        collection_id: EE collection asset ID string, or list of IDs that are merged before checking.
        aoi: Area of interest as ee.Geometry used to filter the collection.
        start_date: Requested start date as ee.Date; must be earlier than end_date.
        end_date: Requested end date as ee.Date; must be later than start_date.
        sensor_label: Human-readable sensor name used in error messages (default 'imagery').

    Returns:
        None. Raises ValueError if the date range is invalid or outside the available data extent.
    """
    if isinstance(collection_id, str):
        col = ee.ImageCollection(collection_id).filterBounds(aoi)
    else:
        cols = [ee.ImageCollection(c).filterBounds(aoi) for c in collection_id]
        col = cols[0]
        for c in cols[1:]:
            col = col.merge(c)

    if col.size().getInfo() == 0:
        raise ValueError(f"No {sensor_label} found for the provided AOI.")

    # getInfo() is typed Optional; in this branch the collection is non-empty so
    # the aggregates/millis are real numbers — cast for the numeric comparisons.
    min_ts = cast(int, col.aggregate_min("system:time_start").getInfo())
    max_ts = cast(int, col.aggregate_max("system:time_start").getInfo())
    min_str = ee.Date(min_ts).format("YYYY-MM-dd").getInfo()
    max_str = ee.Date(max_ts).format("YYYY-MM-dd").getInfo()

    s_ms = cast(int, start_date.millis().getInfo())
    e_ms = cast(int, end_date.millis().getInfo())
    s_str = start_date.format("YYYY-MM-dd").getInfo()
    e_str = end_date.format("YYYY-MM-dd").getInfo()

    if s_ms >= e_ms:
        raise ValueError(
            f"Invalid date range: start_date ({s_str}) must be earlier than end_date ({e_str})."
        )
    if s_ms >= max_ts:
        raise ValueError(
            f"Invalid start_date: {s_str} is on/after the most recent {sensor_label} date over this AOI ({max_str})."
        )
    if e_ms <= min_ts:
        raise ValueError(
            f"Invalid end_date: {e_str} is on/before the earliest {sensor_label} date over this AOI ({min_str})."
        )
    if s_ms < min_ts:
        raise ValueError(
            f"Invalid start_date: {s_str} is earlier than the first available {sensor_label} date over this AOI ({min_str})."
        )
    if e_ms > max_ts:
        raise ValueError(
            f"Invalid end_date: {e_str} is later than the most recent {sensor_label} date over this AOI ({max_str})."
        )


# --------------------------------------------------------------------------- #
# Clipping                                                                     #
# --------------------------------------------------------------------------- #


def clip_image_to_fc(
    fc: ee.FeatureCollection, image: ee.Image, name_field: str = "site_name"
) -> ee.ImageCollection:
    """Clip a single image to each feature in a FeatureCollection, producing one clipped
    image per feature.

    Args:
        fc: ee.FeatureCollection whose features define the clip geometries.
        image: ee.Image to clip.
        name_field: Feature property to copy as the 'site_name' property on each output image (default 'site_name').

    Returns:
        ee.ImageCollection with one clipped image per feature, each carrying the feature's site_name property.
    """

    def clip_feature(feature: ee.Feature) -> ee.Image:
        feature = ee.Feature(feature)
        return ee.Image(
            image.clip(feature.geometry()).set({"site_name": feature.get(name_field)})
        )

    return ee.ImageCollection(fc.map(clip_feature))


def clip_collection_to_geometry(
    collection: ee.ImageCollection,
    geometry: ee.Geometry,
    mask_outside: bool = True,
) -> ee.ImageCollection:
    """Clip every image in a collection to a geometry, optionally hard-masking pixels
    outside it.

    With ``mask_outside=True`` (default) each image is both clipped to the geometry's
    extent and masked by an AOI mask, so pixels outside an irregular boundary are
    dropped — useful before reductions/exports where bounding-box bleed would skew
    results. With ``mask_outside=False`` images are clipped only. Per-image properties
    (including ``system:time_start``) are preserved.

    Args:
        collection: ee.ImageCollection to clip.
        geometry: ee.Geometry defining the clip boundary (e.g. from vector_file_to_ee_geometry).
        mask_outside: If True (default), additionally mask pixels outside the geometry; if False, clip extent only.

    Returns:
        ee.ImageCollection with each image clipped (and optionally masked) to the geometry.
    """
    if mask_outside:

        def _clip(image: ee.Image) -> ee.Image:
            return _clip_and_mask_image(ee.Image(image), geometry)

    else:

        def _clip(image: ee.Image) -> ee.Image:
            return ee.Image(image).clip(geometry)

    return ee.ImageCollection(collection.map(_clip))


# --------------------------------------------------------------------------- #
# Collection and image inspection                                              #
# --------------------------------------------------------------------------- #


def get_collection_min_max(
    collection: ee.ImageCollection,
    band_name: str,
    scale: int,
    max_pixels: float = 1e13,
) -> tuple[float, float]:
    """Get the approximate global min and max of a band across all images in an
    ee.ImageCollection.

    Args:
        collection: ee.ImageCollection to inspect.
        band_name: Name of the band to reduce.
        scale: Pixel scale in metres for the reduceRegion calls.
        max_pixels: Maximum number of pixels to sample per image (default 1e13).

    Returns:
        tuple of (global_min, global_max) as Python floats.
    """
    min_key = f"{band_name}_min"
    max_key = f"{band_name}_max"

    # Reduce each image to its per-image min/max server-side, then aggregate the
    # extremes across the collection — one round-trip instead of one per image.
    def _img_min_max(image: ee.Image) -> ee.Feature:
        image = ee.Image(image)
        stats = image.select(band_name).reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=image.geometry(),
            scale=scale,
            maxPixels=int(max_pixels),
        )
        return ee.Feature(None, {"min": stats.get(min_key), "max": stats.get(max_key)})

    fc = ee.FeatureCollection(collection.map(_img_min_max))
    extremes = cast(
        dict,
        ee.Dictionary(
            {"min": fc.aggregate_min("min"), "max": fc.aggregate_max("max")}
        ).getInfo(),
    )
    return extremes["min"], extremes["max"]


def get_image_min_max(
    image: ee.Image, band_name: str, scale: int, max_pixels: float = 1e13
) -> tuple[float, float]:
    """Get the approximate min and max for a single band of an ee.Image.

    Args:
        image: ee.Image to inspect.
        band_name: Name of the band to reduce.
        scale: Pixel scale in metres for the reduceRegion call.
        max_pixels: Maximum number of pixels to sample (default 1e13).

    Returns:
        tuple of (min_value, max_value) as Python floats. Raises ValueError if no valid statistics are returned.
    """
    stats = (
        image.select(band_name)
        .reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=image.geometry(),
            scale=scale,
            maxPixels=max_pixels,
        )
        .getInfo()
    )

    min_key = f"{band_name}_min"
    max_key = f"{band_name}_max"

    if not stats or min_key not in stats or max_key not in stats:
        raise ValueError(f"No valid min/max returned for band '{band_name}'.")

    return stats[min_key], stats[max_key]


# --------------------------------------------------------------------------- #
# Image operations                                                             #
# --------------------------------------------------------------------------- #


def add_year_month(image: ee.Image) -> ee.Image:
    """Add 'year' and 'month' integer properties to an image derived from its
    system:time_start.

    Args:
        image: ee.Image with a valid system:time_start property.

    Returns:
        ee.Image with 'year' and 'month' properties set from the image's acquisition date.
    """
    date = ee.Date(image.get("system:time_start"))
    return ee.Image(image.set({"year": date.get("year"), "month": date.get("month")}))


def resample_pixel_resolution(image: ee.Image, output_resolution: int) -> ee.Image:
    """Resample an image to a target pixel resolution using mean aggregation for
    downscaling or bilinear interpolation for upscaling.

    Args:
        image: ee.Image to resample.
        output_resolution: Target pixel size in metres.

    Returns:
        ee.Image reprojected to the target resolution, with all original properties preserved.
    """
    projection = image.projection()
    nominal_scale = ee.Number(projection.nominalScale())
    output_scale = ee.Number(output_resolution)

    same_scale = nominal_scale.eq(output_scale)
    is_downscaling = output_scale.gt(nominal_scale)

    downscaled = (
        image.reduceResolution(
            reducer=ee.Reducer.mean(), bestEffort=True, maxPixels=4096
        )
        .reproject(crs=projection.crs(), scale=output_resolution)
        .copyProperties(image, image.propertyNames())
    )

    upscaled = (
        image.resample("bilinear")
        .reproject(crs=projection.crs(), scale=output_resolution)
        .copyProperties(image, image.propertyNames())
    )

    return ee.Image(
        ee.Algorithms.If(
            same_scale,
            image,
            ee.Algorithms.If(is_downscaling, downscaled, upscaled),
        )
    )


def generate_diff_image(
    image_initial: ee.Image,
    image_final: ee.Image,
    bands: list[str] | None = None,
    output_suffix: str | None = None,
) -> ee.Image:
    """Compute the per-pixel, per-band change between two images (final - initial).

    Useful for change products such as the difference in median NDVI between two
    years: pass the earlier composite as image_initial and the later one as
    image_final so positive values denote an increase.

    Args:
        image_initial: Baseline (earlier) ee.Image.
        image_final: Comparison (later) ee.Image; the difference is image_final - image_initial.
        bands: Optional list of band names to select from both images before differencing; if None, both images are differenced as-is and must share the same bands in the same order.
        output_suffix: Optional string appended to each output band name (e.g. '_diff') to mark the result as a difference; if None, band names are preserved.

    Returns:
        ee.Image of the band-wise difference, with bands renamed using output_suffix when provided.
    """
    if bands is not None:
        image_initial = image_initial.select(bands)
        image_final = image_final.select(bands)

    diff = image_final.subtract(image_initial)

    if output_suffix is not None:
        renamed = diff.bandNames().map(lambda name: ee.String(name).cat(output_suffix))
        diff = diff.rename(renamed)

    return diff


# --------------------------------------------------------------------------- #
# Collection operations                                                        #
# --------------------------------------------------------------------------- #


def join_collections(
    col_1: ee.ImageCollection,
    col_2: ee.ImageCollection,
    bands_1: list[str],
    bands_2: list[str],
    renamed_bands_1: list[str] | None = None,
    renamed_bands_2: list[str] | None = None,
    join_property: str = "year",
    copy_properties_from: str = "col_1",
) -> ee.ImageCollection:
    """Join two image collections on a shared property and stack selected bands into a
    single collection.

    Args:
        col_1: Primary ee.ImageCollection.
        col_2: Secondary ee.ImageCollection to join.
        bands_1: List of band names to select from col_1 images.
        bands_2: List of band names to select from col_2 images.
        renamed_bands_1: Optional output band names for col_1 bands; defaults to bands_1.
        renamed_bands_2: Optional output band names for col_2 bands; defaults to bands_2.
        join_property: Image property used as the join key (default 'year').
        copy_properties_from: Which collection's properties to copy onto merged images; 'col_1' or 'col_2' (default 'col_1').

    Returns:
        ee.ImageCollection of merged images, each containing bands from both collections.
    """
    if renamed_bands_1 is None:
        renamed_bands_1 = bands_1
    if renamed_bands_2 is None:
        renamed_bands_2 = bands_2

    join_filter = ee.Filter.equals(leftField=join_property, rightField=join_property)
    joined = ee.Join.inner().apply(
        primary=col_1, secondary=col_2, condition=join_filter
    )

    def _merge_pair(feature: ee.Feature) -> ee.Image:
        feature = ee.Feature(feature)
        img_1 = ee.Image(feature.get("primary")).select(bands_1, renamed_bands_1)
        img_2 = ee.Image(feature.get("secondary")).select(bands_2, renamed_bands_2)
        source_img = (
            ee.Image(feature.get("secondary"))
            if copy_properties_from == "col_2"
            else ee.Image(feature.get("primary"))
        )
        merged = img_1.addBands(img_2)
        return merged.copyProperties(source_img, source_img.propertyNames()).set(
            join_property, ee.Image(feature.get("primary")).get(join_property)
        )

    return ee.ImageCollection(joined.map(_merge_pair))


def temporal_reducer(
    collection: ee.ImageCollection,
    percentiles: list[int] | None = None,
) -> ee.Image:
    """Reduce an ImageCollection over time to mean, percentiles, and stdDev bands per
    pixel.

    Args:
        collection: ee.ImageCollection to reduce.
        percentiles: List of percentile values to compute (default [10, 90]).

    Returns:
        ee.Image with one output band per reducer output (e.g. band_mean, band_p10, band_p90, band_stdDev).
    """
    if percentiles is None:
        percentiles = [10, 90]
    reducer = (
        ee.Reducer.mean()
        .combine(reducer2=ee.Reducer.percentile(percentiles), sharedInputs=True)
        .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
    )
    return collection.reduce(reducer)
