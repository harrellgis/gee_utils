from typing import cast

import ee

# --------------------------------------------------------------------------- #
# Zonal reduction — single region                                             #
# --------------------------------------------------------------------------- #


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
        image: ee.Image to reduce; all scalar (non-system) properties are carried through to the output feature automatically.
        region: Region geometry as ee.Geometry.
        bands: List of band names to include in the reduction.
        reducer: ee.Reducer to apply; defaults to ee.Reducer.mean().
        scale: Pixel scale in metres for the reduceRegion call (default 5566).
        tile_scale: EE tileScale parameter to avoid memory limits (default 4).

    Returns:
        ee.Feature with properties combining all scalar image properties and the per-band reduction results (band stats take precedence on name collision).
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

    props = image.toDictionary().combine(stats, overwrite=True)

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


# --------------------------------------------------------------------------- #
# Zonal reduction — multiple regions                                          #
# --------------------------------------------------------------------------- #


def image_collection_to_region_stats_fc(
    collection: ee.ImageCollection,
    regions_fc: ee.FeatureCollection,
    bands: list[str],
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
        bands: List of band names to include in the reduction.
        scale: Pixel scale in metres for the reduceRegions call.
        reducers: ee.Reducer to apply; defaults to mean combined with stdDev and minMax.
        image_properties: List of image properties to copy onto each output feature; when None (default), all scalar image properties are copied via image.toDictionary().
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

    def _reduce_image(image: ee.Image) -> ee.FeatureCollection:
        image = ee.Image(image)
        props = (
            image.toDictionary()
            if image_properties is None
            else image.toDictionary(image_properties)
        )
        fc = image.select(bands).reduceRegions(
            collection=regions_fc,
            reducer=reducers,
            scale=scale,
            tileScale=tile_scale,
        )
        return fc.map(lambda f: ee.Feature(f).setMulti(props))

    return ee.FeatureCollection(collection.map(_reduce_image).flatten())


def image_collection_to_sample_fc(
    collection: ee.ImageCollection,
    regions_fc: ee.FeatureCollection,
    bands: list[str],
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
        bands: List of band names to include in the samples.
        scale: Pixel scale in metres for the sampleRegions call.
        region_properties: List of region feature properties to copy onto each sample (default ['site_name']).
        image_properties: List of image properties to copy onto each sample; when None (default), all scalar image properties are copied via image.toDictionary().
        tile_scale: EE tileScale parameter to avoid memory limits (default 4).
        geometries: If True, include pixel geometries in the output features (default False).

    Returns:
        ee.FeatureCollection of pixel-level samples, one feature per pixel per image per region, with both region and image properties attached.
    """
    if region_properties is None:
        region_properties = ["site_name"]

    def _sample_image(image: ee.Image) -> ee.FeatureCollection:
        image = ee.Image(image)
        props = (
            image.toDictionary()
            if image_properties is None
            else image.toDictionary(image_properties)
        )
        samples = image.select(bands).sampleRegions(
            collection=regions_fc,
            properties=region_properties,
            scale=scale,
            geometries=geometries,
            tileScale=tile_scale,
        )
        return samples.map(lambda f: ee.Feature(f).setMulti(props))

    return ee.FeatureCollection(collection.map(_sample_image).flatten())


# --------------------------------------------------------------------------- #
# Categorical class-area summaries                                             #
# --------------------------------------------------------------------------- #


def summarize_class_areas(
    regions_fc: ee.FeatureCollection,
    classified_image: ee.Image,
    class_map: dict[str, int],
    scale: int,
    crs: str = "EPSG:4326",
    tile_scale: int = 4,
    unmask_value: int = -9999,
) -> ee.FeatureCollection:
    """Sum per-class pixel area (m^2) by region for a single-band categorical image.

    Builds one area band per class (pixel area where the image equals that class's
    code, zero elsewhere) and sums each over every region, extending each input
    feature with one ``<class_name>_area_m2`` property per class map entry. Chain
    onto the output of :func:`image_collection_to_region_stats_fc` or a prior call
    to this function (or ``reduceRegions``) to add class areas alongside other
    per-region statistics without a separate join.

    Args:
        regions_fc: ee.FeatureCollection of polygon regions to sum area over.
        classified_image: Single-band ee.Image of categorical class codes (e.g. ESA WorldCover 'land_cover'); only its first band is used.
        class_map: Mapping of clean class name -> integer class code (e.g. ``constants.ESA_CLASS_MAP``).
        scale: Pixel scale in metres for the reduceRegions call.
        crs: Coordinate reference system for the reduction (default EPSG:4326).
        tile_scale: EE tileScale parameter to avoid memory limits (default 4).
        unmask_value: Value substituted for masked pixels before comparison, chosen to not collide with any real class code (default -9999).

    Returns:
        ee.FeatureCollection with each input feature extended by one
        ``<class_name>_area_m2`` property (summed pixel area in square metres) per
        entry in class_map, alongside all of the region's original properties.
    """
    band = ee.Image(classified_image).select(0).unmask(unmask_value)

    class_area_bands = [
        ee.Image.pixelArea()
        .multiply(band.eq(class_value))
        .rename(f"{class_name}_area_m2")
        for class_name, class_value in class_map.items()
    ]
    area_stack = ee.Image.cat(class_area_bands)

    return area_stack.reduceRegions(
        collection=regions_fc,
        reducer=ee.Reducer.sum(),
        scale=scale,
        crs=crs,
        tileScale=tile_scale,
    )


# --------------------------------------------------------------------------- #
# Histograms                                                                   #
# --------------------------------------------------------------------------- #


def summarize_collection_histograms(
    collection: ee.ImageCollection,
    band_name: str,
    min_value: float,
    max_value: float,
    steps: int,
    scale: float,
    max_pixels: float = 1e13,
    name_field: str = "site_name",
) -> list[dict]:
    """Compute a fixed-bin histogram of pixel values for each image in an
    ee.ImageCollection.

    Args:
        collection: ee.ImageCollection to summarize.
        band_name: Name of the band to histogram.
        min_value: Lower bound of the histogram range.
        max_value: Upper bound of the histogram range.
        steps: Number of histogram bins.
        scale: Pixel scale in metres for the reduceRegion call.
        max_pixels: Maximum number of pixels to sample per image (default 1e13).
        name_field: Image property used as the label key in each output dict (default 'site_name'); mirrors the name_field convention in clip_image_to_fc.

    Returns:
        list of dicts, one per image, each with keys name_field and 'histogram' (list of [bin_center, count] pairs).
    """

    # Build one feature per image carrying its label and histogram array
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
                name_field: image.get(name_field),
                "histogram": histogram.get(band_name),
            },
        )

    fc = ee.FeatureCollection(collection.map(_histogram_feature))
    info = cast(dict, fc.getInfo())
    return [
        {
            name_field: feature["properties"].get(name_field),
            "histogram": feature["properties"].get("histogram"),
        }
        for feature in info["features"]
    ]
