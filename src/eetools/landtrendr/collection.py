import ee

from eetools.constants import (
    LANDSAT_C2_ADD_OFFSET,
    LANDSAT_C2_SCALE_FACTOR,
    LANDTRENDR_BAND_MAP,
    LANDTRENDR_COMMON_BANDS,
    LANDTRENDR_DEFAULT_DIST_DIR,
    LANDTRENDR_DIST_DIR,
    LANDTRENDR_SENSORS,
    ROY_OLI_TO_ETM_INTERCEPTS,
    ROY_OLI_TO_ETM_SLOPES,
)
from eetools.sensors.indices import INDEX_REGISTRY, resolve_index_names
from eetools.sensors.landsat.masking import build_cloudfree_landsat_col

# Default loss-positive orientation factor: greenness/moisture indices fall with vegetation
# loss (a NEGATIVE delta), so multiply by -1 so that loss = POSITIVE delta as LandTrendr
# requires. Per-index overrides (e.g. +1 for bare-soil indices) live in LANDTRENDR_DIST_DIR.
# Retained as a module symbol because the output parsers default to it (see outputs.py).
DIST_DIR = LANDTRENDR_DEFAULT_DIST_DIR


def medoid_composite(collection: ee.ImageCollection, bands: list[str]) -> ee.Image:
    """Reduce a collection to a per-pixel medoid composite (best single observation).

    The medoid is the observation whose multiband value is closest (smallest summed
    squared distance) to the per-band median across the window — a best-pixel reducer
    that, unlike a per-band median, returns real co-registered observations.

    Args:
        collection: ee.ImageCollection to composite; must contain the named bands.
        bands: Band names used both for the distance computation and the output.

    Returns:
        ee.Image medoid composite containing only the named bands.
    """
    median = collection.select(bands).median()

    def _add_distance(image: ee.Image) -> ee.Image:
        diff = image.select(bands).subtract(median).pow(2).reduce(ee.Reducer.sum())
        # Negate so qualityMosaic (which keeps the MAXIMUM) selects the SMALLEST distance.
        return image.addBands(diff.multiply(-1).rename("medoid_distance"))

    return collection.map(_add_distance).qualityMosaic("medoid_distance").select(bands)


def _harmonize_oli_to_etm(image: ee.Image) -> ee.Image:
    """Apply the Roy et al. 2016 OLI->ETM+ transform to a common-band reflectance image.

    Brings Landsat 8/9 (OLI) reflectance onto the Landsat 5/7 (TM/ETM+) baseline so a
    multi-sensor annual series has no spectral step at the 2013 sensor boundary. Applies
    ``etm = (oli - intercept) / slope`` per band (matching the emaprlab LandTrendr.js).

    Args:
        image: ee.Image with bands renamed to LANDTRENDR_COMMON_BANDS, in C2 reflectance.

    Returns:
        ee.Image of harmonized reflectance with the same common band names.
    """
    slopes = ee.Image.constant(ROY_OLI_TO_ETM_SLOPES).rename(LANDTRENDR_COMMON_BANDS)
    intercepts = ee.Image.constant(ROY_OLI_TO_ETM_INTERCEPTS).rename(
        LANDTRENDR_COMMON_BANDS
    )
    return image.select(LANDTRENDR_COMMON_BANDS).subtract(intercepts).divide(slopes)


def _prep_scene(image: ee.Image, source_bands: list[str], is_oli: bool) -> ee.Image:
    """Scale a cloud-masked Landsat scene, rename to common bands, and harmonize OLI.

    Args:
        image: Cloud-masked raw Landsat C2 L2 ee.Image (output of build_cloudfree_landsat_col).
        source_bands: The sensor's reflective bands in BLUE,GREEN,RED,NIR,SWIR1,SWIR2 order.
        is_oli: True for Landsat 8/9 (apply Roy OLI->ETM+ harmonization); False for TM/ETM+.

    Returns:
        ee.Image of harmonized reflectance with LANDTRENDR_COMMON_BANDS, preserving system:time_start.
    """
    source = image
    scaled = (
        image.select(source_bands)
        .multiply(LANDSAT_C2_SCALE_FACTOR)
        .add(LANDSAT_C2_ADD_OFFSET)
        .rename(LANDTRENDR_COMMON_BANDS)
    )
    if is_oli:
        scaled = _harmonize_oli_to_etm(scaled)
    return ee.Image(scaled.copyProperties(source, ["system:time_start"]))


def _growing_season_range(
    year: int, start_day: str, end_day: str
) -> tuple[ee.Date, ee.Date]:
    """Build the [start, end) date window for a composite year.

    When ``start_day <= end_day`` the window is within the calendar year. When
    ``start_day > end_day`` the season crosses the new year (southern-hemisphere case):
    the window runs from ``year-start_day`` into ``(year+1)-end_day`` and is labelled by
    ``year``.

    Args:
        year: Calendar year that labels the composite.
        start_day: Window start as 'MM-DD'.
        end_day: Window end as 'MM-DD'.

    Returns:
        (start_date, end_date) as ee.Date; end_date is exclusive.
    """
    start = ee.Date(f"{year}-{start_day}")
    end_year = year if start_day <= end_day else year + 1
    end = ee.Date(f"{end_year}-{end_day}")
    return start, end


def _natural_index(composite: ee.Image, index: str) -> ee.Image:
    """Compute a natural-signed index band on a common-band composite (no loss orientation).

    Drives the generic INDEX_REGISTRY via LANDTRENDR_BAND_MAP, so any registered index whose
    bands the Landsat common-band composite provides can be used (for FTV bands).

    Args:
        composite: ee.Image with LANDTRENDR_COMMON_BANDS.
        index: Any INDEX_REGISTRY name computable from the common bands (see LANDTRENDR_BAND_MAP).

    Returns:
        ee.Image single band named ``index``, in its natural sign.

    Raises:
        ValueError: If ``index`` is unknown or requires band_map keys the common bands lack
            (e.g. a red-edge index such as NDRE/CIred_edge, which Landsat cannot provide).
    """
    # Validate the index is known and computable from the common bands (raises otherwise).
    resolve_index_names(LANDTRENDR_BAND_MAP, indices=[index])
    return INDEX_REGISTRY[index].compute(composite, LANDTRENDR_BAND_MAP)


def _segmentation_band(composite: ee.Image, index: str) -> ee.Image:
    """Compute the loss-positive segmentation band from a common-band composite.

    Orients the index so vegetation loss / disturbance is a POSITIVE delta, as LandTrendr
    requires: greenness/moisture indices (dist_dir -1) are negated, while bare-soil /
    burned-area indices (dist_dir +1, per LANDTRENDR_DIST_DIR) are used as-is.

    Args:
        composite: ee.Image with LANDTRENDR_COMMON_BANDS.
        index: Any INDEX_REGISTRY name computable from the common bands (see LANDTRENDR_BAND_MAP).

    Returns:
        ee.Image single band named ``index``, oriented so vegetation loss is positive.

    Raises:
        ValueError: If ``index`` is unknown or requires band_map keys the common bands lack.
    """
    band = _natural_index(composite, index)
    dist_dir = LANDTRENDR_DIST_DIR.get(index, LANDTRENDR_DEFAULT_DIST_DIR)
    # Orient loss-positive so LandTrendr sees vegetation loss / degradation as a rise.
    return band.multiply(dist_dir).rename(index)


def build_annual_composite(
    aoi: ee.Geometry,
    year: int,
    start_day: str = "01-01",
    end_day: str = "12-31",
    sensors: tuple[str, ...] = ("L5", "L7", "L8", "L9"),
) -> ee.Image:
    """Build one harmonized, cloud-masked medoid reflectance composite for a year.

    Merges the cloud-masked scenes of every requested sensor over the growing-season
    window, scales + renames them to common bands (Roy-harmonizing OLI), and medoid-
    composites to a single image. Sensors with no scenes in the window contribute nothing.

    Args:
        aoi: Area of interest as ee.Geometry.
        year: Calendar year that labels the composite.
        start_day: Growing-season window start as 'MM-DD' (default full year).
        end_day: Growing-season window end as 'MM-DD'; if before start_day the window crosses the new year.
        sensors: Landsat sensors to fuse (keys of LANDTRENDR_SENSORS).

    Returns:
        ee.Image medoid composite with LANDTRENDR_COMMON_BANDS and system:time_start set to the year.
    """
    start, end = _growing_season_range(year, start_day, end_day)

    prepped = None
    for sensor in sensors:
        collection_id, source_bands, is_oli = LANDTRENDR_SENSORS[sensor]
        scenes = build_cloudfree_landsat_col(aoi, start, end, collection_id).map(
            lambda img, sb=source_bands, oli=is_oli: _prep_scene(img, sb, oli)
        )
        prepped = scenes if prepped is None else prepped.merge(scenes)

    composite = medoid_composite(ee.ImageCollection(prepped), LANDTRENDR_COMMON_BANDS)
    return ee.Image(
        composite.set("system:time_start", ee.Date.fromYMD(year, 8, 1).millis())
    )


def build_landtrendr_collection(
    aoi: ee.Geometry,
    start_year: int,
    end_year: int,
    segmentation_index: str = "NBR",
    ftv_indices: list[str] | None = None,
    start_day: str = "01-01",
    end_day: str = "12-31",
    sensors: tuple[str, ...] = ("L5", "L7", "L8", "L9"),
) -> ee.ImageCollection:
    """Build the annual one-image-per-year input collection for LandTrendr.

    For each year in ``[start_year, end_year]`` it builds a multi-sensor harmonized medoid
    composite, then assembles the LandTrendr input image: band 1 is the loss-positive
    ``segmentation_index`` (the band LandTrendr segments) followed by any natural-signed
    fit-to-vertices (FTV) index bands. FTV bands let you recover a smoothed annual series
    of OTHER indices; the fitted segmentation index itself is recoverable from the run
    output without an FTV band (see outputs.get_fitted_stack).

    Args:
        aoi: Area of interest as ee.Geometry.
        start_year: First composite year (inclusive).
        end_year: Last composite year (inclusive).
        segmentation_index: Index LandTrendr segments on; any INDEX_REGISTRY index computable
            from the Landsat common bands (see LANDTRENDR_BAND_MAP), oriented loss-positive via
            LANDTRENDR_DIST_DIR (default 'NBR').
        ftv_indices: Additional indices to carry as natural-signed FTV bands; each any
            INDEX_REGISTRY index computable from the common bands, and different from
            segmentation_index. Default none.
        start_day: Growing-season window start as 'MM-DD' (default full year).
        end_day: Growing-season window end as 'MM-DD' (crosses the new year if before start_day).
        sensors: Landsat sensors to fuse (default all of L5/L7/L8/L9).

    Returns:
        ee.ImageCollection of one image per year, band order [segmentation_index, *ftv_indices], sorted by time.

    Raises:
        ValueError: If start_year > end_year, an FTV index duplicates the segmentation index,
            or any requested index is unknown or not computable from the Landsat common bands.
    """
    if start_year > end_year:
        raise ValueError(
            f"start_year ({start_year}) must not exceed end_year ({end_year})."
        )
    ftv = list(ftv_indices or [])
    if segmentation_index in ftv:
        raise ValueError(
            f"ftv_indices must not repeat the segmentation index {segmentation_index!r}; "
            "the fitted segmentation index comes from the LandTrendr output directly."
        )

    def _annual_lt_image(year: int) -> ee.Image:
        composite = build_annual_composite(aoi, year, start_day, end_day, sensors)
        seg = _segmentation_band(composite, segmentation_index)
        bands = [seg] + [_natural_index(composite, idx) for idx in ftv]
        stack = ee.Image.cat(bands) if len(bands) > 1 else seg
        return ee.Image(
            stack.set("system:time_start", composite.get("system:time_start"))
        )

    images = [_annual_lt_image(y) for y in range(start_year, end_year + 1)]
    return ee.ImageCollection(images).sort("system:time_start")
