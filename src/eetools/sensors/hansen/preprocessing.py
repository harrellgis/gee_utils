import ee

from eetools.constants import (
    HANSEN_GAIN_BAND,
    HANSEN_GFC_COLLECTION,
    HANSEN_LOSS_BAND,
    HANSEN_LOSSYEAR_BAND,
    HANSEN_LOSSYEAR_EPOCH,
    HANSEN_LOSSYEAR_MAX,
    HANSEN_TREE_COVER_THRESHOLD,
    HANSEN_TREECOVER_BAND,
)


def get_forest_change_image(aoi: ee.Geometry | None = None) -> ee.Image:
    """Load the Hansen Global Forest Change image, optionally clipped to an AOI.

    The Hansen GFC product is a single static multiband ``ee.Image`` (Landsat-derived
    forest extent and change at 30.92 m). This is the base loader the other functions
    build on, so the asset is referenced in exactly one place.

    Args:
        aoi: Optional area of interest as ee.Geometry; if provided, the image is clipped to it.

    Returns:
        ee.Image with all native Hansen GFC bands (treecover2000, loss, gain, lossyear, first_*, last_*, datamask), clipped to aoi when provided.
    """
    gfc = ee.Image(HANSEN_GFC_COLLECTION)
    if aoi is not None:
        gfc = gfc.clip(aoi)
    return gfc


def get_forest_2000(aoi: ee.Geometry) -> ee.Image:
    """Return tree canopy cover (%) for the year 2000, clipped to the AOI.

    Args:
        aoi: Area of interest as ee.Geometry; the result is clipped to it.

    Returns:
        ee.Image with a single 'treecover2000' band (canopy closure %, 0-100) clipped to the AOI.
    """
    forest_2000 = get_forest_change_image().select(HANSEN_TREECOVER_BAND)
    return forest_2000.clip(aoi)


def get_tree_cover_mask(
    aoi: ee.Geometry,
    threshold: int = HANSEN_TREE_COVER_THRESHOLD,
) -> ee.Image:
    """Return a binary forest/non-forest mask where year-2000 canopy cover meets a
    threshold.

    Args:
        aoi: Area of interest as ee.Geometry; the result is clipped to it.
        threshold: Minimum year-2000 canopy cover % counted as forest (default HANSEN_TREE_COVER_THRESHOLD).

    Returns:
        ee.Image with a single binary band (1 where treecover2000 >= threshold, else 0) clipped to the AOI.
    """
    mask = get_forest_change_image().select(HANSEN_TREECOVER_BAND).gte(threshold)
    return mask.clip(aoi)


def get_forest_loss_image(
    aoi: ee.Geometry,
    tree_cover_threshold: int = HANSEN_TREE_COVER_THRESHOLD,
) -> ee.Image:
    """Return a binary forest-loss image (forest in 2000 AND a loss event), clipped to
    the AOI.

    Args:
        aoi: Area of interest as ee.Geometry; the result is clipped to it.
        tree_cover_threshold: Minimum year-2000 canopy cover % counted as forest (default HANSEN_TREE_COVER_THRESHOLD).

    Returns:
        ee.Image with a single binary 'forest_loss' band (1 where year-2000 forest was lost during the study period) clipped to the AOI.
    """
    gfc = get_forest_change_image()
    forest2000 = gfc.select(HANSEN_TREECOVER_BAND).gte(tree_cover_threshold)
    loss = gfc.select(HANSEN_LOSS_BAND)

    forest_loss = forest2000.And(loss).rename("forest_loss")
    return forest_loss.clip(aoi)


def get_forest_loss_year_image(
    aoi: ee.Geometry,
    tree_cover_threshold: int = HANSEN_TREE_COVER_THRESHOLD,
) -> ee.Image:
    """Return the year-of-loss image masked to year-2000 forest, clipped to the AOI.

    The 'lossyear' band encodes the loss year as 1-25 (2001-2025); add
    ``HANSEN_LOSSYEAR_EPOCH`` to recover the absolute year.

    Args:
        aoi: Area of interest as ee.Geometry; the result is clipped to it.
        tree_cover_threshold: Minimum year-2000 canopy cover % counted as forest (default HANSEN_TREE_COVER_THRESHOLD).

    Returns:
        ee.Image with a single 'lossyear' band (1-25, masked to year-2000 forest pixels) clipped to the AOI.
    """
    gfc = get_forest_change_image()
    forest2000 = gfc.select(HANSEN_TREECOVER_BAND).gte(tree_cover_threshold)
    lossyear = (
        gfc.select(HANSEN_LOSSYEAR_BAND).updateMask(forest2000).rename("lossyear")
    )
    return lossyear.clip(aoi)


def get_forest_gain_image(aoi: ee.Geometry) -> ee.Image:
    """Return the binary forest-gain band, clipped to the AOI.

    Gain is a non-forest -> forest change detected over 2000-2012 only; per the dataset
    it was not updated in later versions, so it does not cover the full study period.

    Args:
        aoi: Area of interest as ee.Geometry; the result is clipped to it.

    Returns:
        ee.Image with a single 'gain' band (1 where forest gain occurred 2000-2012) clipped to the AOI.
    """
    gain = get_forest_change_image().select(HANSEN_GAIN_BAND)
    return gain.clip(aoi)


def get_forest_loss_in_period(
    aoi: ee.Geometry,
    start_year: int,
    end_year: int,
    tree_cover_threshold: int = HANSEN_TREE_COVER_THRESHOLD,
) -> ee.Image:
    """Return binary forest loss restricted to a year range, masked to year-2000 forest.

    The 'lossyear' band encodes 1-25 for 2001-2025; this selects loss whose absolute
    year falls within ``[start_year, end_year]`` (inclusive) and intersects it with the
    year-2000 forest mask.

    Args:
        aoi: Area of interest as ee.Geometry; the result is clipped to it.
        start_year: First loss year to include (inclusive); must be >= HANSEN_LOSSYEAR_EPOCH + 1.
        end_year: Last loss year to include (inclusive); must be <= HANSEN_LOSSYEAR_EPOCH + HANSEN_LOSSYEAR_MAX.
        tree_cover_threshold: Minimum year-2000 canopy cover % counted as forest (default HANSEN_TREE_COVER_THRESHOLD).

    Returns:
        ee.Image with a single binary 'forest_loss' band (1 where year-2000 forest was lost within the period) clipped to the AOI. Raises ValueError if the year range is invalid or outside the dataset's coverage.
    """
    min_year = HANSEN_LOSSYEAR_EPOCH + 1
    max_year = HANSEN_LOSSYEAR_EPOCH + HANSEN_LOSSYEAR_MAX
    if start_year > end_year:
        raise ValueError(
            f"start_year ({start_year}) must not exceed end_year ({end_year})."
        )
    if start_year < min_year or end_year > max_year:
        raise ValueError(
            f"Year range [{start_year}, {end_year}] is outside the Hansen GFC coverage "
            f"[{min_year}, {max_year}]."
        )

    gfc = get_forest_change_image()
    forest2000 = gfc.select(HANSEN_TREECOVER_BAND).gte(tree_cover_threshold)
    lossyear = gfc.select(HANSEN_LOSSYEAR_BAND)

    start_code = start_year - HANSEN_LOSSYEAR_EPOCH
    end_code = end_year - HANSEN_LOSSYEAR_EPOCH
    in_period = lossyear.gte(start_code).And(lossyear.lte(end_code))

    forest_loss = forest2000.And(in_period).rename("forest_loss")
    return forest_loss.clip(aoi)
