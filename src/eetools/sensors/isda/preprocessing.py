from typing import Literal

import ee

from eetools.constants import (
    ISDA_CARBON_COLLECTION,
    ISDA_CARBON_TOPSOIL_BAND,
    ISDA_CLAY_IMAGE,
    ISDA_SAND_IMAGE,
    ISDA_TEXTURE_IMAGE,
)


def get_soil_texture(
    aoi: ee.Geometry | None = None,
    depth: Literal["0_20", "20_50"] = "0_20",
) -> ee.Image:
    """Load iSDAsoil texture class, sand and clay content for one depth.

    These are condition-independent edaphic predictors (unlike soil carbon). Sand and
    clay are stored as integer % with no back-transform (checked against the catalog
    2026-11-05); the texture class is the USDA texture class code.

    Args:
        aoi: Optional ee.Geometry to clip to.
        depth: Depth interval, '0_20' (default) or '20_50' cm.

    Returns:
        ee.Image with bands 'texture_class' (int), 'sand' and 'clay' (%), in the
        native 30 m projection.

    Raises:
        ValueError: If depth is not '0_20' or '20_50'.
    """
    if depth not in ("0_20", "20_50"):
        raise ValueError("depth must be '0_20' or '20_50'.")
    texture = ee.Image(ISDA_TEXTURE_IMAGE).select(
        [f"texture_{depth}"], ["texture_class"]
    )
    sand = ee.Image(ISDA_SAND_IMAGE).select([f"mean_{depth}"], ["sand"])
    clay = ee.Image(ISDA_CLAY_IMAGE).select([f"mean_{depth}"], ["clay"])
    image = texture.addBands(sand.toFloat()).addBands(clay.toFloat())
    if aoi is not None:
        image = image.clip(aoi)
    return image


def get_soil_carbon(aoi: ee.Geometry | None = None) -> ee.Image:
    """Load the iSDA Africa topsoil (0-20cm) total soil carbon layer, optionally
    clipped to an AOI.

    The source asset is a single static ee.Image (Africa-only, 30m nominal
    resolution); this selects the topsoil mean band and renames it for downstream
    use.

    Args:
        aoi: Optional area of interest as ee.Geometry; if provided, the image is clipped to it.

    Returns:
        ee.Image with a single 'soil_carbon' band, clipped to aoi when provided.
    """
    carbon = ee.Image(ISDA_CARBON_COLLECTION).select(
        [ISDA_CARBON_TOPSOIL_BAND], ["soil_carbon"]
    )
    if aoi is not None:
        carbon = carbon.clip(aoi)
    return carbon
