"""VECEA potential natural vegetation of eastern Africa (national vegetation class).

The layer is an uploaded asset (see ``constants.VECEA_ASSET_ID``). LICENCE: the Zenodo
deposit (10.5281/zenodo.11125645) says CC BY 4.0, but the bundled 2015 terms forbid
redistribution and commercial use without permission, so the default asset is private.
To use it under another project, rebuild it (``constants.VECEA_REPRODUCE_STEPS``):
download ``pnv_vecea_v2_0_<Country>_raster.zip`` from Zenodo, upload
``pnv_vecea_v2_0_<Country>.tif`` as an image asset with pyramiding policy MODE, then set
``EETOOLS_VECEA_ASSET`` or pass ``asset_id=`` / ``asset_overrides={"vecea": ...}``.
Cite van Breugel et al. (2015) and the Zenodo DOI.
"""

import ee

from eetools.constants import (
    VECEA_BAND,
    VECEA_CLASS_GROUPS,
    VECEA_CLASS_TABLE,
    VECEA_EXCLUDED_CLASSES,
)
from eetools.sensors.assets import resolve_asset_id


def get_vecea_class(
    aoi: ee.Geometry | None = None, asset_id: str | None = None
) -> ee.Image:
    """Load the VECEA national vegetation class raster (MAP_ID codes).

    Pixels outside the uploaded country coverage are masked; callers fall back to the
    PNV biome + RESOLVE ecoregion pair there.

    Args:
        aoi: Optional ee.Geometry to clip to.
        asset_id: Optional asset ID override (else EETOOLS_VECEA_ASSET, else the
            constant default).

    Returns:
        ee.Image with a single integer 'vecea_class' band (VECEA MAP_ID, 1-92).
    """
    image = ee.Image(resolve_asset_id("vecea", asset_id)).select(
        [VECEA_BAND], ["vecea_class"]
    )
    if aoi is not None:
        image = image.clip(aoi)
    return image


def vecea_class_code(map_id: int) -> str:
    """Return the VECEA short code (e.g. 'Fb') for a MAP_ID, or 'MAP_<id>' if unknown.

    Args:
        map_id: VECEA MAP_ID.

    Returns:
        The class code string.
    """
    entry = VECEA_CLASS_TABLE.get(int(map_id))
    return entry[0] if entry else f"MAP_{int(map_id)}"


def vecea_group(map_id: int) -> str | None:
    """Return the ecological group of a VECEA class (None for sand/water/unknown).

    Args:
        map_id: VECEA MAP_ID.

    Returns:
        Group name from ``VECEA_CLASS_GROUPS``, or None when the class is excluded
        (sand, water bodies) or not in the table.
    """
    if int(map_id) in VECEA_EXCLUDED_CLASSES:
        return None
    return VECEA_CLASS_GROUPS.get(int(map_id))


def vecea_group_members(groups: set[str]) -> list[int]:
    """Return every VECEA MAP_ID belonging to any of the given groups.

    Args:
        groups: Group names from ``VECEA_CLASS_GROUPS``.

    Returns:
        Sorted list of MAP_IDs in those groups.
    """
    return sorted(k for k, g in VECEA_CLASS_GROUPS.items() if g in groups)
