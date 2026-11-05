"""SoilGrids 2.0 WRB most-probable reference soil group (substrate gate).

Uploaded asset (``constants.SOILGRIDS_WRB_ASSET_ID``), clipped from ISRIC's
``wrb/MostProbable.vrt`` without resampling. CC BY 4.0: cite Poggio et al. (2021), ISRIC.
"""

import ee

from eetools.constants import SOILGRIDS_WRB_BAND, WRB_CLASS_TABLE, WRB_RELATED_GROUPS
from eetools.sensors.assets import resolve_asset_id


def get_soilgrids_wrb(
    aoi: ee.Geometry | None = None, asset_id: str | None = None
) -> ee.Image:
    """Load the SoilGrids WRB reference soil group raster (codes 0-29, ~230 m).

    Args:
        aoi: Optional ee.Geometry to clip to.
        asset_id: Optional asset ID override (else EETOOLS_SOILGRIDS_WRB_ASSET, else
            the constant default).

    Returns:
        ee.Image with a single integer 'wrb_group' band (see WRB_CLASS_TABLE); masked
        over the ocean and outside the uploaded extent.
    """
    image = ee.Image(resolve_asset_id("soilgrids_wrb", asset_id)).select(
        [SOILGRIDS_WRB_BAND], ["wrb_group"]
    )
    if aoi is not None:
        image = image.clip(aoi)
    return image


def wrb_name(code: int) -> str:
    """Return the WRB reference soil group name for a code ('WRB_<code>' if unknown).

    Args:
        code: SoilGrids WRB code (0-29).

    Returns:
        Soil group name, e.g. 'Ferralsols'.
    """
    return WRB_CLASS_TABLE.get(int(code), f"WRB_{int(code)}")


def wrb_related_codes(code: int) -> list[int]:
    """Return the codes in the same ``WRB_RELATED_GROUPS`` set as ``code``.

    Args:
        code: SoilGrids WRB code.

    Returns:
        Sorted list of related codes (always includes ``code`` itself).
    """
    for members in WRB_RELATED_GROUPS.values():
        if int(code) in members:
            return sorted(members)
    return [int(code)]
