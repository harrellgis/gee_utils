import ee

from eetools.constants import HUMAN_MODIFICATION_BAND, HUMAN_MODIFICATION_COLLECTION


def get_human_modification(aoi: ee.Geometry | None = None) -> ee.Image:
    """Load the TNC Global Human Modification v3 (2022, 90 m) combined-threats index.

    The source is an ImageCollection holding a single 2022 image; ``.first()`` keeps
    its native 90 m projection (needed for explicit ``reduceResolution``).

    Args:
        aoi: Optional ee.Geometry to clip to.

    Returns:
        ee.Image with a single 'ghm' band (0 = unmodified, 1 = fully modified).
    """
    ghm = (
        ee.ImageCollection(HUMAN_MODIFICATION_COLLECTION)
        .first()
        .select([HUMAN_MODIFICATION_BAND], ["ghm"])
    )
    if aoi is not None:
        ghm = ghm.clip(aoi)
    return ee.Image(ghm)
