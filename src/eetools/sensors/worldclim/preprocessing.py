from collections.abc import Sequence

import ee

from eetools.constants import (
    WORLDCLIM_BIO_IMAGE,
    WORLDCLIM_DEFAULT_BANDS,
    WORLDCLIM_SCALES,
)


def get_worldclim_bio(
    bands: Sequence[str] = tuple(WORLDCLIM_DEFAULT_BANDS),
    aoi: ee.Geometry | None = None,
) -> ee.Image:
    """Load WorldClim V1 bioclimatic variables with catalog scale factors applied.

    Static ~1 km climatology (1960-1990). Temperatures are stored x10 (bio04 x100) and
    are rescaled to degC; precipitation and bio15 are unscaled.

    Args:
        bands: Bioclim band names to select (default bio01, bio04, bio12, bio15).
        aoi: Optional ee.Geometry to clip to.

    Returns:
        ee.Image with the requested bands (float, physical units), native projection.

    Raises:
        ValueError: If a band has no documented scale in WORLDCLIM_SCALES.
    """
    unknown = [b for b in bands if b not in WORLDCLIM_SCALES]
    if unknown:
        raise ValueError(
            f"No scale factor documented for WorldClim band(s) {unknown}; add them to "
            "constants.WORLDCLIM_SCALES."
        )
    raw = ee.Image(WORLDCLIM_BIO_IMAGE)
    image = ee.Image.cat(
        [raw.select(b).multiply(WORLDCLIM_SCALES[b]).rename(b) for b in bands]
    ).toFloat()
    image = image.setDefaultProjection(raw.projection())
    if aoi is not None:
        image = image.clip(aoi)
    return image
