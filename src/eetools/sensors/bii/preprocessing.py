import ee

from eetools.constants import (
    BII_1KM_BANDS,
    BII_1KM_COLLECTION,
    BII_8KM_BANDS,
    BII_8KM_COLLECTION,
    BII_EXCLUDED_LAND_USE_CLASSES,
    BII_LAND_USE_BAND,
    BII_LAND_USE_INTENSITY_BAND,
    BII_MASK_ASSET,
    BII_PROCESSED_BANDS,
    BII_TAXON_BANDS,
)

# Resolution key -> (source collection ID, raw band order for the toBands() rename).
_BII_SOURCES = {
    "1km": (BII_1KM_COLLECTION, BII_1KM_BANDS),
    "8km": (BII_8KM_COLLECTION, BII_8KM_BANDS),
}


def get_bii_mask() -> ee.Image:
    """Load the global BII validity mask.

    The mask flags pixels where the BII assessment is valid (e.g. within the
    sub-Saharan African study area); it is applied to every processed BII image.

    Args:
        None.

    Returns:
        ee.Image to be used with updateMask on a BII image.
    """
    return ee.Image(BII_MASK_ASSET)


def _land_use_intensity(named_image: ee.Image) -> ee.Image:
    """Build the masked Land Use Intensity band from a renamed BII image.

    Land Use Intensity is undefined for certain land-use classes, so those classes
    (``BII_EXCLUDED_LAND_USE_CLASSES``) are masked out of the intensity band.

    Args:
        named_image: BII image whose bands have been renamed to the friendly names.

    Returns:
        ee.Image single-band 'Land Use Intensity' with excluded classes masked.
    """
    land_use = named_image.select(BII_LAND_USE_BAND)
    lc_mask = land_use.neq(BII_EXCLUDED_LAND_USE_CLASSES[0])
    for cls in BII_EXCLUDED_LAND_USE_CLASSES[1:]:
        lc_mask = lc_mask.And(land_use.neq(cls))
    return named_image.select(BII_LAND_USE_INTENSITY_BAND).updateMask(lc_mask)


def get_bii_image(resolution: str = "1km") -> ee.Image:
    """Build the processed multiband BII image for a given resolution.

    Mirrors the published sat-io processing: the per-resolution ImageCollection is
    flattened with ``toBands()`` and renamed, the per-taxon BII bands are self-masked
    (zeros dropped), the Land Use Intensity band is masked for excluded land-use
    classes, and the global BII validity mask is applied.

    Args:
        resolution: Source resolution, either '1km' (default) or '8km'.

    Returns:
        ee.Image with bands BII_PROCESSED_BANDS (per-taxon BII bands + 'Land Use' + 'Land Use Intensity'). Raises ValueError for an unknown resolution.
    """
    if resolution not in _BII_SOURCES:
        raise ValueError(
            f"Invalid resolution '{resolution}'. Expected one of {sorted(_BII_SOURCES)}."
        )

    collection_id, raw_bands = _BII_SOURCES[resolution]
    named = ee.ImageCollection(collection_id).toBands().rename(raw_bands)

    bii = named.select(BII_TAXON_BANDS).selfMask()
    land_use = named.select(BII_LAND_USE_BAND)
    land_use_intensity = _land_use_intensity(named)

    processed = bii.addBands([land_use, land_use_intensity])
    return processed.updateMask(get_bii_mask())


def get_bii(
    aoi: ee.Geometry,
    bands: str | list[str],
    resolution: str = "1km",
) -> ee.Image:
    """Return the BII layer for an AOI, selected to the requested band(s) and clipped.

    Selecting a single band yields a single-band image; selecting several yields a
    multiband image, in the order requested.

    Args:
        aoi: Area of interest as ee.Geometry; the result is clipped to it.
        bands: One band name or a list of band names from BII_PROCESSED_BANDS (per-taxon BII bands plus 'Land Use' and 'Land Use Intensity').
        resolution: Source resolution, either '1km' (default) or '8km'.

    Returns:
        ee.Image of the selected BII band(s) clipped to the AOI. Raises ValueError if no bands are given or any band name is not a valid BII band.
    """
    if isinstance(bands, str):
        bands = [bands]
    if not bands:
        raise ValueError("At least one band must be provided.")

    invalid = [band for band in bands if band not in BII_PROCESSED_BANDS]
    if invalid:
        raise ValueError(
            f"Unknown BII band(s): {invalid}. Valid bands: {BII_PROCESSED_BANDS}."
        )

    image = get_bii_image(resolution=resolution)
    return image.select(bands).clip(aoi)
