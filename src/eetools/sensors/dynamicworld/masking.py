import ee

from eetools.constants import DW_LABEL_BAND


def mask_to_cover_types(
    image: ee.Image,
    cover_types: list[int],
    label_band: str = DW_LABEL_BAND,
) -> ee.Image:
    """Mask a Dynamic World image to only the requested land-cover label classes.

    Keeps pixels whose ``label`` band equals one of ``cover_types`` and masks the rest
    across all bands, so the result retains only the requested cover types (e.g.
    ``[0, 3]`` for water + flooded vegetation).

    Args:
        image: Dynamic World ee.Image carrying the integer ``label`` band (class codes 0-8) and the probability bands.
        cover_types: Non-empty list of Dynamic World class integers (0-8) to keep; see constants.DW_CLASSES for the code-to-name mapping.
        label_band: Name of the argmax label band (default DW_LABEL_BAND).

    Returns:
        ee.Image with pixels outside ``cover_types`` masked across all bands, preserving image properties.

    Raises:
        ValueError: If cover_types is empty.
    """
    types = list(cover_types)
    if not types:
        raise ValueError(
            "cover_types must be a non-empty list of Dynamic World class integers (0-8)."
        )
    keep = image.select(label_band).remap(types, [1] * len(types), 0)
    return image.updateMask(keep)
