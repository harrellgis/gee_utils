"""Sensor-agnostic masking helpers shared across the per-sensor modules.

The spectral water mask and the mask-application logic are identical for
Sentinel-2, Landsat, and HLS apart from the NIR band name, so they live here and
each sensor module wraps them with its band name. Keeping one implementation
means a threshold or logic change happens in exactly one place.
"""

import ee

# The spectral water mask reads these index bands off the collection median, so any
# custom index selection must still produce them when water masking is enabled.
WATER_MASK_INDICES = ("NDVI", "MNDWI")


def validate_water_mask_selection(
    band_map: dict,
    indices: list[str] | tuple[str, ...] | None,
    domains: list[str] | tuple[str, ...] | None,
) -> None:
    """Ensure a custom index selection still yields the bands the water mask needs.

    Args:
        band_map: Logical-key -> band-name map for the sensor (drives availability).
        indices: Explicit index selection passed to the collection builder, or None.
        domains: Domain selection passed to the collection builder, or None.

    Returns:
        None. Raises ValueError if water masking is requested but the resolved selection
        omits NDVI or MNDWI.
    """
    # Local import to avoid a module-load cycle (indices imports nothing from here).
    from eetools.sensors.indices import resolve_index_names

    resolved = set(resolve_index_names(band_map, indices=indices, domains=domains))
    missing = [name for name in WATER_MASK_INDICES if name not in resolved]
    if missing:
        raise ValueError(
            f"Water masking requires {missing} in the index selection; include them "
            "(e.g. add to `indices`, or select the relevant domains) or pass "
            "apply_water_masking=False."
        )


def build_non_water_mask(
    collection: ee.ImageCollection,
    nir_band: str,
    mndwi_thresh: float = 0.1,
    ndvi_thresh: float = 0.2,
    nir_thresh: float = 0.15,
) -> ee.Image:
    """Build a boolean non-water mask from a collection median composite.

    A pixel is treated as water when it is simultaneously high in MNDWI, low in
    NDVI, and dark in the NIR band; the returned mask is the inverse.

    Args:
        collection: ee.ImageCollection with MNDWI, NDVI, and the named NIR band already computed.
        nir_band: Name of the NIR reflectance band (e.g. 'B8' for S2, 'SR_B5' for L8, 'NIR' for HLS).
        mndwi_thresh: MNDWI threshold above which a pixel is considered water (default 0.1).
        ndvi_thresh: NDVI threshold below which a pixel is considered water (default 0.2).
        nir_thresh: NIR reflectance threshold below which a pixel is considered water (default 0.15).

    Returns:
        ee.Image with a single 'non_water' band where 1 = land and 0 = water.
    """
    comp = collection.median()
    water = (
        comp.select("MNDWI")
        .gt(mndwi_thresh)
        .And(comp.select("NDVI").lt(ndvi_thresh))
        .And(comp.select(nir_band).lt(nir_thresh))
    )
    return water.Not().rename("non_water")


def apply_water_mask(image: ee.Image, non_water_mask: ee.Image) -> ee.Image:
    """Apply a precomputed non-water mask to all bands of an image.

    Args:
        image: ee.Image to mask.
        non_water_mask: Single-band ee.Image where 1 = valid land pixel (output of build_non_water_mask).

    Returns:
        ee.Image with water pixels masked out across all bands.
    """
    return image.updateMask(non_water_mask)


def apply_cloud_mask(image: ee.Image, mask_band: str = "cloudmask") -> ee.Image:
    """Apply the inverse of a binary mask band to mask flagged pixels from all bands.

    Args:
        image: ee.Image carrying a binary mask band (1 = pixel to exclude).
        mask_band: Name of the mask band to invert and apply (default 'cloudmask').

    Returns:
        ee.Image with the flagged (cloud/shadow/etc.) pixels masked out across all bands.
    """
    return image.updateMask(image.select(mask_band).Not())
