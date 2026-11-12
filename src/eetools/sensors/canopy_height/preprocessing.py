from collections.abc import Sequence

import ee

from eetools.constants import (
    CHM_VALID_FRACTION_BAND,
    META_CANOPY_HEIGHT_BAND,
    META_CANOPY_HEIGHT_COLLECTION,
    WOODY_FRACTION_BAND_PREFIX,
    WOODY_FRACTION_HEIGHTS_M,
    WOODY_FRACTION_MAX_PIXELS,
    WOODY_FRACTION_SCALE_M,
)

# --------------------------------------------------------------------------- #
# Private helpers                                                              #
# --------------------------------------------------------------------------- #


def _height_label(height_m: float) -> str:
    value = float(height_m)
    if value.is_integer():
        return str(int(value))
    return repr(value).replace(".", "p")


def woody_fraction_band_names(heights_m: Sequence[float]) -> list[str]:
    """Return the woody-fraction band names for a set of height thresholds.

    Args:
        heights_m: Height thresholds in metres, e.g. ``(2, 2.5, 5)``.

    Returns:
        list[str]. E.g. ``["woody_frac_h2", "woody_frac_h2p5", "woody_frac_h5"]``.
    """
    return [f"{WOODY_FRACTION_BAND_PREFIX}{_height_label(h)}" for h in heights_m]


# --------------------------------------------------------------------------- #
# Canopy height                                                                #
# --------------------------------------------------------------------------- #


def get_canopy_height(
    aoi: ee.Geometry | None = None, *, pin_projection: bool = False
) -> ee.Image:
    """Load the Meta/WRI 1m global canopy height mosaic, optionally clipped to an AOI.

    The source asset is a tiled ee.ImageCollection; tiles are mosaicked to a single
    wall-to-wall image. This is a fused 2009-2020 baseline (~80% 2018-2020 imagery),
    not a dated annual layer -- do not difference it against another epoch to infer
    change.

    Args:
        aoi: Optional area of interest as ee.Geometry; if provided, the image is clipped to it.
        pin_projection: Set the mosaic's default projection to its tiles' native (~1 m) projection. A bare ``mosaic()`` has none, which ``reduceResolution`` rejects; pass True before aggregating.

    Returns:
        ee.Image with a single 'canopy_height' band (metres), clipped to aoi when provided.
    """
    collection = ee.ImageCollection(META_CANOPY_HEIGHT_COLLECTION)
    canopy = collection.mosaic().select([META_CANOPY_HEIGHT_BAND], ["canopy_height"])
    if pin_projection:
        native = collection.first().select(META_CANOPY_HEIGHT_BAND).projection()
        canopy = canopy.setDefaultProjection(native)
    if aoi is not None:
        canopy = canopy.clip(aoi)
    return canopy


def woody_fraction_from_chm(
    chm: ee.Image | None = None,
    *,
    heights_m: Sequence[float] = WOODY_FRACTION_HEIGHTS_M,
    crs: str,
    crs_transform: list[float] | None = None,
    scale: float = WOODY_FRACTION_SCALE_M,
    max_pixels: int = WOODY_FRACTION_MAX_PIXELS,
) -> ee.Image:
    """Aggregate a fine canopy height model to the fraction of each coarse pixel
    covered by canopy at or above each height threshold.

    For each ``h``: ``chm.gte(h).reduceResolution(mean, maxPixels=max_pixels)``
    reprojected to the output grid, i.e. the area-weighted share of *valid* CHM
    pixels with height >= h. ``chm_valid_frac`` is the share of the output pixel
    covered by valid (unmasked) CHM pixels, so a fraction resting on few valid
    pixels can be screened out. Output is float32 in [0, 1];
    ``export_composites_per_site`` scales it to int16 like the other bands.

    The default CHM is the Meta/WRI 1 m canopy height (``get_canopy_height``): a
    fused 2009-2020 baseline (~80% 2018-2020 imagery), not a dated layer. Never
    difference it against another epoch to infer change.

    Gotcha: ``reduceResolution`` defaults to ``maxPixels=64``, but 10 m from 1 m
    needs 100 input pixels per output pixel, so the default here is 256. Raise it
    for coarser outputs (EE errors with "Too many input pixels per output pixel").
    The final ``reproject`` is what drives the aggregation; keep this at the end of
    the pipeline and export on the same grid (pass the export's ``crs_transform``).

    Args:
        chm: Single-band canopy height image in metres at its native (fine) resolution, with a fixed default projection (call ``setDefaultProjection`` on a ``mosaic()``, which has none); defaults to the Meta 1 m mosaic, pinned to its tiles' native projection.
        heights_m: Height thresholds in metres, each > 0 (default 2, 3, 5).
        crs: Output CRS, e.g. the project UTM zone.
        crs_transform: Output affine ``[xScale, 0, x0, 0, -yScale, y0]`` (preferred; e.g. from ``io.snapped_transform``). When given, ``scale`` is ignored.
        scale: Output pixel size in ``crs`` units when ``crs_transform`` is None (default 10).
        max_pixels: ``reduceResolution`` maxPixels (default 256).

    Returns:
        ee.Image (float32) with bands ``woody_frac_h<label>`` per height (``h2``, ``h2p5`` for 2.5 m, ...) then ``chm_valid_frac``.

    Raises:
        ValueError: If ``heights_m`` is empty, has a non-positive or duplicate height, ``crs_transform`` is not 6 numbers, or ``scale``/``max_pixels`` is not positive.
    """
    heights = [float(h) for h in heights_m]
    if not heights:
        raise ValueError("heights_m must contain at least one height.")
    if any(h <= 0 for h in heights):
        raise ValueError(f"heights_m must all be > 0, got {list(heights_m)}.")
    names = woody_fraction_band_names(heights)
    if len(set(names)) != len(names):
        raise ValueError(f"heights_m has duplicates: {list(heights_m)}.")
    if crs_transform is not None and len(crs_transform) != 6:
        raise ValueError(
            f"crs_transform must have 6 elements, got {len(crs_transform)}."
        )
    if scale <= 0 or max_pixels <= 0:
        raise ValueError("scale and max_pixels must be > 0.")

    if chm is None:
        # mosaic() has no usable default projection, which reduceResolution
        # rejects; pin the mosaic to the native (~1 m) projection of its tiles.
        chm = get_canopy_height(pin_projection=True)
    height = ee.Image(chm).select(0)

    def _to_grid(image: ee.Image) -> ee.Image:
        reduced = image.reduceResolution(
            reducer=ee.Reducer.mean(), maxPixels=max_pixels
        )
        if crs_transform is not None:
            return reduced.reproject(
                crs=crs, crsTransform=[float(v) for v in crs_transform]
            )
        return reduced.reproject(crs=crs, scale=scale)

    # gte keeps the CHM mask, so each fraction is over valid CHM pixels only. The
    # mask image is itself unmasked (0 where invalid), so its mean is the valid share.
    bands = [
        _to_grid(height.gte(h).toFloat()).rename(n) for h, n in zip(heights, names)
    ]
    valid = _to_grid(height.mask().gt(0).toFloat()).rename(CHM_VALID_FRACTION_BAND)
    return ee.Image.cat([*bands, valid]).toFloat()
