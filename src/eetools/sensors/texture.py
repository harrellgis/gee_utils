"""Image texture: GLCM (Haralick) and first-order neighbourhood statistics.

Texture is a sensor-agnostic band operation, like the spectral indices in
:mod:`eetools.sensors.indices`: apply it to the output of any ``get_<sensor>_collection``
(ideally a same-season composite of a vegetation index such as EVI). It wraps
``ee.Image.glcmTexture`` and ``reduceNeighborhood`` to remove GEE's failure modes:

* ``glcmTexture`` only accepts integer input (float input raises). Bands are quantized
  against a FIXED value range (``TEXTURE_VALUE_RANGES``) so texture stays comparable
  across sites and dates; a per-image stretch would not be.
* ``maxcorr`` is emitted but never computed, and ``inertia`` duplicates ``contrast``;
  both are rejected at validation time.
* ``size`` is a radius, not a width. :func:`glcm_radius` converts a ground window width.
* The neighbourhood always runs on the *request* grid, so a pixel radius covers a
  different ground extent at each export scale or map zoom. Pinning the input
  (``reproject``/``setDefaultProjection``) does not fix this; only reprojecting the
  texture output does, which :func:`calc_texture` offers via ``pin_scale=True``.

Masked input pixels are excluded from the GLCM (they are not read as gray level 0 and do
not mask the window), but texture within ``radius`` px of a mask edge is computed from
fewer pixel pairs and is biased relative to interior pixels.

Output bands are named ``<band>_<metric>_r<radius>`` for every call, so each exported
band records its window.

References: Silveira et al. 2026, *Satellite image texture for habitat and biodiversity
assessments*; Haralick et al. 1973; Hall-Beyer 2017 (quantization).
"""

from collections.abc import Sequence

import ee

from eetools.constants import (
    DEFAULT_GLCM_METRICS,
    FIRST_ORDER_STATS,
    GLCM_EQUIVALENT_METRICS,
    GLCM_METRICS,
    GLCM_UNCOMPUTED_METRICS,
    TEXTURE_MAX_QUANT_LEVELS,
    TEXTURE_MAX_RADIUS,
    TEXTURE_MIN_QUANT_LEVELS,
    TEXTURE_QUANT_LEVELS,
    TEXTURE_VALUE_RANGES,
)

ValueRange = tuple[float, float]


# --------------------------------------------------------------------------- #
# Pure-Python helpers (no Earth Engine session required)                       #
# --------------------------------------------------------------------------- #
def glcm_radius(window_m: float, scale_m: float) -> int:
    """Convert a ground window width to a GEE neighbourhood radius in pixels.

    ``glcmTexture(size=r)`` uses a ``(2r + 1) x (2r + 1)`` window, so the radius is
    ``round((window_m / scale_m - 1) / 2)``. The same ``window_m`` gives extent-matched
    windows across sensors, e.g. 750 m -> r=37 at 10 m, r=12 at 30 m, r=1 at 250 m.

    Args:
        window_m: Window width on the ground, in metres.
        scale_m: Pixel size the texture will be computed (and exported) at, in metres.

    Returns:
        int radius in pixels, in ``[1, TEXTURE_MAX_RADIUS]``.

    Raises:
        ValueError: If either argument is not positive, or the radius falls outside
            ``[1, TEXTURE_MAX_RADIUS]``.
    """
    if window_m <= 0 or scale_m <= 0:
        raise ValueError(
            f"window_m and scale_m must be positive, got {window_m} and {scale_m}."
        )
    radius = round((window_m / scale_m - 1) / 2)
    if radius < 1:
        raise ValueError(
            f"window_m={window_m} at scale_m={scale_m} gives radius {radius}; the window "
            f"must be at least 3 pixels wide ({3 * scale_m} m)."
        )
    if radius > TEXTURE_MAX_RADIUS:
        raise ValueError(
            f"window_m={window_m} at scale_m={scale_m} gives radius {radius}, above "
            f"TEXTURE_MAX_RADIUS={TEXTURE_MAX_RADIUS}; use a coarser scale or a smaller window."
        )
    return radius


def validate_glcm_metrics(metrics: Sequence[str]) -> tuple[str, ...]:
    """Validate a GLCM metric selection.

    Args:
        metrics: GLCM metric suffixes, e.g. ``("idm", "ent")``.

    Returns:
        tuple of the validated metric names, in the given order.

    Raises:
        ValueError: If a metric is unknown, is ``maxcorr`` (never computed), appears
            twice, or both members of an equivalent pair (``contrast``/``inertia``) are
            requested.
    """
    metrics = tuple(metrics)
    for m in metrics:
        if m in GLCM_UNCOMPUTED_METRICS:
            raise ValueError(f"{m} is not computed by glcmTexture (always 0).")
        if m not in GLCM_METRICS:
            raise ValueError(
                f"Unknown GLCM metric {m!r}; expected one of {GLCM_METRICS}."
            )
    if len(set(metrics)) != len(metrics):
        raise ValueError(f"Duplicate GLCM metrics in {metrics}.")
    for pair in GLCM_EQUIVALENT_METRICS:
        if set(pair) <= set(metrics):
            raise ValueError(
                f"GLCM metrics {pair} are the same quantity; request only one of them."
            )
    return metrics


def validate_first_order_stats(stats: Sequence[str]) -> tuple[str, ...]:
    """Validate a first-order texture statistic selection.

    Args:
        stats: Statistics from ``FIRST_ORDER_STATS`` (``mean``, ``sd``, ``ent1``).

    Returns:
        tuple of the validated statistic names, in the given order.

    Raises:
        ValueError: If a statistic is unknown or appears twice.
    """
    stats = tuple(stats)
    for s in stats:
        if s not in FIRST_ORDER_STATS:
            raise ValueError(
                f"Unknown first-order stat {s!r}; expected one of {FIRST_ORDER_STATS}."
            )
    if len(set(stats)) != len(stats):
        raise ValueError(f"Duplicate first-order stats in {stats}.")
    return stats


def texture_band_name(band: str, metric: str, radius: int) -> str:
    """Build the output band name for a texture metric.

    Args:
        band: Input band name, e.g. ``"EVI"``.
        metric: GLCM metric or first-order statistic, e.g. ``"idm"``.
        radius: Neighbourhood radius in pixels.

    Returns:
        str ``<band>_<metric>_r<radius>``, e.g. ``"EVI_idm_r2"``.
    """
    return f"{band}_{metric}_r{radius}"


def _validate_levels(levels: int) -> None:
    if not TEXTURE_MIN_QUANT_LEVELS <= levels <= TEXTURE_MAX_QUANT_LEVELS:
        raise ValueError(
            f"levels must be in [{TEXTURE_MIN_QUANT_LEVELS}, {TEXTURE_MAX_QUANT_LEVELS}], "
            f"got {levels}."
        )


def _validate_radius(radius: int) -> None:
    if not 1 <= radius <= TEXTURE_MAX_RADIUS:
        raise ValueError(f"radius must be in [1, {TEXTURE_MAX_RADIUS}], got {radius}.")


def _resolve_value_range(
    band: str, value_ranges: dict[str, ValueRange] | None
) -> ValueRange:
    """Return the fixed quantization range for a band (caller override first)."""
    if value_ranges and band in value_ranges:
        lo, hi = value_ranges[band]
    elif band in TEXTURE_VALUE_RANGES:
        lo, hi = TEXTURE_VALUE_RANGES[band]
    else:
        raise ValueError(
            f"No default quantization range for band {band!r}. Pass a fixed range via "
            "value_ranges, derived once over the WHOLE study domain (e.g. with "
            "eetools.utils.get_collection_min_max), never per image or per site."
        )
    if lo >= hi:
        raise ValueError(
            f"Invalid value range for {band!r}: ({lo}, {hi}); need lo < hi."
        )
    return float(lo), float(hi)


def _as_list(value: str | Sequence[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


# --------------------------------------------------------------------------- #
# Earth Engine functions (lazy, server-side)                                   #
# --------------------------------------------------------------------------- #
def quantize_band(
    image: ee.Image,
    band: str,
    value_range: ValueRange | None = None,
    levels: int = TEXTURE_QUANT_LEVELS,
) -> ee.Image:
    """Quantize one band to integer gray levels against a FIXED value range.

    The band is clamped to ``value_range``, stretched to ``[0, levels - 1]`` and cast to
    ``uint8``. The range must be the same for every site and date being compared;
    a per-image stretch makes texture incomparable. Derive a custom range once over the
    whole study domain (e.g. ``eetools.utils.get_collection_min_max``).

    Args:
        image: ee.Image containing ``band``.
        band: Name of the band to quantize; the output keeps this name.
        value_range: ``(lo, hi)`` stretch range. Defaults to
            ``TEXTURE_VALUE_RANGES[band]``.
        levels: Number of gray levels, in
            ``[TEXTURE_MIN_QUANT_LEVELS, TEXTURE_MAX_QUANT_LEVELS]`` (default 256).

    Returns:
        ee.Image with a single ``uint8`` band named ``band``.

    Raises:
        ValueError: If ``levels`` is out of bounds, the band has no default range and
            none is given, or ``lo >= hi``.
    """
    _validate_levels(levels)
    lo, hi = _resolve_value_range(band, {band: value_range} if value_range else None)
    return (
        image.select(band)
        .clamp(lo, hi)
        .unitScale(lo, hi)
        .multiply(levels - 1)
        .round()
        .toUint8()
        .rename(band)
    )


def calc_glcm_texture(
    image: ee.Image,
    bands: str | Sequence[str],
    radius: int,
    metrics: Sequence[str] = DEFAULT_GLCM_METRICS,
    value_ranges: dict[str, ValueRange] | None = None,
    levels: int = TEXTURE_QUANT_LEVELS,
    prequantized: bool = False,
) -> ee.Image:
    """Compute GLCM (Haralick) texture metrics for one or more bands.

    Each band is quantized with :func:`quantize_band` (unless ``prequantized``), then
    ``glcmTexture(size=radius, average=True)`` is run (4 directions, distance 1,
    averaged) and only the requested metrics are kept.

    Masked input pixels are excluded from the GLCM; texture within ``radius`` px of a
    mask edge is computed from fewer pairs and is biased relative to interior pixels.
    The radius is in pixels of the *requested* scale; see :func:`calc_texture`.

    Args:
        image: ee.Image containing ``bands``.
        bands: Band name or names to texture.
        radius: Neighbourhood radius in pixels (window = ``2 * radius + 1``).
        metrics: GLCM metric suffixes (default ``("idm", "ent")``).
        value_ranges: Per-band ``(lo, hi)`` quantization overrides.
        levels: Number of gray levels (default 256).
        prequantized: If True, use the bands as-is; they must already be integer
            (``glcmTexture`` raises on float input).

    Returns:
        ee.Image of texture bands only, named ``<band>_<metric>_r<radius>``, with the
        input image's properties (including ``system:time_start``).

    Raises:
        ValueError: On invalid metrics, radius, levels or value ranges.
    """
    metrics = validate_glcm_metrics(metrics)
    if not metrics:
        raise ValueError("metrics must name at least one GLCM metric.")
    _validate_radius(radius)
    band_list = _as_list(bands)

    if prequantized:
        quantized = image.select(band_list)
    else:
        quantized = ee.Image.cat(
            [
                quantize_band(image, b, _resolve_value_range(b, value_ranges), levels)
                for b in band_list
            ]
        )

    src = [f"{b}_{m}" for b in band_list for m in metrics]
    dst = [texture_band_name(b, m, radius) for b in band_list for m in metrics]
    texture = quantized.glcmTexture(size=radius, average=True).select(src, dst)
    return ee.Image(texture.copyProperties(image, image.propertyNames()))


def calc_first_order_texture(
    image: ee.Image,
    bands: str | Sequence[str],
    radius: int,
    stats: Sequence[str] = ("mean", "sd"),
    value_ranges: dict[str, ValueRange] | None = None,
    levels: int = TEXTURE_QUANT_LEVELS,
) -> ee.Image:
    """Compute first-order (histogram) texture statistics in a square neighbourhood.

    ``mean`` and ``sd`` are computed on the **unquantized** band with a combined
    ``reduceNeighborhood``; ``ent1`` (first-order entropy) runs ``ee.Image.entropy`` on
    the **quantized** band, which also needs integer input. Variance is omitted because
    it is redundant with ``sd``.

    Args:
        image: ee.Image containing ``bands``.
        bands: Band name or names to texture.
        radius: Neighbourhood radius in pixels (window = ``2 * radius + 1``).
        stats: Statistics from ``("mean", "sd", "ent1")`` (default ``("mean", "sd")``).
        value_ranges: Per-band ``(lo, hi)`` quantization overrides; only used by
            ``ent1``.
        levels: Number of gray levels for ``ent1`` (default 256).

    Returns:
        ee.Image of texture bands only, named ``<band>_<stat>_r<radius>``, with the input
        image's properties (including ``system:time_start``).

    Raises:
        ValueError: On invalid stats, radius, levels or (for ``ent1``) value ranges.
    """
    stats = validate_first_order_stats(stats)
    if not stats:
        raise ValueError("stats must name at least one first-order statistic.")
    _validate_radius(radius)
    kernel = ee.Kernel.square(radius, "pixels")
    band_list = _as_list(bands)

    mean_sd = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)

    out: list[ee.Image] = []
    for b in band_list:
        per_stat: dict[str, ee.Image] = {}
        if "mean" in stats or "sd" in stats:
            neigh = image.select([b]).reduceNeighborhood(reducer=mean_sd, kernel=kernel)
            per_stat["mean"] = neigh.select(f"{b}_mean")
            per_stat["sd"] = neigh.select(f"{b}_stdDev")
        if "ent1" in stats:
            q = quantize_band(image, b, _resolve_value_range(b, value_ranges), levels)
            per_stat["ent1"] = q.entropy(kernel)
        out.extend(per_stat[s].rename(texture_band_name(b, s, radius)) for s in stats)

    texture = ee.Image.cat(out)
    return ee.Image(texture.copyProperties(image, image.propertyNames()))


def calc_texture(
    image: ee.Image,
    bands: str | Sequence[str],
    window_m: float | Sequence[float],
    scale_m: float,
    glcm_metrics: Sequence[str] = DEFAULT_GLCM_METRICS,
    first_order: Sequence[str] = (),
    mask: ee.Image | None = None,
    value_ranges: dict[str, ValueRange] | None = None,
    levels: int = TEXTURE_QUANT_LEVELS,
    pin_scale: bool = False,
    crs: str | None = None,
) -> ee.Image:
    """Compute GLCM and first-order texture at one or more ground window sizes.

    The main texture entry point. Recipe, in order:

    1. Convert each ``window_m`` to a pixel radius at ``scale_m`` (:func:`glcm_radius`).
    2. Apply ``mask`` (roads, built-up, water) **before** texturing. Masked pixels are
       excluded from the GLCM, but texture within ``radius`` px of a mask edge is
       computed from fewer pairs and is biased. To drop those pixels, apply
       ``mask.focalMin(radius, "square", "pixels")`` to the result.
    3. Compute GLCM and first-order bands per radius. The input is NOT reprojected:
       pinning the input does not fix the window extent and only adds cost.
    4. If ``pin_scale``, reproject the texture **output** to ``crs`` at ``scale_m``.

    The radius is in pixels of the **requested** scale. An unpinned result is correct
    when exported or sampled at ``scale_m`` (the common case, no pinning needed). For
    map display or ``reduceRegion``/sampling at any other scale, set ``pin_scale=True``
    (about 55% more EECU on a 10 m export, where it changes nothing).

    ``window_m`` has no default; choose it from the ecological scale:

    ===============================================  =========================
    Purpose                                          Window
    ===============================================  =========================
    Within-habitat heterogeneity                     3x3 to 7x7 px
    Between-class heterogeneity, homogeneous grass   > 15x15 px
    Wildlife                                         territory / home range
    ===============================================  =========================

    Prefer a vegetation index (EVI; NDVI where haze is common) on a same-season
    composite; texture of a single reflectance band is not a vegetation-heterogeneity
    proxy. Buffer the AOI by at least the window half-width, texture, then clip.

    Args:
        image: ee.Image containing ``bands`` (typically a composite).
        bands: Band name or names to texture.
        window_m: Window width(s) in metres; a list gives multi-scale texture with one
            set of bands per window.
        scale_m: Pixel size in metres the texture is defined at (and should be exported at).
        glcm_metrics: GLCM metric suffixes (default ``("idm", "ent")``); may be empty.
        first_order: First-order stats from ``("mean", "sd", "ent1")`` (default none).
        mask: Optional ee.Image mask applied to the input before texturing.
        value_ranges: Per-band ``(lo, hi)`` quantization overrides.
        levels: Number of gray levels (default 256).
        pin_scale: If True, reproject the texture output to ``crs`` at ``scale_m`` so the
            window keeps its ground extent at any request scale (default False).
        crs: CRS for ``pin_scale``. Defaults to the first band's CRS; pass it explicitly
            for composites, whose default projection is EPSG:4326.

    Returns:
        ee.Image of texture bands only, named ``<band>_<metric>_r<radius>`` (GLCM then
        first-order, per window), with the input image's properties plus
        ``texture_window_m``, ``texture_radius_px``, ``texture_scale_m``,
        ``texture_levels``, ``texture_value_ranges`` and ``texture_pinned``.

    Raises:
        ValueError: If nothing is requested, two windows map to the same radius, or any
            window, metric, stat, level or value range is invalid.
    """
    band_list = _as_list(bands)
    windows = [float(window_m)] if isinstance(window_m, int | float) else list(window_m)
    glcm_metrics = validate_glcm_metrics(glcm_metrics)
    first_order = validate_first_order_stats(first_order)
    if not glcm_metrics and not first_order:
        raise ValueError("Request at least one GLCM metric or first-order statistic.")
    _validate_levels(levels)

    radii = [glcm_radius(w, scale_m) for w in windows]
    if len(set(radii)) != len(radii):
        raise ValueError(
            f"window_m={windows} map to duplicate radii {radii} at scale_m={scale_m}."
        )

    # Validate ranges up front for every band that gets quantized.
    quantized_bands = band_list if (glcm_metrics or "ent1" in first_order) else []
    ranges = {b: _resolve_value_range(b, value_ranges) for b in quantized_bands}

    source = image.updateMask(mask) if mask is not None else image

    parts = []
    for r in radii:
        if glcm_metrics:
            parts.append(
                calc_glcm_texture(source, band_list, r, glcm_metrics, ranges, levels)
            )
        if first_order:
            parts.append(
                calc_first_order_texture(
                    source, band_list, r, first_order, ranges, levels
                )
            )
    texture = ee.Image.cat(parts)

    if pin_scale:
        target_crs = crs or image.select(band_list[0]).projection().crs()
        texture = texture.reproject(crs=target_crs, scale=scale_m)

    texture = ee.Image(texture.copyProperties(image, image.propertyNames()))
    return texture.set(
        {
            "texture_window_m": windows,
            "texture_radius_px": radii,
            "texture_scale_m": scale_m,
            "texture_levels": levels,
            "texture_value_ranges": {b: list(v) for b, v in ranges.items()},
            "texture_pinned": pin_scale,
        }
    )


def add_texture_to_collection(
    collection: ee.ImageCollection, **calc_texture_kwargs
) -> ee.ImageCollection:
    """Append :func:`calc_texture` bands to every image of a composite collection.

    Meant for per-period composites (e.g. from
    ``eetools.compositing.build_period_composites``), NOT scene-level collections:
    texture belongs on a peak-greenness or same-season composite, never on every scene.

    Args:
        collection: ee.ImageCollection of composites.
        **calc_texture_kwargs: Keyword arguments forwarded to :func:`calc_texture`
            (``bands``, ``window_m``, ``scale_m`` are required).

    Returns:
        ee.ImageCollection with texture bands added to each image; image properties
        (including ``system:time_start``) are preserved.

    Raises:
        ValueError: Propagated from :func:`calc_texture` argument validation. ``map``
            traces the function client-side, so this is raised at call time.
    """

    def _add(img: ee.Image) -> ee.Image:
        img = ee.Image(img)
        return img.addBands(calc_texture(img, **calc_texture_kwargs))

    return ee.ImageCollection(collection).map(_add)
