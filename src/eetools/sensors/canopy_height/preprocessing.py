from collections.abc import Sequence

import ee

from eetools.constants import (
    CANOPY_HEIGHT_ASSET_DATES_PROPERTY,
    CANOPY_HEIGHT_ASSET_LICENCE_PROPERTY,
    CANOPY_HEIGHT_ASSET_VERSION_PROPERTY,
    CANOPY_HEIGHT_SOURCE_BAND,
    CHM_VALID_FRACTION_BAND,
    ECHOSAT_BAND_TEMPLATE,
    ECHOSAT_COLLECTION,
    ECHOSAT_SCALE_FACTOR,
    ECHOSAT_SOURCE_BANDS,
    ECHOSAT_YEARS,
    ESA_WC_WATER_CLASS,
    ETH_CANOPY_HEIGHT_2020,
    ETH_CANOPY_HEIGHT_NODATA,
    ETH_CANOPY_HEIGHT_SD_2020,
    GLAD_CANOPY_HEIGHT_2019,
    GLAD_CANOPY_HEIGHT_MAX_M,
    META_CANOPY_HEIGHT_BAND,
    META_CANOPY_HEIGHT_COLLECTION,
    META_CANOPY_HEIGHT_LICENCE,
    META_CANOPY_HEIGHT_VERSION,
    WOODY_FRACTION_BAND_PREFIX,
    WOODY_FRACTION_HEIGHTS_M,
    WOODY_FRACTION_MAX_PIXELS,
    WOODY_FRACTION_SCALE_M,
)
from eetools.sensors.esa.preprocessing import get_land_cover

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


def _is_default_chm(asset: str | None) -> bool:
    return asset is None or asset == META_CANOPY_HEIGHT_COLLECTION


def _get_asset_metadata(asset: str) -> dict:
    """``ee.data.getAsset`` with a readable error when the asset cannot be read."""
    try:
        return ee.data.getAsset(asset)
    except Exception as exc:  # noqa: BLE001 - EE raises a generic EEException
        raise ValueError(
            f"Cannot read canopy height asset '{asset}' ({exc}). Check the ID and "
            "that this Earth Engine project can read it."
        ) from exc


def get_canopy_height(
    aoi: ee.Geometry | None = None,
    *,
    pin_projection: bool = False,
    asset: str | None = None,
) -> ee.Image:
    """Load a canopy height model as a single ``canopy_height`` band (metres).

    By default this is the Meta/WRI 1 m v1 mosaic. Its source is a tiled
    ee.ImageCollection, so the tiles are mosaicked into one wall-to-wall image. v1 is
    a fused 2009-2020 baseline (~80% 2018-2020 imagery), not a dated annual layer.

    Pass ``asset`` to load another CHM instead, e.g. a per-project Meta/WRI CHMv2 clip
    (``projects/<p>/assets/chmv2_...``). An ``ee.Image`` asset is used as it is (it
    keeps its native projection, so ``pin_projection`` changes nothing). An
    ``ee.ImageCollection`` asset is mosaicked like v1. Either way the first band
    becomes ``canopy_height``. Checking the asset type costs one ``ee.data.getAsset``
    call.

    Every CHM here is a single-date baseline. Never difference one against another
    epoch or product to infer change.

    Args:
        aoi: Optional area of interest as ee.Geometry; if provided, the image is clipped to it.
        pin_projection: Set a mosaic's default projection to its tiles' native (~1 m) projection. A bare ``mosaic()`` has none, which ``reduceResolution`` rejects; pass True before aggregating.
        asset: Optional EE ID of another canopy height image or image collection (metres in the first band). None (default) loads Meta v1.

    Returns:
        ee.Image with a single 'canopy_height' band (metres), clipped to aoi when provided.

    Raises:
        ValueError: If ``asset`` cannot be read or is neither an image nor an image collection.
    """
    if _is_default_chm(asset):
        collection = ee.ImageCollection(META_CANOPY_HEIGHT_COLLECTION)
        canopy = collection.mosaic().select(
            [META_CANOPY_HEIGHT_BAND], ["canopy_height"]
        )
        if pin_projection:
            native = collection.first().select(META_CANOPY_HEIGHT_BAND).projection()
            canopy = canopy.setDefaultProjection(native)
    else:
        asset_id = str(asset)
        kind = _get_asset_metadata(asset_id).get("type")
        if kind == "IMAGE":
            canopy = ee.Image(asset_id).select([0], ["canopy_height"])
        elif kind == "IMAGE_COLLECTION":
            collection = ee.ImageCollection(asset_id)
            canopy = collection.mosaic().select([0], ["canopy_height"])
            if pin_projection:
                native = ee.Image(collection.first()).select(0).projection()
                canopy = canopy.setDefaultProjection(native)
        else:
            raise ValueError(
                f"Canopy height asset '{asset_id}' is a {kind}; expected an IMAGE "
                "or IMAGE_COLLECTION."
            )
    if aoi is not None:
        canopy = canopy.clip(aoi)
    return canopy


def canopy_height_asset_info(asset: str | None = None) -> dict[str, str]:
    """Data-register fields of the canopy height source used by ``get_canopy_height``.

    For the default Meta v1 mosaic this needs no EE call. For another asset it reads
    the asset's ``version``, ``acq_dates`` and ``license`` properties, which the
    CHMv2 project clips carry. A missing property is reported as such and never
    guessed. A licence read from an asset property is the uploader's record, not the
    provider's terms page.

    Args:
        asset: EE ID passed to ``get_canopy_height(asset=...)``; None for Meta v1.

    Returns:
        dict with ``dataset_id``, ``version`` (e.g. 'v2 (acquired 2017-02-12..2020-02-20)') and ``licence``.

    Raises:
        ValueError: If ``asset`` cannot be read.
    """
    if _is_default_chm(asset):
        return {
            "dataset_id": META_CANOPY_HEIGHT_COLLECTION,
            "version": META_CANOPY_HEIGHT_VERSION,
            "licence": META_CANOPY_HEIGHT_LICENCE,
        }
    asset_id = str(asset)
    props = _get_asset_metadata(asset_id).get("properties", {}) or {}
    version = str(props.get(CANOPY_HEIGHT_ASSET_VERSION_PROPERTY) or "unrecorded")
    dates = props.get(CANOPY_HEIGHT_ASSET_DATES_PROPERTY)
    if dates:
        version = f"{version} (acquired {dates})"
    licence = props.get(CANOPY_HEIGHT_ASSET_LICENCE_PROPERTY)
    return {
        "dataset_id": asset_id,
        "version": version,
        "licence": (
            f"{licence} (from asset property)" if licence else "unrecorded on asset"
        ),
    }


def get_eth_canopy_height_2020(aoi: ee.Geometry | None = None) -> ee.Image:
    """ETH Global Canopy Height 2020 (10 m, Sentinel-2 + GEDI) and its uncertainty.

    Static 2020 baseline from Lang et al. (2023). Built-up areas, snow/ice and
    permanent water were set to no-data (255) by the producers and are masked here.
    CC BY 4.0.

    Args:
        aoi: Optional area of interest as ee.Geometry; the image is clipped to it when given.

    Returns:
        ee.Image with bands ``canopy_height`` and ``canopy_height_sd`` (metres, native 10 m projection).
    """
    bands = []
    for asset, name in (
        (ETH_CANOPY_HEIGHT_2020, "canopy_height"),
        (ETH_CANOPY_HEIGHT_SD_2020, "canopy_height_sd"),
    ):
        band = ee.Image(asset).select([CANOPY_HEIGHT_SOURCE_BAND], [name])
        bands.append(band.updateMask(band.neq(ETH_CANOPY_HEIGHT_NODATA)))
    image = ee.Image.cat(bands)
    return image.clip(aoi) if aoi is not None else image


def get_glad_canopy_height_2019(
    aoi: ee.Geometry | None = None, *, pin_projection: bool = False
) -> ee.Image:
    """GLAD Global Forest Canopy Height 2019 (30 m, GEDI + Landsat), mosaicked.

    Values 0-60 are canopy height in metres; the codes 101 (water), 102 (snow/ice)
    and 103 (no data) are masked. 0 is kept as 0 m (non-forest): the GLAD page
    also lists 0 as no-data for its EE copy, so treat 0 with care. The model
    saturates above ~30 m. Static 2019 baseline (Potapov et al.).

    Args:
        aoi: Optional area of interest as ee.Geometry; the image is clipped to it when given.
        pin_projection: Set the mosaic's default projection to the tiles' native 30 m projection; pass True before ``reduceResolution``.

    Returns:
        ee.Image with a single ``canopy_height`` band (metres).
    """
    collection = ee.ImageCollection(GLAD_CANOPY_HEIGHT_2019)
    height = collection.mosaic().select([CANOPY_HEIGHT_SOURCE_BAND], ["canopy_height"])
    height = height.updateMask(height.lte(GLAD_CANOPY_HEIGHT_MAX_M))
    if pin_projection:
        native = collection.first().select(CANOPY_HEIGHT_SOURCE_BAND).projection()
        height = height.setDefaultProjection(native)
    return height.clip(aoi) if aoi is not None else height


def _check_echosat_years(years: Sequence[int] | None) -> list[int]:
    if years is None:
        return list(ECHOSAT_YEARS)
    out = [int(y) for y in years]
    if not out:
        raise ValueError("years must name at least one year.")
    missing = sorted(set(out) - set(ECHOSAT_YEARS))
    if missing:
        raise ValueError(
            f"ECHOSAT has no maps for {missing}; available years are "
            f"{list(ECHOSAT_YEARS)}."
        )
    if len(set(out)) != len(out):
        raise ValueError(f"Duplicate years in {out}.")
    return out


def echosat_band_names(years: Sequence[int] | None = None) -> list[str]:
    """Output band names of ``get_echosat``, e.g. ``['canopy_height_2018', ...]``.

    Args:
        years: ECHOSAT years (default every year, 2018-2024).

    Returns:
        list[str], one ``canopy_height_<year>`` per year, in the order given.

    Raises:
        ValueError: If a year is outside 2018-2024, repeated, or ``years`` is empty.
    """
    return [ECHOSAT_BAND_TEMPLATE.format(year=y) for y in _check_echosat_years(years)]


def get_echosat(
    aoi: ee.Geometry | None = None,
    years: Sequence[int] | None = None,
    *,
    mask_water: bool = True,
    pin_projection: bool = False,
) -> ee.Image:
    """ECHOSAT annual canopy height 2018-2024 (10 m), one band per year, in metres.

    The MGRS-zone tiles are mosaicked (filtered to ``aoi`` when given). Bands ``b1..b7``
    are 2018..2024, stored as int16 centimetres, and are returned as float metres
    named ``canopy_height_<year>``. ECHOSAT has no water mask of its own (open
    water reads ~2.5-5.8 m), so ESA WorldCover permanent water (class 80) is masked
    by default.

    ECHOSAT is one consistent annual series, so year-to-year differences within it
    are meaningful (unlike differencing two single-date products). Its accuracy is
    indicative: treat changes of a metre or two as noise, and never difference it
    against another CHM.

    Args:
        aoi: Optional area of interest; tiles are filtered to it and the image is clipped to it.
        years: Years to return (default every year, 2018-2024), in the band order wanted.
        mask_water: Mask ESA WorldCover v200 permanent water, class 80 (default True).
        pin_projection: Set the mosaic's default projection to the first tile's native 10 m UTM projection. A bare ``mosaic()`` has none, which ``reduceResolution`` rejects; pass True before aggregating.

    Returns:
        ee.Image (float) with one ``canopy_height_<year>`` band (metres) per year.

    Raises:
        ValueError: If a year is outside 2018-2024, repeated, or ``years`` is empty.
    """
    wanted = _check_echosat_years(years)
    sources = [ECHOSAT_SOURCE_BANDS[ECHOSAT_YEARS.index(y)] for y in wanted]
    names = echosat_band_names(wanted)
    collection = ee.ImageCollection(ECHOSAT_COLLECTION)
    if aoi is not None:
        collection = collection.filterBounds(aoi)
    height = (
        collection.mosaic()
        .select(sources, names)
        .toFloat()
        .multiply(ECHOSAT_SCALE_FACTOR)
    )
    if mask_water:
        height = height.updateMask(get_land_cover().neq(ESA_WC_WATER_CLASS))
    if pin_projection:
        native = ee.Image(collection.first()).select(sources[0]).projection()
        height = height.setDefaultProjection(native)
    return height.clip(aoi) if aoi is not None else height


def get_echosat_collection(
    aoi: ee.Geometry | None = None,
    years: Sequence[int] | None = None,
    *,
    mask_water: bool = True,
    pin_projection: bool = False,
) -> ee.ImageCollection:
    """ECHOSAT as a time series: one ``canopy_height`` image per year.

    The same data as ``get_echosat``, laid out for the time-series tools
    (``run_site_timeseries``, ``image_collection_to_region_stats_fc``).

    Args:
        aoi: Optional area of interest (see ``get_echosat``).
        years: Years to return (default 2018-2024).
        mask_water: Mask ESA WorldCover permanent water (default True).
        pin_projection: Pin each image to the first tile's native projection (see ``get_echosat``).

    Returns:
        ee.ImageCollection of float ``canopy_height`` images (metres), each with ``year`` and ``system:time_start`` (1 January).

    Raises:
        ValueError: If a year is outside 2018-2024, repeated, or ``years`` is empty.
    """
    wanted = _check_echosat_years(years)
    stack = get_echosat(
        aoi, wanted, mask_water=mask_water, pin_projection=pin_projection
    )
    images = [
        stack.select([name], ["canopy_height"]).set(
            {"year": year, "system:time_start": ee.Date.fromYMD(year, 1, 1).millis()}
        )
        for year, name in zip(wanted, echosat_band_names(wanted))
    ]
    return ee.ImageCollection(images)


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

    The default CHM is the Meta/WRI 1 m v1 canopy height (``get_canopy_height``): a
    fused 2009-2020 baseline (~80% 2018-2020 imagery), not a dated layer. For
    another CHM (e.g. a CHMv2 project clip) pass
    ``chm=get_canopy_height(asset=..., pin_projection=True)``. Never difference a
    CHM against another epoch to infer change.

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
