import ee

from eetools.constants import (
    S1_TIMESCAN_BANDS,
    S1_TIMESCAN_INPUT_BANDS,
    S1_TIMESCAN_STATS,
)

# --------------------------------------------------------------------------- #
# Private helpers                                                              #
# --------------------------------------------------------------------------- #


def _validate_season_months(season_months: tuple[int, int]) -> None:
    start_month, end_month = season_months
    if not (1 <= start_month <= end_month <= 12):
        raise ValueError(
            f"season_months must satisfy 1 <= start <= end <= 12; got {season_months}"
        )


def _timescan_reducer() -> ee.Reducer:
    # ee.Reducer.<stat> only exists after ee.Initialize(), so look it up at call time.
    # Combined reducers name outputs "<band>_<stat>", in S1_TIMESCAN_STATS order.
    reducer = getattr(ee.Reducer, S1_TIMESCAN_STATS[0])()
    for stat in S1_TIMESCAN_STATS[1:]:
        reducer = reducer.combine(getattr(ee.Reducer, stat)(), sharedInputs=True)
    return reducer


# --------------------------------------------------------------------------- #
# Backscatter conversions                                                      #
# --------------------------------------------------------------------------- #


def db_to_linear(image: ee.Image) -> ee.Image:
    """Convert Sentinel-1 backscatter from decibels to linear power (10^(dB/10)).

    Args:
        image: ee.Image of backscatter bands in dB (e.g. VV, VH).

    Returns:
        ee.Image of the same bands in linear power, with band names and system:time_start preserved.
    """
    linear = ee.Image.constant(10).pow(image.divide(10)).rename(image.bandNames())
    return ee.Image(linear.copyProperties(image, ["system:time_start"]))


def add_polarization_ratios(image: ee.Image) -> ee.Image:
    """Append per-image VV/VH polarization ratios computed in linear power.

    Adds ``VVVH_ratio`` (VV / VH) and ``VVVH_ND_ratio`` ((VV - VH) / (VV + VH)), the
    two ratio bands summarized by the timescan.

    Args:
        image: Sentinel-1 ee.Image carrying VV and VH backscatter bands in dB.

    Returns:
        ee.Image with the original bands plus VVVH_ratio and VVVH_ND_ratio, preserving image properties.
    """
    linear = db_to_linear(image.select(["VV", "VH"]))
    vv, vh = linear.select("VV"), linear.select("VH")
    ratio = vv.divide(vh).rename("VVVH_ratio")
    nd_ratio = vv.subtract(vh).divide(vv.add(vh)).rename("VVVH_ND_ratio")
    return image.addBands(ratio).addBands(nd_ratio)


# --------------------------------------------------------------------------- #
# Timescan                                                                     #
# --------------------------------------------------------------------------- #


def build_s1_timescan(collection: ee.ImageCollection) -> ee.Image:
    """Summarize a Sentinel-1 VV+VH collection into a single multi-temporal timescan image.

    Each image gets VV/VH ratio bands (see :func:`add_polarization_ratios`), then
    VV, VH, VVVH_ratio, and VVVH_ND_ratio are each reduced over time with min, max,
    mean, stdDev (population), and median. Five comparison bands follow:
    ``VV_median_VH_median_nat`` (VV_median - VH_median, dB),
    ``VV_median_VH_median_db`` (VV_median / VH_median, dB), ``VV_CV`` / ``VH_CV``
    (stdDev / mean in linear power), and ``NDCV`` ((VV_CV - VH_CV) / (VV_CV + VH_CV)).
    The band set and order match the GEE book A3.2 mangrove timescans.

    Filter the collection to the window of interest first (or use
    :func:`build_seasonal_s1_timescan`); statistics are computed in dB for the
    backscatter bands, so pass edge-masked, speckle-filtered dB input such as the
    output of ``get_s1_grd_collection``.

    Args:
        collection: ee.ImageCollection of Sentinel-1 images with VV and VH backscatter bands in dB.

    Returns:
        ee.Image with the 25 S1_TIMESCAN_BANDS, in that order. An empty collection yields the same bands, fully masked.
    """
    # Reducing an empty collection returns an image with no bands, which breaks the
    # band selections below. A fully masked VV/VH placeholder keeps the band structure;
    # reducers ignore masked pixels, so it never changes the statistics.
    placeholder = ee.Image.constant([0, 0]).rename(["VV", "VH"]).toFloat().selfMask()
    collection = ee.ImageCollection(collection).merge(ee.ImageCollection([placeholder]))

    stats = (
        collection.map(add_polarization_ratios)
        .select(S1_TIMESCAN_INPUT_BANDS)
        .reduce(_timescan_reducer())
    )

    vv_median, vh_median = stats.select("VV_median"), stats.select("VH_median")
    median_nat = vv_median.subtract(vh_median).rename("VV_median_VH_median_nat")
    median_db = vv_median.divide(vh_median).rename("VV_median_VH_median_db")

    linear = collection.select(["VV", "VH"]).map(db_to_linear)
    cv = (
        linear.reduce(ee.Reducer.stdDev())
        .divide(linear.mean())
        .rename(["VV_CV", "VH_CV"])
    )
    vv_cv, vh_cv = cv.select("VV_CV"), cv.select("VH_CV")
    ndcv = vv_cv.subtract(vh_cv).divide(vv_cv.add(vh_cv)).rename("NDCV")

    return (
        stats.addBands(median_nat)
        .addBands(median_db)
        .addBands(cv)
        .addBands(ndcv)
        .select(S1_TIMESCAN_BANDS)
    )


def build_seasonal_s1_timescan(
    collection: ee.ImageCollection,
    year: int,
    season_months: tuple[int, int],
    season_name: str,
) -> ee.Image:
    """Build a Sentinel-1 timescan for one season of one year, with season-suffixed bands.

    Filters the collection to months [season_months[0], season_months[1]] (inclusive)
    of ``year``, runs :func:`build_s1_timescan`, and appends ``_<season_name>`` to
    every band (e.g. ``VV_min_wet``) so wet and dry timescans can be stacked. Pass
    ``eetools.constants.SEASON_WET`` / ``SEASON_DRY`` as ``season_months`` for the
    default East-African seasons.

    Args:
        collection: ee.ImageCollection of Sentinel-1 images with VV and VH backscatter bands in dB (e.g. from get_s1_grd_collection).
        year: Calendar year of the season.
        season_months: (start_month, end_month) as 1-based integers, inclusive; both must fall within the same calendar year (cross-year seasons such as Nov–Jan are not supported).
        season_name: Season label used as the band suffix and stored as the 'season' property (e.g. 'wet', 'dry').

    Returns:
        ee.Image with the 25 S1_TIMESCAN_BANDS suffixed with ``_<season_name>``, carrying system:time_start (season start), year, season, season_months, and image_count properties. Raises ValueError if season_months is not a valid (start, end) pair with 1 ≤ start ≤ end ≤ 12.
    """
    _validate_season_months(season_months)
    start_month, end_month = season_months

    window_start = ee.Date.fromYMD(year, start_month, 1)
    window_end = ee.Date.fromYMD(year, end_month, 1).advance(1, "month")
    seasonal = ee.ImageCollection(collection).filterDate(window_start, window_end)

    timescan = build_s1_timescan(seasonal).rename(
        [f"{band}_{season_name}" for band in S1_TIMESCAN_BANDS]
    )
    return ee.Image(
        timescan.set(
            {
                "system:time_start": window_start.millis(),
                "year": year,
                "season": season_name,
                "season_months": f"{start_month}-{end_month}",
                "image_count": seasonal.size(),
            }
        )
    )
