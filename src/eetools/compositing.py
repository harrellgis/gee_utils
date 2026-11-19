from collections.abc import Sequence
from typing import Any

import ee

from eetools.constants import CLEAR_COUNT_BAND

# --------------------------------------------------------------------------- #
# Private helpers                                                              #
# --------------------------------------------------------------------------- #


def _validate_composite_stat(composite_stat: str) -> None:
    valid = {"mean", "median", "sum"}
    if composite_stat not in valid:
        raise ValueError(f"composite_stat must be one of {sorted(valid)}")


def _apply_stat(collection: ee.ImageCollection, composite_stat: str) -> ee.Image:
    if composite_stat == "mean":
        return collection.mean()
    if composite_stat == "median":
        return collection.median()
    if composite_stat == "sum":
        return collection.sum()
    raise ValueError(f"Unsupported composite_stat: {composite_stat}")


def _composite(
    collection: ee.ImageCollection,
    bands: list[str],
    composite_stat: str,
    add_clear_count: bool,
) -> ee.Image:
    # `collection` is already band-selected. clear_count counts the unmasked
    # observations of the first band per pixel (masked = cloud/shadow/invalid).
    composite = _apply_stat(collection, composite_stat)
    if not add_clear_count:
        return composite
    clear_count = (
        collection.select(bands[0]).count().toUint16().rename(CLEAR_COUNT_BAND)
    )
    return composite.addBands(clear_count)


def _time_windows(
    start_date: ee.Date, end_date: ee.Date, temporal_scale: str
) -> ee.FeatureCollection:
    start_date = ee.Date(start_date)
    end_date = ee.Date(end_date)

    if temporal_scale == "annual":
        start_year = ee.Number.parse(start_date.format("YYYY"))
        end_year = ee.Number.parse(end_date.advance(-1, "day").format("YYYY"))
        offsets = ee.List.sequence(0, end_year.subtract(start_year))

        def make_window(offset: ee.Number) -> ee.Feature:
            offset = ee.Number(offset)
            window_start = ee.Date.fromYMD(start_year.add(offset), 1, 1)
            return ee.Feature(
                None,
                {
                    "window_start": window_start.millis(),
                    "date": window_start.format("YYYY-MM-dd"),
                    "year": window_start.get("year"),
                    "month": None,
                    "day": None,
                },
            )

    elif temporal_scale == "monthly":
        # Count whole calendar months in [start, end) rather than relying on
        # ee.Date.difference(unit="month"), which uses an average month length
        # (~30.44 days) and truncates — dropping the final window whenever the
        # span includes 31-day months. advance(-1, "day") makes the end
        # exclusive, mirroring the annual branch.
        last_day = end_date.advance(-1, "day")
        start_months = (
            ee.Number(start_date.get("year")).multiply(12).add(start_date.get("month"))
        )
        end_months = (
            ee.Number(last_day.get("year")).multiply(12).add(last_day.get("month"))
        )
        n_months = end_months.subtract(start_months).add(1)
        offsets = ee.List.sequence(0, n_months.subtract(1))

        def make_window(offset: ee.Number) -> ee.Feature:
            offset = ee.Number(offset)
            window_start = start_date.advance(offset, "month")
            return ee.Feature(
                None,
                {
                    "window_start": window_start.millis(),
                    "date": window_start.format("YYYY-MM-dd"),
                    "year": window_start.get("year"),
                    "month": window_start.get("month"),
                    "day": None,
                },
            )

    else:
        raise ValueError("temporal_scale must be either 'annual' or 'monthly'")

    return ee.FeatureCollection(offsets.map(make_window))


# --------------------------------------------------------------------------- #
# Compositing                                                                  #
# --------------------------------------------------------------------------- #


def build_composite(
    collection: ee.ImageCollection,
    bands: list[str],
    composite_stat: str = "median",
    add_clear_count: bool = False,
) -> ee.Image:
    """Reduce an ImageCollection to a single composite image.

    Applies the specified statistic across all images in the collection after
    selecting the requested bands. Use this when you have already filtered a
    collection to a date window and need one composite image. For multi-period
    compositing see :func:`build_period_composites` and
    :func:`build_seasonal_composites`.

    Args:
        collection: ee.ImageCollection to composite.
        bands: List of band names to select before compositing.
        composite_stat: Statistic used to combine images; one of 'mean',
            'median', or 'sum' (default 'median').
        add_clear_count: Add a uint16 'clear_count' band: the number of
            unmasked observations of the first band in ``bands`` per pixel
            (cloud-free valid observations when the collection is cloud-masked).

    Returns:
        ee.Image of the composited bands (plus 'clear_count' if requested).

    Raises:
        ValueError: If composite_stat is not 'mean', 'median', or 'sum'.
    """
    _validate_composite_stat(composite_stat)
    return _composite(
        ee.ImageCollection(collection).select(bands),
        bands,
        composite_stat,
        add_clear_count,
    )


PERIOD_BAND_NAME = "{band}_{label}"


def _validate_periods(
    bands: Sequence[str], periods: Sequence[tuple[str, Any, Any]], band_name: str
) -> list[str]:
    if not bands:
        raise ValueError("bands must name at least one band.")
    if not periods:
        raise ValueError("periods must hold at least one (label, start, end).")
    labels = [str(p[0]) for p in periods]
    if len(set(labels)) != len(labels):
        raise ValueError(f"period labels must be unique; got {labels}")
    if "{band}" not in band_name or "{label}" not in band_name:
        raise ValueError("band_name must contain '{band}' and '{label}'.")
    return labels


def period_stack_band_names(
    bands: Sequence[str],
    labels: Sequence[str],
    add_clear_count: bool = False,
    band_name: str = PERIOD_BAND_NAME,
) -> list[str]:
    """The band names :func:`build_period_stack` writes, in order (client-side, no EE).

    Args:
        bands: Composite bands, in order.
        labels: Period labels, in order.
        add_clear_count: Whether each period also has a ``clear_count`` band.
        band_name: Format string with ``{band}`` and ``{label}`` (default ``PERIOD_BAND_NAME``).

    Returns:
        List of output band names: for each period, its bands (then ``clear_count``).
    """
    per = [*bands, CLEAR_COUNT_BAND] if add_clear_count else list(bands)
    return [band_name.format(band=b, label=lab) for lab in labels for b in per]


def build_period_stack(
    collection: ee.ImageCollection,
    bands: Sequence[str],
    periods: Sequence[tuple[str, str | ee.Date, str | ee.Date]],
    composite_stat: str = "median",
    add_clear_count: bool = False,
    band_name: str = PERIOD_BAND_NAME,
) -> ee.Image:
    """Composite each named period and stack the results into one multi-band image,
    keeping every period, including empty ones.

    Use this when the band list must be fixed by the arguments alone, e.g. for
    export stacks with manifests or parameter hashes. :func:`build_period_composites`
    and :func:`build_seasonal_composites` drop empty periods instead. Here an empty
    period, or a pixel with no clear observation, gives masked composite bands and
    ``clear_count`` 0.

    Bands are cast to float before compositing. Index bands carry typed ranges
    (e.g. ``Float<-1, 1>``), and Earth Engine refuses to merge them with the
    placeholder that keeps empty periods ("Expected a homogeneous image collection").
    The cast leaves the values unchanged.

    Args:
        collection: ee.ImageCollection to composite (e.g. a cloud-masked Sentinel-2 collection).
        bands: Bands to composite, in output order; ``clear_count`` counts the first band's unmasked observations.
        periods: (label, start, end) per period, end exclusive; dates as 'YYYY-MM-DD' strings or ee.Date. Labels must be unique and are written into the band names.
        composite_stat: 'mean', 'median' (default) or 'sum'.
        add_clear_count: Add a uint16 ``clear_count`` band per period (0 where no clear observation).
        band_name: Format string for output band names, with ``{band}`` and ``{label}`` (default ``PERIOD_BAND_NAME``, ``{band}_{label}``); ``clear_count`` is named the same way.

    Returns:
        ee.Image with the bands of :func:`period_stack_band_names`, period by period. Composite bands are float; ``clear_count`` bands are uint16.

    Raises:
        ValueError: If composite_stat is invalid, bands or periods are empty, period labels repeat, or band_name lacks a placeholder.
    """
    _validate_composite_stat(composite_stat)
    bands = list(bands)
    labels = _validate_periods(bands, periods, band_name)

    col = ee.ImageCollection(collection).select(bands)
    empty = ee.Image.constant([0] * len(bands)).rename(bands).toFloat()
    empty_col = ee.ImageCollection([empty.updateMask(ee.Image(0))])

    parts = []
    for label, (_, start, end) in zip(labels, periods, strict=True):
        sub = col.filterDate(start, end).map(lambda i: ee.Image(i).toFloat())
        merged = sub.merge(empty_col)
        comp = _apply_stat(merged, composite_stat)
        if add_clear_count:
            count = (
                merged.select(bands[0])
                .count()
                .unmask(0)
                .toUint16()
                .rename(CLEAR_COUNT_BAND)
            )
            comp = comp.addBands(count)
        names = period_stack_band_names(bands, [label], add_clear_count, band_name)
        per = [*bands, CLEAR_COUNT_BAND] if add_clear_count else bands
        parts.append(comp.select(per).rename(names))
    return ee.Image.cat(parts)


def build_period_composites(
    collection: ee.ImageCollection,
    bands: list[str],
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    temporal_scale: str = "annual",
    composite_stat: str = "median",
    add_clear_count: bool = False,
) -> ee.ImageCollection:
    """Build annual or monthly composites from an ImageCollection, producing one
    composite per period with data.

    Args:
        collection: ee.ImageCollection to composite.
        bands: List of band names to select before compositing.
        start_date: Compositing start date as a string or ee.Date.
        end_date: Compositing end date as a string or ee.Date.
        temporal_scale: Temporal aggregation window; either 'annual' or 'monthly' (default 'annual').
        composite_stat: Statistic used to combine images within each window; one of 'mean', 'median', or 'sum' (default 'median').
        add_clear_count: Add a uint16 'clear_count' band per composite: the number of unmasked observations of the first band in ``bands`` per pixel in that period (image_count only counts scenes).

    Returns:
        ee.ImageCollection of composite images, one per period with data (empty periods are dropped), each with date, year, month, day, image_count, temporal_scale, and composite_stat properties.
    """
    _validate_composite_stat(composite_stat)

    collection = ee.ImageCollection(collection).select(bands)
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    windows = _time_windows(start, end, temporal_scale)

    def make_composite(feature: ee.Feature) -> ee.Image:
        feature = ee.Feature(feature)
        window_start = ee.Date(feature.get("window_start"))
        window_end = (
            window_start.advance(1, "year")
            if temporal_scale == "annual"
            else window_start.advance(1, "month")
        )
        subset = collection.filterDate(window_start, window_end)
        image_count = subset.size()
        composite = _composite(subset, bands, composite_stat, add_clear_count)

        return ee.Image(
            ee.Algorithms.If(
                image_count.gt(0),
                composite.set(
                    {
                        "system:time_start": window_start.millis(),
                        "date": feature.get("date"),
                        "year": feature.get("year"),
                        "month": feature.get("month"),
                        "day": feature.get("day"),
                        "image_count": image_count,
                        "temporal_scale": temporal_scale,
                        "composite_stat": composite_stat,
                    }
                ),
                None,
            )
        )

    images_list = windows.toList(windows.size()).map(make_composite)
    return ee.ImageCollection.fromImages(images_list)


def build_seasonal_composites(
    collection: ee.ImageCollection,
    bands: list[str],
    start_year: int,
    end_year: int,
    season_months: tuple[int, int],
    season_name: str,
    composite_stat: str = "median",
    add_clear_count: bool = False,
) -> ee.ImageCollection:
    """Build per-year seasonal composites by filtering each year's collection to a
    fixed window of consecutive months.

    For each year in [start_year, end_year] the collection is filtered to the
    months [season_months[0], season_months[1]] (inclusive) and reduced with
    composite_stat. Years that contain no images in the season window are
    excluded from the output (the ImageCollection contains no null entries).

    Args:
        collection: ee.ImageCollection to composite.
        bands: List of band names to select before compositing.
        start_year: First year to process, inclusive.
        end_year: Last year to process, inclusive.
        season_months: (start_month, end_month) as 1-based integers (e.g. (3, 5)
            for March through May, inclusive).  Both months must fall within the
            same calendar year; cross-year seasons (e.g. Nov–Jan) are not
            supported.
        season_name: Label stored as the 'season' property on each output image
            (e.g. 'wet', 'dry').
        composite_stat: Statistic used to combine images; one of 'mean',
            'median', or 'sum' (default 'median').
        add_clear_count: Add a uint16 'clear_count' band per composite: the
            number of unmasked observations of the first band in ``bands`` per
            pixel in that season window.

    Returns:
        ee.ImageCollection of per-year composites, one per year with imagery in
        the season window, each carrying system:time_start, year, season,
        season_months, image_count, and composite_stat properties.

    Raises:
        ValueError: If composite_stat is not 'mean', 'median', or 'sum', or if
            season_months is not a valid (start, end) pair with 1 ≤ start ≤ end ≤ 12.
    """
    _validate_composite_stat(composite_stat)
    start_month, end_month = season_months
    if not (1 <= start_month <= end_month <= 12):
        raise ValueError(
            f"season_months must satisfy 1 <= start <= end <= 12; got {season_months}"
        )

    season_label = f"{start_month}-{end_month}"
    col = ee.ImageCollection(collection).select(bands)
    years = ee.List.sequence(start_year, end_year)

    def _make_composite(year: ee.Number) -> ee.Image:
        year = ee.Number(year)
        window_start = ee.Date.fromYMD(year, start_month, 1)
        window_end = ee.Date.fromYMD(year, end_month, 1).advance(1, "month")
        subset = col.filterDate(window_start, window_end)
        image_count = subset.size()
        composite = _composite(subset, bands, composite_stat, add_clear_count)
        return ee.Image(
            ee.Algorithms.If(
                image_count.gt(0),
                composite.set(
                    {
                        "system:time_start": window_start.millis(),
                        "year": year,
                        "season": season_name,
                        "season_months": season_label,
                        "image_count": image_count,
                        "composite_stat": composite_stat,
                    }
                ),
                None,
            )
        )

    return ee.ImageCollection.fromImages(years.map(_make_composite))
