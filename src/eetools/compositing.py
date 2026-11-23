from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, cast

import ee

from eetools.constants import (
    CLEAR_COUNT_BAND,
    COMPOSITE_QA_LAYERS,
    LINEAGE_MAX_SCENES,
    LINEAGE_PROPERTIES,
    OBS_DOY_BAND,
    OBS_DOY_BANDS,
)

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


def _resolve_qa_layers(add_clear_count: bool, qa_layers: Sequence[str]) -> list[str]:
    """QA layers in output order; ``add_clear_count`` is the same as ``"clear_count"``."""
    if isinstance(qa_layers, str):
        raise ValueError("qa_layers must be a sequence of names, e.g. ('doy',).")
    unknown = sorted(set(qa_layers) - set(COMPOSITE_QA_LAYERS))
    if unknown:
        raise ValueError(
            f"unknown qa_layers {unknown}; choose from {list(COMPOSITE_QA_LAYERS)}"
        )
    wanted = set(qa_layers) | ({"clear_count"} if add_clear_count else set())
    return [layer for layer in COMPOSITE_QA_LAYERS if layer in wanted]


def _qa_band_names(layers: Sequence[str]) -> list[str]:
    out: list[str] = []
    for layer in layers:
        out += [CLEAR_COUNT_BAND] if layer == "clear_count" else list(OBS_DOY_BANDS)
    return out


def period_stack_band_names(
    bands: Sequence[str],
    labels: Sequence[str],
    add_clear_count: bool = False,
    band_name: str = PERIOD_BAND_NAME,
    qa_layers: Sequence[str] = (),
) -> list[str]:
    """The band names :func:`build_period_stack` writes, in order (client-side, no EE).

    Args:
        bands: Composite bands, in order.
        labels: Period labels, in order.
        add_clear_count: Whether each period also has a ``clear_count`` band (same as ``"clear_count"`` in ``qa_layers``).
        band_name: Format string with ``{band}`` and ``{label}`` (default ``PERIOD_BAND_NAME``).
        qa_layers: Per-pixel QA layers, any of ``COMPOSITE_QA_LAYERS`` (``"clear_count"``, ``"doy"``).

    Returns:
        List of output band names: for each period, its bands, then ``clear_count``, then ``obs_doy_median``, ``obs_doy_min``, ``obs_doy_max`` (each when requested).

    Raises:
        ValueError: If ``qa_layers`` names an unknown layer.
    """
    layers = _resolve_qa_layers(add_clear_count, qa_layers)
    per = [*bands, *_qa_band_names(layers)]
    return [band_name.format(band=b, label=lab) for lab in labels for b in per]


def _add_obs_doy(image: ee.Image, first_band: str, start: Any) -> ee.Image:
    """Add an ``obs_doy`` band: the image's day counted from 1 January of the period's
    start year (1-based), masked where ``first_band`` is masked (no clear observation).
    """
    image = ee.Image(image)
    jan1 = ee.Date.fromYMD(ee.Date(start).get("year"), 1, 1)
    day = ee.Date(image.get("system:time_start")).difference(jan1, "day").floor().add(1)
    doy = (
        ee.Image.constant(day)
        .toFloat()
        .rename(OBS_DOY_BAND)
        .updateMask(image.select(first_band).mask())
    )
    return image.addBands(doy)


def build_period_stack(
    collection: ee.ImageCollection,
    bands: Sequence[str],
    periods: Sequence[tuple[str, str | ee.Date, str | ee.Date]],
    composite_stat: str = "median",
    add_clear_count: bool = False,
    band_name: str = PERIOD_BAND_NAME,
    qa_layers: Sequence[str] = (),
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

    QA layers (``qa_layers``, opt-in; the defaults leave the band list unchanged):

    - ``"clear_count"``: a uint16 band with the number of clear (unmasked) observations
      of the first band, 0 where there is none. Same as ``add_clear_count=True``.
    - ``"doy"``: ``obs_doy_median``, ``obs_doy_min`` and ``obs_doy_max``, the median,
      earliest and latest day of the clear observations of the first band, counted from
      1 January of the period's start year (1 = 1 January; a window that crosses a year
      end continues past 365/366). Masked where there is no clear observation. They
      describe when the composite's inputs were observed: with a median composite each
      band's median can come from a different scene, so they are not "the date of the
      median value". Each image needs ``system:time_start``. Cost: about one extra
      composited band (scene loading and masking, the main cost, is shared).

    Args:
        collection: ee.ImageCollection to composite (e.g. a cloud-masked Sentinel-2 collection).
        bands: Bands to composite, in output order; the QA layers are computed on the first band's mask (its clear observations).
        periods: (label, start, end) per period, end exclusive; dates as 'YYYY-MM-DD' strings or ee.Date. Labels must be unique and are written into the band names.
        composite_stat: 'mean', 'median' (default) or 'sum'.
        add_clear_count: Add a uint16 ``clear_count`` band per period (0 where no clear observation); same as ``"clear_count"`` in ``qa_layers``.
        band_name: Format string for output band names, with ``{band}`` and ``{label}`` (default ``PERIOD_BAND_NAME``, ``{band}_{label}``); QA bands are named the same way.
        qa_layers: Per-pixel QA layers to add, any of ``COMPOSITE_QA_LAYERS`` (``"clear_count"``, ``"doy"``); default none.

    Returns:
        ee.Image with the bands of :func:`period_stack_band_names`, period by period. Composite and ``obs_doy_*`` bands are float; ``clear_count`` bands are uint16.

    Raises:
        ValueError: If composite_stat is invalid, bands or periods are empty, period labels repeat, band_name lacks a placeholder, or qa_layers names an unknown layer.
    """
    _validate_composite_stat(composite_stat)
    bands = list(bands)
    labels = _validate_periods(bands, periods, band_name)
    layers = _resolve_qa_layers(add_clear_count, qa_layers)
    doy = "doy" in layers

    col = ee.ImageCollection(collection).select(bands)
    placeholder = [*bands, OBS_DOY_BAND] if doy else bands
    empty = ee.Image.constant([0] * len(placeholder)).rename(placeholder).toFloat()
    empty_col = ee.ImageCollection([empty.updateMask(ee.Image(0))])
    doy_reducer = (
        ee.Reducer.median()
        .combine(ee.Reducer.min(), sharedInputs=True)
        .combine(ee.Reducer.max(), sharedInputs=True)
    )

    parts = []
    for label, (_, start, end) in zip(labels, periods, strict=True):
        sub = col.filterDate(start, end).map(lambda i: ee.Image(i).toFloat())
        if doy:
            sub = sub.map(lambda i, s=start: _add_obs_doy(i, bands[0], s))
        merged = sub.merge(empty_col)
        comp = _apply_stat(merged.select(bands), composite_stat)
        if "clear_count" in layers:
            count = (
                merged.select(bands[0])
                .count()
                .unmask(0)
                .toUint16()
                .rename(CLEAR_COUNT_BAND)
            )
            comp = comp.addBands(count)
        if doy:
            stats = merged.select(OBS_DOY_BAND).reduce(doy_reducer)
            comp = comp.addBands(stats.rename(list(OBS_DOY_BANDS)))
        names = period_stack_band_names(
            bands, [label], band_name=band_name, qa_layers=layers
        )
        per = [*bands, *_qa_band_names(layers)]
        parts.append(comp.select(per).rename(names))
    return ee.Image.cat(parts)


def _iso_from_millis(value: Any) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    dt = datetime.fromtimestamp(value / 1000, tz=timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


def collection_lineage(
    collection: ee.ImageCollection,
    periods: Sequence[tuple[str, str | ee.Date, str | ee.Date]],
    *,
    region: ee.Geometry | None = None,
    properties: Sequence[str] = LINEAGE_PROPERTIES,
    max_scenes: int = LINEAGE_MAX_SCENES,
) -> dict[str, dict[str, Any]]:
    """The scenes in each period of a collection, as plain Python (one ``getInfo``).

    Scene lineage for a period stack: which images fall in each ``(label, start, end)``
    period (and touch ``region``), with the requested properties. Metadata only: no
    pixels are computed, so it costs essentially no EECU. Pass the result to
    ``io.export_composites_per_site(lineage=...)`` to record it in the manifests (not
    hashed), which makes a changed scene set (e.g. an archive reprocessing) visible.

    It lists scene-level membership after any scene filter already applied to
    ``collection`` (e.g. a cloud-cover filter), not per-pixel use: a listed scene can
    be fully cloud-masked over a pixel. Per-pixel use is ``build_period_stack``'s
    ``qa_layers`` (``clear_count``, ``doy``).

    Args:
        collection: ee.ImageCollection, normally the one passed to :func:`build_period_stack`.
        periods: (label, start, end) per period, end exclusive, as for :func:`build_period_stack`.
        region: Optional ee.Geometry; only scenes intersecting it are listed (e.g. one export window).
        properties: Image properties to record per scene (default ``LINEAGE_PROPERTIES``: ``system:index``, ``system:time_start``); missing ones are None.
        max_scenes: Maximum scenes listed per period (default ``LINEAGE_MAX_SCENES``).

    Returns:
        ``{label: {"start", "end", "n_scenes", "scenes": [{property: value, ..., "datetime": ISO-8601 UTC}]}}``, scenes sorted by time. ``start``/``end`` are ``YYYY-MM-DD`` (end exclusive).

    Raises:
        ValueError: If periods are empty or labels repeat, properties is empty, or a period holds more than ``max_scenes`` scenes.
    """
    labels = _validate_periods(["_"], periods, PERIOD_BAND_NAME)
    props = list(properties)
    if not props:
        raise ValueError("properties must name at least one image property.")
    if "system:time_start" not in props:
        props.append("system:time_start")
    col = ee.ImageCollection(collection)
    if region is not None:
        col = col.filterBounds(region)

    # Feature.toDictionary() leaves out "system:" properties, so carry every property
    # under a neutral key (p0, p1, ...) and map it back client-side.
    keys = {f"p{i}": p for i, p in enumerate(props)}

    def _row(image: ee.Image) -> ee.Feature:
        image = ee.Image(image)
        return ee.Feature(None, {k: image.get(p) for k, p in keys.items()})

    per: dict[str, Any] = {}
    for label, (_, start, end) in zip(labels, periods, strict=True):
        sub = col.filterDate(start, end).sort("system:time_start")
        rows = ee.FeatureCollection(sub.map(_row)).toList(max_scenes + 1)
        per[label] = ee.Dictionary(
            {
                "start": ee.Date(start).format("YYYY-MM-dd"),
                "end": ee.Date(end).format("YYYY-MM-dd"),
                "n_scenes": sub.size(),
                "scenes": rows.map(lambda f: ee.Feature(f).toDictionary()),
            }
        )
    info = cast(dict, ee.Dictionary(per).getInfo())

    out: dict[str, dict[str, Any]] = {}
    for label in labels:
        rec = info[label]
        if int(rec["n_scenes"]) > max_scenes:
            raise ValueError(
                f"period {label!r} holds {rec['n_scenes']} scenes, more than "
                f"max_scenes={max_scenes}; narrow the region or raise max_scenes."
            )
        scenes = []
        for raw in rec["scenes"]:
            scene = {p: raw.get(k) for k, p in keys.items()}
            scene["datetime"] = _iso_from_millis(scene.get("system:time_start"))
            scenes.append(scene)
        out[label] = {
            "start": rec["start"],
            "end": rec["end"],
            "n_scenes": int(rec["n_scenes"]),
            "scenes": scenes,
        }
    return out


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
