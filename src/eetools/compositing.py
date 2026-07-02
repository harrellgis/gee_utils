import ee

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

    Returns:
        ee.Image of the composited bands.

    Raises:
        ValueError: If composite_stat is not 'mean', 'median', or 'sum'.
    """
    _validate_composite_stat(composite_stat)
    return _apply_stat(ee.ImageCollection(collection).select(bands), composite_stat)


def build_period_composites(
    collection: ee.ImageCollection,
    bands: list[str],
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    temporal_scale: str = "annual",
    composite_stat: str = "median",
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

    Returns:
        ee.ImageCollection of composite images, one per period, each with date, year, month, day, image_count, temporal_scale, and composite_stat properties.
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
        composite = _apply_stat(subset, composite_stat)

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
        composite = _apply_stat(subset, composite_stat)
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
