import ee

from eetools.landtrendr.collection import DIST_DIR

# Default getChangeMap parameters. Magnitudes/values are in NATIVE index units (e.g. an
# NBR delta of 0.2), not the x1000-scaled units of the LandTrendr.js change mapper, since
# this pipeline keeps the segmentation index as float reflectance.
DEFAULT_CHANGE_PARAMS: dict = {
    "delta": "loss",  # 'loss' (disturbance) or 'gain' (recovery)
    "sort": "greatest",  # greatest | least | newest | oldest | fastest | slowest
    "year": {"checked": False, "start": None, "end": None},
    "mag": {"checked": False, "value": 0.0, "operator": ">"},
    "dur": {"checked": False, "value": 4, "operator": "<"},
    "preval": {"checked": False, "value": 0.0, "operator": ">"},
    "mmu": {"checked": True, "value": 11},
}


def get_segment_data(lt: ee.Image, dist_dir: int = DIST_DIR) -> ee.Image:
    """Extract per-segment attributes from a LandTrendr output as an array image.

    Pairs consecutive vertices into segments and returns an 8-row x nSegments array, with
    spectral values/magnitudes re-oriented to their NATURAL index sign (the inverse of the
    loss-positive orientation used for segmentation).

    Rows: 0 start year, 1 end year, 2 start value, 3 end value, 4 magnitude (natural delta,
    end - start), 5 duration (years), 6 rate (delta/yr), 7 DSNR (magnitude / fit RMSE).

    Args:
        lt: LandTrendr output image (must have the 'LandTrendr' band and 'rmse').
        dist_dir: Loss-orientation factor applied to band 1 at input (-1 for NBR/NDVI/NDMI); used to restore natural sign.

    Returns:
        ee.Image whose single band is an 8 x nSegments array of per-segment attributes.
    """
    lt_band = lt.select("LandTrendr")
    rmse = lt.select("rmse")

    vertex_mask = lt_band.arraySlice(0, 3, 4)  # 'is vertex' row
    vertices = lt_band.arrayMask(vertex_mask)

    # Pair each segment's start (left) and end (right) vertex by shifting one column.
    left = vertices.arraySlice(1, 0, -1)
    right = vertices.arraySlice(1, 1, None)

    start_year = left.arraySlice(0, 0, 1)
    start_val = left.arraySlice(0, 2, 3)
    end_year = right.arraySlice(0, 0, 1)
    end_val = right.arraySlice(0, 2, 3)

    duration = end_year.subtract(start_year)
    # Natural-signed values and magnitude (undo the loss-positive orientation).
    start_val_nat = start_val.multiply(dist_dir)
    end_val_nat = end_val.multiply(dist_dir)
    magnitude = end_val.subtract(start_val).multiply(dist_dir)
    rate = magnitude.divide(duration)
    dsnr = magnitude.divide(rmse)

    return ee.Image.cat(
        [
            start_year.add(1),  # year of first detectable change
            end_year,
            start_val_nat,
            end_val_nat,
            magnitude,
            duration,
            rate,
            dsnr,
        ]
    ).toArray(0)


def get_segment_count(segment_data: ee.Image) -> ee.Image:
    """Return the per-pixel number of segments in a get_segment_data array.

    Args:
        segment_data: Array image from get_segment_data (8 x nSegments).

    Returns:
        ee.Image with a single integer 'n_segments' band.
    """
    return segment_data.arrayLength(1).rename("n_segments")


def _row_to_band(segment: ee.Image, row: int, name: str) -> ee.Image:
    """Flatten one attribute row of a single-segment array to a scalar band."""
    return segment.arraySlice(0, row, row + 1).arrayProject([1]).arrayFlatten([[name]])


def _apply_operator(band: ee.Image, spec: dict) -> ee.Image:
    """Build a boolean keep-mask from a {'value','operator'} filter spec."""
    value = spec["value"]
    operator = spec.get("operator", ">")
    if operator == ">":
        return band.gt(value)
    if operator == ">=":
        return band.gte(value)
    if operator == "<":
        return band.lt(value)
    if operator == "<=":
        return band.lte(value)
    if operator == "==":
        return band.eq(value)
    raise ValueError(f"Unsupported operator {operator!r}.")


def get_change_map(
    lt: ee.Image,
    change_params: dict | None = None,
    dist_dir: int = DIST_DIR,
) -> ee.Image:
    """Reduce a LandTrendr output to a single change-event image per pixel.

    Selects, per pixel, the one disturbance ('loss') or recovery ('gain') segment chosen by
    ``sort`` (greatest/least/newest/oldest/fastest/slowest), then emits per-pixel attribute
    bands and applies optional magnitude/duration/pre-value/year filters and a minimum-
    mapping-unit (connected-pixel) cleanup. Magnitudes are absolute, in native index units.

    Args:
        lt: LandTrendr output image.
        change_params: Overrides merged over DEFAULT_CHANGE_PARAMS (delta, sort, year, mag, dur, preval, mmu).
        dist_dir: Loss-orientation factor used by get_segment_data (default DIST_DIR).

    Returns:
        ee.Image with bands 'yod' (year of detection), 'mag', 'dur', 'rate', 'dsnr', 'preval', masked to the filters/MMU.
    """
    params = {**DEFAULT_CHANGE_PARAMS, **(change_params or {})}

    seg = get_segment_data(lt, dist_dir)  # 8 x nSegments, natural-signed

    mag_row = seg.arraySlice(0, 4, 5)
    # Keep only segments of the requested change type (loss = decrease, gain = increase).
    type_mask = mag_row.lt(0) if params["delta"] == "loss" else mag_row.gt(0)
    seg = seg.arrayMask(type_mask)

    # Sort key (arraySort is ascending; negate for "most"-style criteria).
    end_year_row = seg.arraySlice(0, 1, 2)
    rate_row = seg.arraySlice(0, 6, 7)
    sort = params["sort"]
    if sort == "greatest":
        key = mag_row.arrayMask(type_mask).abs().multiply(-1)
    elif sort == "least":
        key = mag_row.arrayMask(type_mask).abs()
    elif sort == "newest":
        key = end_year_row.multiply(-1)
    elif sort == "oldest":
        key = end_year_row
    elif sort == "fastest":
        key = rate_row.abs().multiply(-1)
    elif sort == "slowest":
        key = rate_row.abs()
    else:
        raise ValueError(f"Unsupported sort {sort!r}.")

    selected = seg.arraySort(key).arraySlice(1, 0, 1)  # 8 x 1 (chosen segment)

    yod = _row_to_band(selected, 0, "yod")
    end_yr = _row_to_band(selected, 1, "endYr")
    preval = _row_to_band(selected, 2, "preval")
    mag = _row_to_band(selected, 4, "mag").abs()
    dur = _row_to_band(selected, 5, "dur")
    rate = _row_to_band(selected, 6, "rate").abs()
    dsnr = _row_to_band(selected, 7, "dsnr").abs()

    change = ee.Image.cat([yod, mag, dur, rate, dsnr, preval, end_yr]).select(
        ["yod", "mag", "dur", "rate", "dsnr", "preval"]
    )

    # Optional attribute filters.
    keep = ee.Image.constant(1)
    if params["mag"]["checked"]:
        keep = keep.And(_apply_operator(change.select("mag"), params["mag"]))
    if params["dur"]["checked"]:
        keep = keep.And(_apply_operator(change.select("dur"), params["dur"]))
    if params["preval"]["checked"]:
        keep = keep.And(_apply_operator(change.select("preval"), params["preval"]))
    if params["year"]["checked"]:
        keep = keep.And(change.select("yod").gte(params["year"]["start"])).And(
            change.select("yod").lte(params["year"]["end"])
        )
    change = change.updateMask(keep)

    # Minimum-mapping-unit: drop patches smaller than `value` connected pixels.
    if params["mmu"]["checked"]:
        mmu_pixels = params["mmu"]["value"]
        patch = (
            change.select("yod")
            .toInt()
            .connectedPixelCount(mmu_pixels, True)
            .gte(mmu_pixels)
        )
        change = change.updateMask(patch)

    return change


def get_fitted_stack(
    lt: ee.Image,
    start_year: int,
    end_year: int,
    index: str | None = None,
    dist_dir: int = DIST_DIR,
) -> ee.Image:
    """Flatten a fit-to-vertices series to an annual band stack (one band per year).

    When ``index`` is None the fitted segmentation index is taken from the LandTrendr array
    (its fitted row, re-oriented to natural sign) — no FTV band is required. Otherwise the
    ``<index>_fit`` FTV band is flattened (natural-signed already).

    Args:
        lt: LandTrendr output image.
        start_year: First year in the series (must match the run window).
        end_year: Last year in the series (inclusive).
        index: FTV index name to flatten; None flattens the fitted segmentation index.
        dist_dir: Orientation factor used to restore the segmentation index's natural sign (only used when index is None).

    Returns:
        ee.Image with one band per year ('yr_<year>'), gap-filled along the fitted segments.
    """
    year_names = [f"yr_{year}" for year in range(start_year, end_year + 1)]

    if index is None:
        fitted = lt.select("LandTrendr").arraySlice(0, 2, 3).multiply(dist_dir)
        return fitted.arrayProject([1]).arrayFlatten([year_names])

    return lt.select(f"{index}_fit").arrayFlatten([year_names])
