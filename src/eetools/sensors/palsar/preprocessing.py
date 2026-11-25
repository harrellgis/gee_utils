"""ALOS-2 PALSAR-2 L-band backscatter as gamma-naught.

Two sources:

* the yearly 25 m mosaics (``get_palsar2_yearly``, 2015-2024 in EE);
* the per-scene ScanSAR Level 2.2 collection (``get_palsar2_scansar``), composited
  per year over a fixed season with ``palsar2_scansar_composite``. It reaches past
  the mosaic's last year.

Average backscatter in linear power, never in dB.
"""

from collections.abc import Sequence

import ee

from eetools.constants import (
    PALSAR2_CALIBRATION_DB,
    PALSAR2_POL_BANDS,
    PALSAR2_QA_BAND,
    PALSAR2_QA_LAND,
    PALSAR2_SCALE_M,
    PALSAR2_SCANSAR_COLLECTION,
    PALSAR2_SCANSAR_COUNT_BAND,
    PALSAR2_SCANSAR_DEFAULT_POLS,
    PALSAR2_SCANSAR_LIN_BAND,
    PALSAR2_SCANSAR_LIN_MEAN_BAND,
    PALSAR2_SCANSAR_LIN_SCALE,
    PALSAR2_SCANSAR_POL_PROPERTY,
    PALSAR2_SCANSAR_POLARIZATIONS,
    PALSAR2_SCANSAR_QA_BAND,
    PALSAR2_SCANSAR_QA_BITMASK,
    PALSAR2_SCANSAR_QA_VALID,
    PALSAR2_SCANSAR_REDUCERS,
    PALSAR2_YEARLY_COLLECTION,
)
from eetools.utils import validate_collection_date_range, validate_collection_year_range

_UNIT_PROPERTY = "backscatter_unit"


def _dn_to_db(dn: ee.Image) -> ee.Image:
    """gamma0 dB = 20 * log10(DN) + PALSAR2_CALIBRATION_DB, with DN 0 (no data) masked."""
    return (
        dn.toFloat()
        .log10()
        .multiply(20)
        .add(PALSAR2_CALIBRATION_DB)
        .updateMask(dn.gt(0))
    )


def _db_to_linear(db: ee.Image) -> ee.Image:
    """Linear power = 10^(dB / 10), keeping the band names of ``db``.

    ``ee.Image(10).pow(x)`` names its output after the first operand when both have
    one band, so a single-polarisation input would come back as ``constant``.
    """
    return ee.Image(10).pow(db.divide(10)).rename(db.bandNames())


def _linear_to_db(linear: ee.Image) -> ee.Image:
    """dB = 10 * log10(linear power)."""
    return linear.log10().multiply(10)


def palsar2_dn_to_db(image: ee.Image) -> ee.Image:
    """Convert PALSAR-2 HH/HV digital numbers to gamma-naught dB, land pixels only.

    gamma0 dB = 10 * log10(DN^2) - 83.0 (JAXA calibration), computed as
    20 * log10(DN) - 83.0. DN 0 (no data) and every pixel whose ``qa`` is not land
    (water, layover, radar shadow) are masked.

    Args:
        image: PALSAR-2 yearly ee.Image with bands HH, HV and qa.

    Returns:
        ee.Image with float bands ``HH`` and ``HV`` in dB, properties preserved.
    """
    image = ee.Image(image)
    db = (
        _dn_to_db(image.select(PALSAR2_POL_BANDS))
        .updateMask(image.select(PALSAR2_QA_BAND).eq(PALSAR2_QA_LAND))
        .rename(PALSAR2_POL_BANDS)
    )
    return ee.Image(db.copyProperties(image, image.propertyNames()))


def get_palsar2_yearly(
    aoi: ee.Geometry, start_year: int, end_year: int
) -> ee.ImageCollection:
    """PALSAR-2 yearly 25 m mosaics (HH, HV gamma-naught dB), one image per year.

    Each year's tiles over the AOI are converted with ``palsar2_dn_to_db`` and
    mosaicked, with the default projection pinned to the source tiles (a bare
    ``mosaic()`` has none, which breaks ``reduceResolution``). Average backscatter in
    linear power, not dB, when aggregating. Record: 2015-2024.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_year: First year (inclusive).
        end_year: Last year (inclusive).

    Returns:
        ee.ImageCollection of float HH/HV dB images with ``year`` and ``system:time_start`` (1 January).

    Raises:
        ValueError: If end_year < start_year or the years fall outside the record over the AOI.
    """
    validate_collection_year_range(
        PALSAR2_YEARLY_COLLECTION, aoi, start_year, end_year, "PALSAR-2 yearly mosaic"
    )
    source = ee.ImageCollection(PALSAR2_YEARLY_COLLECTION).filterBounds(aoi)
    images = []
    for year in range(start_year, end_year + 1):
        tiles = source.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        projection = ee.Image(tiles.first()).select(PALSAR2_POL_BANDS[0]).projection()
        images.append(
            tiles.map(palsar2_dn_to_db)
            .mosaic()
            .setDefaultProjection(projection)
            .set(
                {
                    "year": year,
                    "system:time_start": ee.Date.fromYMD(year, 1, 1).millis(),
                }
            )
        )
    return ee.ImageCollection(images)


def _check_polarizations(polarizations: Sequence[str]) -> list[str]:
    """Return ``polarizations`` as a list, or raise ValueError if empty or unknown."""
    pols = list(polarizations)
    if not pols:
        raise ValueError("polarizations must name at least one polarisation.")
    unknown = [p for p in pols if p not in PALSAR2_SCANSAR_POLARIZATIONS]
    if unknown:
        raise ValueError(
            f"Unknown PALSAR-2 polarisation(s) {unknown}. Expected a subset of "
            f"{list(PALSAR2_SCANSAR_POLARIZATIONS)}."
        )
    if len(set(pols)) != len(pols):
        raise ValueError(f"Duplicate polarisations in {pols}.")
    return pols


def palsar2_scansar_to_gamma0(
    image: ee.Image,
    polarizations: Sequence[str] = PALSAR2_SCANSAR_DEFAULT_POLS,
    *,
    mask_quality: bool = True,
    linear: bool = False,
) -> ee.Image:
    """Convert one PALSAR-2 ScanSAR L2.2 scene from DN to gamma-naught.

    gamma0 dB = 10 * log10(DN^2) - 83.0. DN 0 (no data) is masked in each
    polarisation band. With ``mask_quality``, every band is masked except where
    ``MSK`` bits 0-2 == 1 (valid). That drops layover (2), shadow (3), ocean (4) and
    invalid (5) pixels.

    Args:
        image: ScanSAR L2.2 ee.Image with the requested polarisation bands, ``LIN`` and ``MSK``.
        polarizations: Polarisation bands to convert (default ('HH', 'HV')).
        mask_quality: Mask every non-valid ``MSK`` pixel (default True).
        linear: Return linear power instead of dB (default False).

    Returns:
        ee.Image with float polarisation bands (dB, or linear power when ``linear``) and float ``LIN`` (local incidence angle, degrees), all source properties kept, plus ``backscatter_unit`` ('dB' or 'linear').

    Raises:
        ValueError: If ``polarizations`` is empty, repeats a band or names an unknown polarisation.
    """
    pols = _check_polarizations(polarizations)
    image = ee.Image(image)
    backscatter = _dn_to_db(image.select(pols))
    if linear:
        backscatter = _db_to_linear(backscatter)
    lin = (
        image.select(PALSAR2_SCANSAR_LIN_BAND)
        .toFloat()
        .multiply(PALSAR2_SCANSAR_LIN_SCALE)
        .rename(PALSAR2_SCANSAR_LIN_BAND)
    )
    out = backscatter.rename(pols).addBands(lin)
    if mask_quality:
        valid = (
            image.select(PALSAR2_SCANSAR_QA_BAND)
            .bitwiseAnd(PALSAR2_SCANSAR_QA_BITMASK)
            .eq(PALSAR2_SCANSAR_QA_VALID)
        )
        out = out.updateMask(valid)
    return ee.Image(
        out.copyProperties(image, image.propertyNames()).set(
            _UNIT_PROPERTY, "linear" if linear else "dB"
        )
    )


def get_palsar2_scansar(
    aoi: ee.Geometry,
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    *,
    polarizations: Sequence[str] = PALSAR2_SCANSAR_DEFAULT_POLS,
    mask_quality: bool = True,
    linear: bool = False,
) -> ee.ImageCollection:
    """PALSAR-2 ScanSAR L2.2 scenes (25 m), terrain-flattened gamma-naught.

    Keeps only scenes whose ``Polarizations`` property contains every requested
    polarisation; the collection also holds single-pol HH scenes. Each scene is
    converted with ``palsar2_scansar_to_gamma0``. Scenes keep their own UTM
    projection, and ``system:time_start`` is preserved.

    Use the same season every year (``palsar2_scansar_composite``). ScanSAR scenes
    mix incidence angles, look directions and soil-moisture states, so comparing
    different seasons confounds structure with moisture.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Start date (inclusive), 'YYYY-MM-DD' or ee.Date.
        end_date: End date (exclusive), 'YYYY-MM-DD' or ee.Date.
        polarizations: Polarisations every kept scene must have (default ('HH', 'HV')); they become the output bands.
        mask_quality: Mask every pixel whose ``MSK`` class is not valid (default True).
        linear: Return linear power instead of dB (default False).

    Returns:
        ee.ImageCollection of float polarisation bands plus ``LIN`` (degrees), with ``backscatter_unit`` set on each image.

    Raises:
        ValueError: If ``polarizations`` is invalid, the date range is reversed, or it falls outside the ScanSAR record over the AOI.
    """
    pols = _check_polarizations(polarizations)
    start, end = ee.Date(start_date), ee.Date(end_date)
    validate_collection_date_range(
        collection_id=PALSAR2_SCANSAR_COLLECTION,
        aoi=aoi,
        start_date=start,
        end_date=end,
        sensor_label="PALSAR-2 ScanSAR L2.2",
    )
    col = (
        ee.ImageCollection(PALSAR2_SCANSAR_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start, end)
    )
    for pol in pols:
        col = col.filter(ee.Filter.listContains(PALSAR2_SCANSAR_POL_PROPERTY, pol))
    return col.map(
        lambda img: palsar2_scansar_to_gamma0(
            img, pols, mask_quality=mask_quality, linear=linear
        )
    )


def _check_months(months: tuple[int, int]) -> tuple[int, int]:
    """Validate an inclusive (start_month, end_month) window inside one year."""
    if len(months) != 2:
        raise ValueError(f"months must be (start_month, end_month), got {months}.")
    m0, m1 = int(months[0]), int(months[1])
    if not (1 <= m0 <= 12 and 1 <= m1 <= 12):
        raise ValueError(f"months must be in 1-12, got {months}.")
    if m0 > m1:
        raise ValueError(
            f"months {months} crosses the year end. Seasons stay inside one "
            "calendar year; split a cross-year window into two composites."
        )
    return m0, m1


def palsar2_scansar_composite(
    collection: ee.ImageCollection,
    year: int,
    months: tuple[int, int],
    *,
    polarizations: Sequence[str] = PALSAR2_SCANSAR_DEFAULT_POLS,
    reducer: str = "median",
) -> ee.Image:
    """One seasonal ScanSAR composite: scenes of ``months`` in ``year``.

    Each polarisation is reduced per pixel in **linear power** and converted back to
    dB, so two scenes at -10 and -20 dB give about -12.6 dB (median or mean of two),
    not -15 dB. Seasons never cross the year end.

    ``scene_count`` is the number of valid scenes per pixel, the analogue of S2's
    ``clear_count``. It is 0 where no scene falls in the window, and those pixels
    stay masked in the backscatter bands. Where adjacent frames of one pass overlap,
    the seam pixels count that acquisition twice. ``lin_mean`` is the mean local
    incidence angle of the scenes used. The default projection is pinned to the
    first scene, so ``reduceResolution`` works, as in ``get_palsar2_yearly``.

    Args:
        collection: Output of ``get_palsar2_scansar``, in dB or linear (read from each image's ``backscatter_unit``; anything other than 'linear' is treated as dB).
        year: Composite year.
        months: Inclusive (start_month, end_month) within ``year``, e.g. (6, 9) for June-September.
        polarizations: Polarisation bands to composite (default ('HH', 'HV')); must be bands of ``collection``.
        reducer: 'median' (default) or 'mean'.

    Returns:
        ee.Image with float dB polarisation bands, ``lin_mean`` (degrees) and int ``scene_count``. Properties: ``year``, ``start_month``, ``end_month``, ``reducer``, ``n_scenes`` (scenes in the window), ``backscatter_unit`` ('dB') and ``system:time_start`` (first day of ``start_month``).

    Raises:
        ValueError: If ``months`` is invalid or crosses the year end, ``reducer`` is unknown, or ``polarizations`` is invalid.
    """
    pols = _check_polarizations(polarizations)
    m0, m1 = _check_months(months)
    if reducer not in PALSAR2_SCANSAR_REDUCERS:
        raise ValueError(
            f"Unknown reducer '{reducer}'. Expected one of "
            f"{list(PALSAR2_SCANSAR_REDUCERS)}."
        )
    ee_reducer = ee.Reducer.median() if reducer == "median" else ee.Reducer.mean()

    window = (
        ee.ImageCollection(collection)
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .filter(ee.Filter.calendarRange(m0, m1, "month"))
    )

    def _to_linear(img: ee.Image) -> ee.Image:
        img = ee.Image(img)
        bands = img.select(pols)
        is_linear = ee.Algorithms.IsEqual(img.get(_UNIT_PROPERTY), "linear")
        return ee.Image(ee.Algorithms.If(is_linear, bands, _db_to_linear(bands)))

    # One fully masked placeholder per collection keeps every reduction's bands even
    # when the window holds no scene (an empty collection reduces to a band-less
    # image), without ee.Algorithms.If: an If makes EE evaluate the scene filter just
    # to know the band names, which ran out of user memory for a stack of yearly
    # composites over a few hundred scenes. Masked pixels are ignored by every
    # reducer, so the placeholder changes no value; an empty window gives masked
    # backscatter and scene_count 0.
    blank = ee.Image.constant([0.0] * len(pols)).rename(pols).toFloat().updateMask(0)
    blank_lin = (
        ee.Image.constant(0.0).rename(PALSAR2_SCANSAR_LIN_BAND).toFloat().updateMask(0)
    )
    fallback = ee.Projection("EPSG:4326").atScale(PALSAR2_SCALE_M)
    linear = window.map(_to_linear).merge(
        ee.ImageCollection([blank.setDefaultProjection(fallback)])
    )
    backscatter = _linear_to_db(linear.reduce(ee_reducer).rename(pols))
    lin_mean = (
        window.select(PALSAR2_SCANSAR_LIN_BAND)
        .merge(ee.ImageCollection([blank_lin]))
        .mean()
    )
    count = linear.select(pols[0]).count().unmask(0)
    # The first real scene when there is one (merge appends the placeholder last).
    projection = ee.Image(linear.first()).select(pols[0]).projection()
    image = (
        backscatter.addBands(lin_mean.rename(PALSAR2_SCANSAR_LIN_MEAN_BAND))
        .addBands(count.toInt16().rename(PALSAR2_SCANSAR_COUNT_BAND))
        .setDefaultProjection(projection)
    )
    n_scenes = window.size()
    return ee.Image(
        image.set(
            {
                "year": year,
                "start_month": m0,
                "end_month": m1,
                "reducer": reducer,
                "n_scenes": n_scenes,
                _UNIT_PROPERTY: "dB",
                "system:time_start": ee.Date.fromYMD(year, m0, 1).millis(),
            }
        )
    )
