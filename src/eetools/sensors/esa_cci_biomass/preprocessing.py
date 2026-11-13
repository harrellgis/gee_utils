"""ESA CCI Biomass v6 above-ground biomass (100 m) and its published change pairs."""

from collections.abc import Sequence

import ee

from eetools.constants import (
    ESA_CCI_AGB_BANDS,
    ESA_CCI_AGB_CHANGE_BANDS,
    ESA_CCI_AGB_COLLECTION,
    ESA_CCI_AGB_DIFF_COLLECTION,
    ESA_CCI_AGB_DIFF_NAME,
    ESA_CCI_AGB_DIFF_PAIRS,
    ESA_CCI_AGB_YEARS,
)


def _check_years(years: Sequence[int]) -> list[int]:
    years = list(years)
    if not years:
        raise ValueError("years must name at least one year.")
    missing = sorted(set(years) - set(ESA_CCI_AGB_YEARS))
    if missing:
        raise ValueError(
            f"ESA CCI AGB v6 has no maps for {missing}; available years are "
            f"{list(ESA_CCI_AGB_YEARS)}."
        )
    return years


def _agb_year(aoi: ee.Geometry, year: int) -> ee.Image:
    image = ee.Image(
        ee.ImageCollection(ESA_CCI_AGB_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(aoi)
        .first()
    )
    return (
        image.select(ESA_CCI_AGB_BANDS)
        .toFloat()
        .set({"year": year, "system:time_start": ee.Date.fromYMD(year, 1, 1).millis()})
    )


def get_esa_cci_agb(
    aoi: ee.Geometry, years: Sequence[int] | None = None
) -> ee.ImageCollection:
    """ESA CCI Biomass v6 above-ground biomass, one 100 m image per map year.

    Bands ``AGB`` (above-ground woody biomass of living trees, Mg/ha, oven-dry) and
    ``SD`` (its per-pixel standard deviation, Mg/ha). No-data was ingested as 0, so
    those pixels are masked. Map years are 2007, 2010 and 2015-2022
    (``ESA_CCI_AGB_YEARS``); v7 (2023-24) is not in EE. Years are checked locally,
    with no EE request.

    Args:
        aoi: Area of interest as ee.Geometry (filters the collection).
        years: Map years to return (default: all of ``ESA_CCI_AGB_YEARS``).

    Returns:
        ee.ImageCollection of float images with bands ``AGB`` and ``SD``, each with ``year`` (int) and ``system:time_start`` (1 January).

    Raises:
        ValueError: If years is empty or names a year without a map.
    """
    year_list = _check_years(ESA_CCI_AGB_YEARS if years is None else years)
    return ee.ImageCollection([_agb_year(aoi, year) for year in year_list])


def get_esa_cci_agb_change(
    aoi: ee.Geometry, start_year: int, end_year: int
) -> ee.Image:
    """AGB change between two map years, with the product's published uncertainty.

    ``agb_change`` = AGB(end_year) - AGB(start_year) (Mg/ha, from the main maps).
    ``agb_change_sd`` and ``agb_change_qf`` come from the ``ESA_CCI_AGB_DIFF``
    collection (bands ``AGB_DIFF_SD`` and ``AGB_DIFF_QF``; see the CCI Biomass
    product user guide for the flag codes). Only the published pairs
    (``ESA_CCI_AGB_DIFF_PAIRS``: consecutive years 2015-2022, 2007-2010 and
    2010-2020) are allowed, because other pairs have no uncertainty layer.

    Args:
        aoi: Area of interest as ee.Geometry (filters the collections).
        start_year: Earlier map year.
        end_year: Later map year.

    Returns:
        ee.Image with float bands ``agb_change``, ``agb_change_sd`` and int ``agb_change_qf``, with ``start_year`` and ``end_year`` set.

    Raises:
        ValueError: If (start_year, end_year) is not a published change pair.
    """
    if (start_year, end_year) not in ESA_CCI_AGB_DIFF_PAIRS:
        raise ValueError(
            f"No published ESA CCI AGB change for {start_year}-{end_year}; pairs are "
            f"{list(ESA_CCI_AGB_DIFF_PAIRS)}."
        )
    change = (
        _agb_year(aoi, end_year)
        .select("AGB")
        .subtract(_agb_year(aoi, start_year).select("AGB"))
    )
    diff = ee.Image(
        ee.ImageCollection(ESA_CCI_AGB_DIFF_COLLECTION)
        .filter(
            ee.Filter.eq(
                "system:index",
                ESA_CCI_AGB_DIFF_NAME.format(start=start_year, end=end_year),
            )
        )
        .first()
    )
    return (
        ee.Image.cat(
            [
                change,
                diff.select("AGB_DIFF_SD").toFloat(),
                diff.select("AGB_DIFF_QF"),
            ]
        )
        .rename(ESA_CCI_AGB_CHANGE_BANDS)
        .set({"start_year": start_year, "end_year": end_year})
    )
