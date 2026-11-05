import ee

from eetools.constants import (
    CLIMATOLOGY_END_YEAR,
    CLIMATOLOGY_START_YEAR,
    TERRACLIMATE_COLLECTION,
    TERRACLIMATE_SCALES,
)
from eetools.utils import validate_collection_date_range


def scale_terraclimate_image(image: ee.Image) -> ee.Image:
    """Apply TerraClimate's per-band scale factors to the bands in TERRACLIMATE_SCALES.

    ``def``, ``pet``, ``tmmn`` and ``tmmx`` are stored x10; ``pr`` is NOT scaled. Only
    those five bands are kept.

    Args:
        image: Raw TerraClimate ee.Image (or a temporal mean of raw images).

    Returns:
        ee.Image with bands pr (mm), pet (mm), def (mm), tmmn and tmmx (degC), with
        the source image's properties preserved.
    """
    image = ee.Image(image)
    bands = list(TERRACLIMATE_SCALES)
    scaled = ee.Image.cat(
        [image.select(b).multiply(TERRACLIMATE_SCALES[b]).rename(b) for b in bands]
    )
    return ee.Image(scaled.copyProperties(image, image.propertyNames()))


def get_terraclimate_collection(
    aoi: ee.Geometry,
    start_date: str | ee.Date,
    end_date: str | ee.Date,
) -> ee.ImageCollection:
    """Build a scaled monthly TerraClimate collection for the AOI and date range.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as a string (e.g. '1991-01-01') or ee.Date.
        end_date: Collection end date (exclusive) as a string or ee.Date.

    Returns:
        ee.ImageCollection of monthly images with bands pr, pet, def (mm/month) and
        tmmn, tmmx (degC), per-band scale factors applied.

    Raises:
        ValueError: If the date range falls outside the available TerraClimate data.
    """
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    validate_collection_date_range(
        collection_id=TERRACLIMATE_COLLECTION,
        aoi=aoi,
        start_date=start,
        end_date=end,
        sensor_label="TerraClimate",
    )
    return (
        ee.ImageCollection(TERRACLIMATE_COLLECTION)
        .filterDate(start, end)
        .map(scale_terraclimate_image)
    )


def terraclimate_climatology_from_mean(mean_monthly_raw: ee.Image) -> ee.Image:
    """Derive the climatology bands from a mean of raw (unscaled) monthly images.

    Annual totals are the mean monthly value x 12, so aridity
    ``(P - PET) / PET`` is the same whether computed on monthly means or annual sums.
    Scaling is applied here, per band (``pr`` unscaled).

    Args:
        mean_monthly_raw: ee.Image holding the temporal mean of raw TerraClimate bands
            pr, pet, def, tmmn and tmmx.

    Returns:
        ee.Image with bands 'cwd' (mean annual climatic water deficit, mm/yr),
        'aridity' ((P - PET) / PET, unitless), 'tmin' and 'tmax' (degC).
    """
    scaled = scale_terraclimate_image(mean_monthly_raw)
    pr = scaled.select("pr")
    pet = scaled.select("pet")
    cwd = scaled.select("def").multiply(12).rename("cwd")
    aridity = pr.subtract(pet).divide(pet).rename("aridity")
    return ee.Image.cat(
        [
            cwd,
            aridity,
            scaled.select("tmmn").rename("tmin"),
            scaled.select("tmmx").rename("tmax"),
        ]
    ).toFloat()


def get_terraclimate_climatology(
    start_year: int = CLIMATOLOGY_START_YEAR,
    end_year: int = CLIMATOLOGY_END_YEAR,
) -> ee.Image:
    """Build a static TerraClimate climatology image (one mean over a year window).

    TerraClimate is used for climatology only, never trends. The monthly series is
    averaged once, at native ~4.6 km, before any reduction.

    Args:
        start_year: First year of the window (inclusive; default 1991).
        end_year: Last year of the window (inclusive; default 2020).

    Returns:
        ee.Image with bands 'cwd' (mm/yr), 'aridity', 'tmin', 'tmax' (degC), in the
        dataset's native projection.

    Raises:
        ValueError: If end_year < start_year.
    """
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) must be >= start_year ({start_year}).")
    col = ee.ImageCollection(TERRACLIMATE_COLLECTION).filterDate(
        f"{start_year}-01-01", f"{end_year + 1}-01-01"
    )
    projection = col.first().projection()
    mean_raw = col.select(list(TERRACLIMATE_SCALES)).mean()
    return terraclimate_climatology_from_mean(mean_raw).setDefaultProjection(projection)
