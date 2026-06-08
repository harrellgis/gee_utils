import ee

from eetools.constants import (
    FPAR_SCALE_FACTOR,
    FPAR_STDDEV_SCALE_FACTOR,
    LAI_SCALE_FACTOR,
    LAI_STDDEV_SCALE_FACTOR,
    MODIS_LAI_FPAR_BANDS,
    MODIS_LAI_FPAR_COLLECTION,
)


def _mask_modis_lai_fpar_qa(image: ee.Image) -> ee.Image:
    qa_mask = (
        image.select("FparLai_QC")
        .bitwiseAnd(1)
        .eq(0)
        .And(image.select("FparLai_QC").rightShift(5).bitwiseAnd(7).lte(1))
        .And(image.select("FparExtra_QC").rightShift(5).bitwiseAnd(1).eq(0))
        .And(image.select("FparExtra_QC").rightShift(6).bitwiseAnd(1).eq(0))
    )
    return image.updateMask(qa_mask)


def _scale_modis_lai_fpar_bands(image: ee.Image) -> ee.Image:
    scaled_bands = ee.Image.cat(
        [
            image.select("Fpar").multiply(FPAR_SCALE_FACTOR),
            image.select("Lai").multiply(LAI_SCALE_FACTOR),
            image.select("FparStdDev").multiply(FPAR_STDDEV_SCALE_FACTOR),
            image.select("LaiStdDev").multiply(LAI_STDDEV_SCALE_FACTOR),
        ]
    )
    qa_bands = image.select(["FparLai_QC", "FparExtra_QC"])
    return scaled_bands.addBands(qa_bands).copyProperties(image, image.propertyNames())


def get_modis_lai_fpar_col(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    apply_qa_mask: bool = True,
    scale_bands: bool = True,
    select_bands: bool = True,
) -> ee.ImageCollection:
    """Build a MODIS MCD15A3H LAI/FPAR collection filtered to the AOI and date range.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as a string (e.g. '2020-01-01').
        end_date: Collection end date as a string (e.g. '2021-01-01').
        apply_qa_mask: If True, masks pixels to good-quality, main-algorithm, cloud-free, and shadow-free retrievals (default True).
        scale_bands: If True, applies the MODIS scale factors to Fpar, Lai, FparStdDev, and LaiStdDev bands (default True).
        select_bands: If True, restricts the collection to the standard MODIS_LAI_FPAR_BANDS before any processing (default True).

    Returns:
        ee.ImageCollection of MODIS MCD15A3H images with optional QA masking and band scaling applied.
    """
    col = (
        ee.ImageCollection(MODIS_LAI_FPAR_COLLECTION)
        .filterDate(start_date, end_date)
        .filterBounds(aoi)
    )

    if select_bands:
        col = col.select(MODIS_LAI_FPAR_BANDS)

    if apply_qa_mask:
        col = col.map(_mask_modis_lai_fpar_qa)

    if scale_bands:
        col = col.map(_scale_modis_lai_fpar_bands)

    return col


def overwrite_fpar_band(
    image: ee.Image,
    red_band: str,
    nir_band: str,
    ndvi_soil: float,
    ndvi_veg: float,
    max_fpar: float = 0.95,
    output_band: str = "fpar",
) -> ee.Image:
    """Recompute and overwrite an fPAR band using alternative NDVI endpoint parameters.

    Args:
        image: ee.Image containing at least red_band and nir_band.
        red_band: Name of the red reflectance band.
        nir_band: Name of the near-infrared reflectance band.
        ndvi_soil: NDVI value representing bare soil (lower endpoint of the linear fPAR scale).
        ndvi_veg: NDVI value representing full vegetation cover (upper endpoint).
        max_fpar: Maximum fPAR value applied as an upper clamp (default 0.95).
        output_band: Name of the output fPAR band to overwrite (default 'fpar').

    Returns:
        ee.Image with the specified fPAR band replaced by the newly computed values in [0, max_fpar].
    """
    ndvi = image.normalizedDifference([nir_band, red_band])
    fpar = (
        ndvi.subtract(ndvi_soil)
        .divide(ndvi_veg - ndvi_soil)
        .clamp(0, 1)
        .multiply(max_fpar)
        .rename(output_band)
    )
    return image.addBands(fpar, overwrite=True)
