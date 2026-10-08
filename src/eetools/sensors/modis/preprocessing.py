import ee

from eetools.constants import (
    FPAR_SCALE_FACTOR,
    FPAR_STDDEV_SCALE_FACTOR,
    LAI_SCALE_FACTOR,
    LAI_STDDEV_SCALE_FACTOR,
    MODIS_BURNED_AREA_BANDS,
    MODIS_BURNED_AREA_COLLECTION,
    MODIS_LAI_FPAR_BANDS,
    MODIS_LAI_FPAR_COLLECTION,
    MODIS_NPP_BANDS,
    MODIS_NPP_COLLECTION,
    MODIS_NPP_SCALE_FACTOR,
    MODIS_NPP_SCALED_BANDS,
    MODIS_VCF_BANDS,
    MODIS_VCF_COLLECTION,
    MODIS_VI_BANDS,
    MODIS_VI_COLLECTION,
    MODIS_VI_MAX_SUMMARY_QA,
    MODIS_VI_SCALE_FACTOR,
    MODIS_VI_SCALED_BANDS,
)
from eetools.sensors.modis.masking import (
    add_modis_burned_band,
    mask_modis_npp_fill,
    mask_modis_vcf_fill,
    mask_modis_vi_qa,
)
from eetools.utils import validate_collection_date_range

# --------------------------------------------------------------------------- #
# Private helpers                                                              #
# --------------------------------------------------------------------------- #


def _scale_bands(image: ee.Image, bands: list[str], scale_factor: float) -> ee.Image:
    scaled = image.select(bands).multiply(scale_factor)
    return ee.Image(
        image.addBands(scaled, overwrite=True).copyProperties(
            image, image.propertyNames()
        )
    )


def _validate_modis_date_range(
    collection_id: str,
    aoi: ee.Geometry,
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    sensor_label: str,
) -> tuple[ee.Date, ee.Date]:
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    validate_collection_date_range(
        collection_id=collection_id,
        aoi=aoi,
        start_date=start,
        end_date=end,
        sensor_label=sensor_label,
    )
    return start, end


# --------------------------------------------------------------------------- #
# MCD15A3H LAI / FPAR                                                          #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# fPAR recomputation                                                           #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# MOD13Q1 vegetation indices                                                   #
# --------------------------------------------------------------------------- #


def get_modis_vi_collection(
    aoi: ee.Geometry,
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    apply_qa_mask: bool = True,
    max_summary_qa: int = MODIS_VI_MAX_SUMMARY_QA,
    scale_bands: bool = True,
) -> ee.ImageCollection:
    """Build a MODIS MOD13Q1 250 m 16-day NDVI/EVI collection for the AOI and date range.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as a string (e.g. '2020-01-01') or ee.Date.
        end_date: Collection end date as a string or ee.Date.
        apply_qa_mask: If True, masks pixels by SummaryQA reliability (default True).
        max_summary_qa: Highest SummaryQA value kept when masking: 0 good, 1 marginal, 2 snow/ice, 3 cloudy (default 1).
        scale_bands: If True, applies the 0.0001 scale factor to NDVI, EVI, and the surface-reflectance bands (default True).

    Returns:
        ee.ImageCollection of MOD13Q1 images with the MODIS_VI_BANDS (NDVI, EVI, red/nir/blue/swir2 reflectance, SummaryQA, DetailedQA). Raises ValueError if the date range falls outside the available MOD13Q1 data.
    """
    start, end = _validate_modis_date_range(
        MODIS_VI_COLLECTION, aoi, start_date, end_date, "MODIS MOD13Q1 imagery"
    )
    col = (
        ee.ImageCollection(MODIS_VI_COLLECTION)
        .filterDate(start, end)
        .filterBounds(aoi)
        .select(MODIS_VI_BANDS)
    )

    if apply_qa_mask:
        col = col.map(lambda img: mask_modis_vi_qa(img, max_summary_qa))

    if scale_bands:
        col = col.map(
            lambda img: _scale_bands(img, MODIS_VI_SCALED_BANDS, MODIS_VI_SCALE_FACTOR)
        )

    return col


# --------------------------------------------------------------------------- #
# MCD64A1 burned area                                                          #
# --------------------------------------------------------------------------- #


def get_modis_burned_area_collection(
    aoi: ee.Geometry,
    start_date: str | ee.Date,
    end_date: str | ee.Date,
) -> ee.ImageCollection:
    """Build a MODIS MCD64A1 500 m monthly burned-area collection with a 0/1 burned flag.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as a string (e.g. '2020-01-01') or ee.Date.
        end_date: Collection end date as a string or ee.Date.

    Returns:
        ee.ImageCollection of monthly MCD64A1 images with the MODIS_BURNED_AREA_BANDS plus a 'burned' band (1 burned, 0 unburned, masked over water/insufficient data); sum 'burned' over time for burn frequency. Raises ValueError if the date range falls outside the available MCD64A1 data.
    """
    start, end = _validate_modis_date_range(
        MODIS_BURNED_AREA_COLLECTION,
        aoi,
        start_date,
        end_date,
        "MODIS MCD64A1 imagery",
    )
    return (
        ee.ImageCollection(MODIS_BURNED_AREA_COLLECTION)
        .filterDate(start, end)
        .filterBounds(aoi)
        .select(MODIS_BURNED_AREA_BANDS)
        .map(add_modis_burned_band)
    )


# --------------------------------------------------------------------------- #
# MOD17A3HGF annual NPP                                                        #
# --------------------------------------------------------------------------- #


def get_modis_npp_collection(
    aoi: ee.Geometry,
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    mask_fill: bool = True,
    scale_bands: bool = True,
) -> ee.ImageCollection:
    """Build a MODIS MOD17A3HGF 500 m annual gap-filled NPP/GPP collection.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as a string (e.g. '2020-01-01') or ee.Date. Images are stamped January 1 of each year.
        end_date: Collection end date as a string or ee.Date.
        mask_fill: If True, masks Npp/Gpp fill codes such as water and barren land (default True).
        scale_bands: If True, applies the 0.0001 scale factor so Npp and Gpp are in kg C/m^2/year (default True).

    Returns:
        ee.ImageCollection of annual MOD17A3HGF images with the Npp, Gpp, and Npp_QC bands. Raises ValueError if the date range falls outside the available MOD17A3HGF data.
    """
    start, end = _validate_modis_date_range(
        MODIS_NPP_COLLECTION, aoi, start_date, end_date, "MODIS MOD17A3HGF imagery"
    )
    col = (
        ee.ImageCollection(MODIS_NPP_COLLECTION)
        .filterDate(start, end)
        .filterBounds(aoi)
        .select(MODIS_NPP_BANDS)
    )

    if mask_fill:
        col = col.map(mask_modis_npp_fill)

    if scale_bands:
        col = col.map(
            lambda img: _scale_bands(
                img, MODIS_NPP_SCALED_BANDS, MODIS_NPP_SCALE_FACTOR
            )
        )

    return col


# --------------------------------------------------------------------------- #
# MOD44B vegetation continuous fields                                          #
# --------------------------------------------------------------------------- #


def get_modis_vcf_collection(
    aoi: ee.Geometry,
    start_date: str | ee.Date,
    end_date: str | ee.Date,
    mask_fill: bool = True,
) -> ee.ImageCollection:
    """Build a MODIS MOD44B 250 m annual vegetation continuous fields (percent cover) collection.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_date: Collection start date as a string (e.g. '2020-01-01') or ee.Date. Images are stamped early March of each year.
        end_date: Collection end date as a string or ee.Date.
        mask_fill: If True, masks the water (200) and fill (253) codes in the percent-cover bands (default True).

    Returns:
        ee.ImageCollection of annual MOD44B images with Percent_Tree_Cover, Percent_NonTree_Vegetation, and Percent_NonVegetated bands (0-100 %). Raises ValueError if the date range falls outside the available MOD44B data.
    """
    start, end = _validate_modis_date_range(
        MODIS_VCF_COLLECTION, aoi, start_date, end_date, "MODIS MOD44B imagery"
    )
    col = (
        ee.ImageCollection(MODIS_VCF_COLLECTION)
        .filterDate(start, end)
        .filterBounds(aoi)
        .select(MODIS_VCF_BANDS)
    )

    if mask_fill:
        col = col.map(mask_modis_vcf_fill)

    return col
