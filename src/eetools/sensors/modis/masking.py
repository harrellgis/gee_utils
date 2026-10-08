import ee

from eetools.constants import (
    MODIS_BURN_DATE_BAND,
    MODIS_BURNED_AREA_VALID_QA_BITS,
    MODIS_BURNED_BAND,
    MODIS_GPP_VALID_MAX,
    MODIS_NPP_VALID_MAX,
    MODIS_VCF_BANDS,
    MODIS_VCF_VALID_MAX,
    MODIS_VI_MAX_SUMMARY_QA,
)

# --------------------------------------------------------------------------- #
# MOD13Q1 vegetation indices                                                   #
# --------------------------------------------------------------------------- #


def mask_modis_vi_qa(
    image: ee.Image, max_summary_qa: int = MODIS_VI_MAX_SUMMARY_QA
) -> ee.Image:
    """Mask a MOD13Q1 image to pixels whose SummaryQA reliability is good enough.

    Args:
        image: Raw MOD13Q1 ee.Image containing the SummaryQA band.
        max_summary_qa: Highest SummaryQA value kept: 0 good, 1 marginal, 2 snow/ice, 3 cloudy (default MODIS_VI_MAX_SUMMARY_QA = 1). Fill pixels (-1) are always dropped.

    Returns:
        ee.Image with every band masked where SummaryQA is fill or exceeds max_summary_qa.
    """
    summary_qa = image.select("SummaryQA")
    return image.updateMask(summary_qa.gte(0).And(summary_qa.lte(max_summary_qa)))


# --------------------------------------------------------------------------- #
# MCD64A1 burned area                                                          #
# --------------------------------------------------------------------------- #


def add_modis_burned_band(image: ee.Image) -> ee.Image:
    """Append a 0/1 burned flag to a MCD64A1 image, masked to valid land observations.

    GEE masks BurnDate over unburned land, so the flag unmasks it to 0 and then
    restricts the result to pixels whose QA marks land with valid data (bits 0 and 1).

    Args:
        image: Raw MCD64A1 ee.Image containing the BurnDate and QA bands.

    Returns:
        ee.Image with an added 'burned' band (1 = burned this month, 0 = unburned, masked over water/insufficient data).
    """
    valid = (
        image.select("QA")
        .bitwiseAnd(MODIS_BURNED_AREA_VALID_QA_BITS)
        .eq(MODIS_BURNED_AREA_VALID_QA_BITS)
    )
    burned = (
        image.select(MODIS_BURN_DATE_BAND)
        .unmask(0)
        .gt(0)
        .updateMask(valid)
        .toByte()
        .rename(MODIS_BURNED_BAND)
    )
    return image.addBands(burned)


# --------------------------------------------------------------------------- #
# MOD17A3HGF annual NPP                                                        #
# --------------------------------------------------------------------------- #


def mask_modis_npp_fill(image: ee.Image) -> ee.Image:
    """Mask MOD17A3HGF fill codes (water, barren, urban, etc.) in the Npp and Gpp bands.

    Args:
        image: Raw (unscaled) MOD17A3HGF ee.Image containing the Npp and Gpp bands.

    Returns:
        ee.Image with Npp and Gpp each masked above their valid-range maximum; other bands are unchanged.
    """
    npp = image.select("Npp")
    gpp = image.select("Gpp")
    masked = ee.Image.cat(
        [
            npp.updateMask(npp.lte(MODIS_NPP_VALID_MAX)),
            gpp.updateMask(gpp.lte(MODIS_GPP_VALID_MAX)),
        ]
    )
    return image.addBands(masked, overwrite=True)


# --------------------------------------------------------------------------- #
# MOD44B vegetation continuous fields                                          #
# --------------------------------------------------------------------------- #


def mask_modis_vcf_fill(image: ee.Image) -> ee.Image:
    """Mask MOD44B water (200) and fill (253) codes in the percent-cover bands.

    Args:
        image: Raw MOD44B ee.Image containing the MODIS_VCF_BANDS percent-cover bands.

    Returns:
        ee.Image with each percent-cover band masked wherever its value exceeds 100; other bands are unchanged.
    """
    cover = image.select(MODIS_VCF_BANDS)
    return image.addBands(
        cover.updateMask(cover.lte(MODIS_VCF_VALID_MAX)), overwrite=True
    )
