"""Generic spectral-index functions.

These follow the Awesome Spectral Indices (ASI) standard — Montero et al. 2023,
*Scientific Data* (doi:10.1038/s41597-023-02096-0) — the catalogue behind the
``spyndex``/``eemont`` ecosystem. eetools computes them natively (server-side
``ee.Image`` ops) rather than depending on spyndex; each function's docstring records
the canonical ASI ``short_name`` and the original-paper reference so the formulas stay
traceable to the standard.

The ``band_map`` keys used throughout are a relabeling of ASI's standardized band
letters, so an index defined once works on any sensor:

    B = blue, G = green, R = red, RE1 = red_edge, N = nir, S1 = swir1, S2 = swir2

(see ``eetools.constants.ASI_BAND_LETTERS``).

Three names deliberately diverge from ASI — read the per-function notes before relying
on them: ``NDWI`` here is McFeeters open-water, NOT ASI's Gao moisture ``NDWI``; ``NDRE``
is ASI's ``NDREI``; and ``kNDVI`` uses the σ-parameterized RBF form, not ASI's
``tanh(NDVI²)`` shortcut. ``Fpar`` is not an ASI index at all.
"""

from collections.abc import Callable
from dataclasses import dataclass

import ee

from eetools.constants import SIGMA


##################### Generic vegetation index functions ####################
def calc_ndvi(
    image: ee.Image,
    nir_band: str,
    red_band: str,
    output_band: str = "NDVI",
) -> ee.Image:
    """Calculate NDVI as a standard proxy for green vegetation vigor and cover.

    Args:
        image: ee.Image containing at least nir_band and red_band.
        nir_band: Name of the near-infrared reflectance band.
        red_band: Name of the red reflectance band.
        output_band: Name of the output NDVI band (default 'NDVI').

    Returns:
        ee.Image with a single band named output_band containing NDVI values in [-1, 1].

    ASI: NDVI — Normalized Difference Vegetation Index (doi:10.1016/0034-4257(94)90134-1).
    """
    return image.normalizedDifference([nir_band, red_band]).rename(output_band)


def calc_evi(
    image: ee.Image,
    nir_band: str,
    red_band: str,
    blue_band: str,
    output_band: str = "EVI",
) -> ee.Image:
    """Calculate EVI for improved vegetation sensitivity in higher biomass and variable
    soil backgrounds.

    Args:
        image: ee.Image containing at least nir_band, red_band, and blue_band.
        nir_band: Name of the near-infrared reflectance band.
        red_band: Name of the red reflectance band.
        blue_band: Name of the blue reflectance band.
        output_band: Name of the output EVI band (default 'EVI').

    Returns:
        ee.Image with a single band named output_band containing EVI values.

    ASI: EVI; the hardcoded 2.5 / 6 / 7.5 / 1 coefficients are ASI's defaults
        g=2.5, C1=6, C2=7.5, L=1 (doi:10.1016/S0034-4257(96)00112-5).
    """
    return image.expression(
        "2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))",
        {
            "nir": image.select(nir_band),
            "red": image.select(red_band),
            "blue": image.select(blue_band),
        },
    ).rename(output_band)


def calc_savi(
    image: ee.Image,
    nir_band: str,
    red_band: str,
    L: float = 0.5,
    output_band: str = "SAVI",
) -> ee.Image:
    """Calculate SAVI to reduce soil background effects in sparsely vegetated
    landscapes.

    Args:
        image: ee.Image containing at least nir_band and red_band.
        nir_band: Name of the near-infrared reflectance band.
        red_band: Name of the red reflectance band.
        L: Soil brightness correction factor; typically 0.5 for intermediate vegetation cover (default 0.5).
        output_band: Name of the output SAVI band (default 'SAVI').

    Returns:
        ee.Image with a single band named output_band containing SAVI values.

    ASI: SAVI (doi:10.1016/0034-4257(88)90106-X).
    """
    return image.expression(
        "((nir - red) / (nir + red + L)) * (1 + L)",
        {
            "nir": image.select(nir_band),
            "red": image.select(red_band),
            "L": L,
        },
    ).rename(output_band)


def calc_ndwi(
    image: ee.Image,
    green_band: str,
    nir_band: str,
    output_band: str = "NDWI",
) -> ee.Image:
    """Calculate NDWI to highlight surface water and vegetation water-related contrast.

    Args:
        image: ee.Image containing at least green_band and nir_band.
        green_band: Name of the green reflectance band.
        nir_band: Name of the near-infrared reflectance band.
        output_band: Name of the output NDWI band (default 'NDWI').

    Returns:
        ee.Image with a single band named output_band containing NDWI values in [-1, 1].

    ASI: This is McFeeters open-water NDWI (G - N)/(G + N), doi:10.1080/01431169608948714.
        NOTE: ASI's "NDWI" is the Gao moisture index (N - S1)/(N + S1) — identical to
        ``calc_ndmi`` here — NOT this index. Do not conflate the two.
    """
    return image.normalizedDifference([green_band, nir_band]).rename(output_band)


def calc_mndwi(
    image: ee.Image,
    green_band: str,
    swir1_band: str,
    output_band: str = "MNDWI",
) -> ee.Image:
    """Calculate MNDWI using green and SWIR1 to enhance open-water detection against
    land surfaces.

    Args:
        image: ee.Image containing at least green_band and swir1_band.
        green_band: Name of the green reflectance band.
        swir1_band: Name of the SWIR1 reflectance band.
        output_band: Name of the output MNDWI band (default 'MNDWI').

    Returns:
        ee.Image with a single band named output_band containing MNDWI values in [-1, 1].

    ASI: MNDWI (doi:10.1080/01431160600589179).
    """
    return image.normalizedDifference([green_band, swir1_band]).rename(output_band)


def calc_ndmi(
    image: ee.Image,
    nir_band: str,
    swir1_band: str,
    output_band: str = "NDMI",
) -> ee.Image:
    """Calculate NDMI as a moisture-sensitive index for vegetation and surface dryness
    assessment.

    Args:
        image: ee.Image containing at least nir_band and swir1_band.
        nir_band: Name of the near-infrared reflectance band.
        swir1_band: Name of the SWIR1 reflectance band.
        output_band: Name of the output NDMI band (default 'NDMI').

    Returns:
        ee.Image with a single band named output_band containing NDMI values in [-1, 1].

    ASI: NDMI (doi:10.1016/S0034-4257(01)00318-2); formula-identical to ASI's Gao "NDWI".
    """
    return image.normalizedDifference([nir_band, swir1_band]).rename(output_band)


def calc_nbr(
    image: ee.Image,
    nir_band: str,
    swir2_band: str,
    output_band: str = "NBR",
) -> ee.Image:
    """Calculate NBR for burn severity and broader disturbance screening.

    Args:
        image: ee.Image containing at least nir_band and swir2_band.
        nir_band: Name of the near-infrared reflectance band.
        swir2_band: Name of the SWIR2 reflectance band.
        output_band: Name of the output NBR band (default 'NBR').

    Returns:
        ee.Image with a single band named output_band containing NBR values in [-1, 1].

    ASI: NBR (doi:10.3133/ofr0211).
    """
    return image.normalizedDifference([nir_band, swir2_band]).rename(output_band)


def calc_nirv(
    image: ee.Image,
    nir_band: str,
    red_band: str,
    output_band: str = "NIRv",
) -> ee.Image:
    """Calculate NIRv as a vegetation productivity proxy combining NIR reflectance and
    NDVI.

    Args:
        image: ee.Image containing at least nir_band and red_band.
        nir_band: Name of the near-infrared reflectance band.
        red_band: Name of the red reflectance band.
        output_band: Name of the output NIRv band (default 'NIRv').

    Returns:
        ee.Image with a single band named output_band containing NIRv values (NIR * NDVI).

    ASI: NIRv (doi:10.1126/sciadv.abb7578).
    """
    ndvi = calc_ndvi(image, nir_band=nir_band, red_band=red_band)
    return image.select(nir_band).multiply(ndvi).rename(output_band)


def calc_ndre(
    image: ee.Image,
    nir_band: str,
    red_edge_band: str,
    output_band: str = "NDRE",
) -> ee.Image:
    """Calculate NDRE using NIR and red-edge reflectance as a canopy chlorophyll proxy.

    Args:
        image: ee.Image containing at least nir_band and red_edge_band.
        nir_band: Name of the near-infrared reflectance band.
        red_edge_band: Name of the red-edge reflectance band.
        output_band: Name of the output NDRE band (default 'NDRE').

    Returns:
        ee.Image with a single band named output_band containing NDRE values in [-1, 1].

    ASI: NDREI — ASI's name for this NIR/RE1 index; eetools keeps the band name "NDRE"
        (doi:10.1016/1011-1344(93)06963-4).
    """
    return image.normalizedDifference([nir_band, red_edge_band]).rename(output_band)


def calc_bsi(
    image: ee.Image,
    swir1_band: str,
    red_band: str,
    nir_band: str,
    blue_band: str,
    output_band: str = "BSI",
) -> ee.Image:
    """Calculate the Bare Soil Index (BSI) to highlight exposed/bare soil against
    vegetation.

    BSI = ((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE)).

    Uses .expression() rather than normalizedDifference because the numerator and
    denominator are band sums and BSI is validly negative over vegetated surfaces.

    Args:
        image: ee.Image containing at least swir1_band, red_band, nir_band, and blue_band.
        swir1_band: Name of the SWIR1 reflectance band.
        red_band: Name of the red reflectance band.
        nir_band: Name of the near-infrared reflectance band.
        blue_band: Name of the blue reflectance band.
        output_band: Name of the output BSI band (default 'BSI').

    Returns:
        ee.Image with a single band named output_band containing BSI values in [-1, 1].

    ASI: BSI (doi:10.1016/S0034-4257(03)00146-7).
    """
    return image.expression(
        "((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))",
        {
            "swir1": image.select(swir1_band),
            "red": image.select(red_band),
            "nir": image.select(nir_band),
            "blue": image.select(blue_band),
        },
    ).rename(output_band)


##################### Fpar calculation ####################
def calc_fpar(
    image: ee.Image,
    nir_band: str,
    red_band: str,
    ndvi_soil: float = 0.15,
    ndvi_veg: float = 0.80,
    max_fpar: float = 0.95,
    output_band: str = "Fpar",
) -> ee.Image:
    """Estimate fPAR from surface reflectance using a linear NDVI scaling approach.

    Args:
        image: ee.Image containing at least nir_band and red_band.
        nir_band: Name of the near-infrared reflectance band.
        red_band: Name of the red reflectance band.
        ndvi_soil: NDVI value representing bare soil (lower endpoint of the linear scale, default 0.15).
        ndvi_veg: NDVI value representing full vegetation cover (upper endpoint, default 0.80).
        max_fpar: Maximum fPAR value applied as an upper clamp (default 0.95).
        output_band: Name of the output fPAR band (default 'Fpar').

    Returns:
        ee.Image with a single band named output_band containing fPAR values in [0, max_fpar].

    ASI: Not an ASI catalogue index; bespoke linear NDVI->fPAR scaling.
    """
    ndvi = image.normalizedDifference([nir_band, red_band])
    fpar = (
        ndvi.subtract(ndvi_soil)
        .divide(ndvi_veg - ndvi_soil)
        .clamp(0, 1)
        .multiply(max_fpar)
    )
    return fpar.rename(output_band)


##################### kNDVI variants ####################
def calc_kndvi_fixed_sigma(
    image: ee.Image,
    red_band: str,
    nir_band: str,
    sigma: "ee.Image | ee.Number | float" = SIGMA,
    output_band: str = "kNDVI_fixed",
) -> ee.Image:
    """Calculate RBF-kernel NDVI using a fixed or supplied sigma: kNDVI = tanh((NIR - RED)^2 / (2*sigma)^2).

    Args:
        image: ee.Image containing at least red_band and nir_band.
        red_band: Name of the red reflectance band.
        nir_band: Name of the near-infrared reflectance band.
        sigma: Kernel width parameter as ee.Number or ee.Image; defaults to the global SIGMA constant.
        output_band: Name of the output kNDVI band (default 'kNDVI_fixed').

    Returns:
        ee.Image with a single band named output_band containing kNDVI values in [0, 1].

    ASI: kNDVI (doi:10.1038/s41467-021-22951-1). ASI defines kNDVI as the tanh(NDVI^2)
        shortcut; eetools intentionally uses the sigma-parameterized RBF form instead —
        see ``calc_kndvi_est_sigma`` / ``calc_kndvi_temp_est_sigma`` for data-driven sigma.
    """
    red = image.select(red_band)
    nir = image.select(nir_band)
    d2 = nir.subtract(red).pow(2)
    kndvi = d2.divide(sigma).divide(sigma).divide(4.0).tanh()
    return kndvi.rename(output_band)


def calc_kndvi_est_sigma(
    image: ee.Image,
    aoi: ee.Geometry,
    red_band: str,
    nir_band: str,
    scale: int,
    max_pixels: int = 1_000_000_000,
    tile_scale: int = 4,
) -> ee.Image:
    """Calculate kNDVI using a per-image sigma estimated from the regional median of
    0.5*(NIR+RED) within the AOI.

    Args:
        image: ee.Image containing at least red_band and nir_band.
        aoi: Area of interest as ee.Geometry used for the sigma reduction.
        red_band: Name of the red reflectance band.
        nir_band: Name of the near-infrared reflectance band.
        scale: Pixel scale in metres for the sigma reduceRegion call.
        max_pixels: Maximum number of pixels for the reduction (default 1_000_000_000).
        tile_scale: EE tileScale parameter to avoid memory limits (default 4).

    Returns:
        ee.Image with a single band 'kNDVI_est' computed with the per-image estimated sigma.
    """
    red = image.select(red_band)
    nir = image.select(nir_band)
    sigma_image = nir.add(red).multiply(0.5).rename("sigma")

    sigma_raw = sigma_image.reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=aoi,
        scale=scale,
        maxPixels=max_pixels,
        tileScale=tile_scale,
    ).get("sigma")

    sigma = ee.Number(ee.Algorithms.If(sigma_raw, sigma_raw, SIGMA))

    return calc_kndvi_fixed_sigma(
        image=image,
        red_band=red_band,
        nir_band=nir_band,
        sigma=sigma,
        output_band="kNDVI_est",
    )


def calc_kndvi_temp_est_sigma(
    collection: ee.ImageCollection,
    red_band: str,
    nir_band: str,
    reducer: ee.Reducer | None = None,
) -> ee.Image:
    """Estimate a per-pixel temporal sigma image from 0.5*(NIR+RED) reduced across an
    image collection.

    Args:
        collection: ee.ImageCollection used to derive the temporal sigma estimate.
        red_band: Name of the red reflectance band in each image.
        nir_band: Name of the near-infrared reflectance band in each image.
        reducer: ee.Reducer applied across time to compute sigma; defaults to ee.Reducer.mean().

    Returns:
        ee.Image with a single band 'sigma_temp_est' containing the per-pixel temporal sigma values.
    """
    if reducer is None:
        reducer = ee.Reducer.mean()

    def _add_sigma(image: ee.Image) -> ee.Image:
        red = image.select(red_band)
        nir = image.select(nir_band)
        sigma = nir.add(red).multiply(0.5).rename("sigma")
        return image.addBands(sigma)

    sigma_collection = collection.map(_add_sigma).select("sigma")
    sigma_image = sigma_collection.reduce(reducer)
    # ee.Reducer.mean() produces "sigma_mean" — rename to a clean, predictable band name
    return sigma_image.rename("sigma_temp_est")


##################### Collection-level kNDVI helpers ####################
def add_kndvi_est_to_collection(
    collection: ee.ImageCollection,
    aoi: ee.Geometry,
    red_band: str,
    nir_band: str,
    scale: int,
    max_pixels: int = 1_000_000_000,
    tile_scale: int = 4,
    crs: str | None = None,
    best_effort: bool = False,
) -> ee.ImageCollection:
    """Add a per-image estimated-sigma kNDVI band to every image in a collection.

    Args:
        collection: ee.ImageCollection to process.
        aoi: Area of interest as ee.Geometry used for per-image sigma estimation.
        red_band: Name of the red reflectance band.
        nir_band: Name of the near-infrared reflectance band.
        scale: Pixel scale in metres for sigma reduction.
        max_pixels: Maximum number of pixels for each sigma reduction (default 1_000_000_000).
        tile_scale: EE tileScale parameter to avoid memory limits (default 4).
        crs: Optional CRS string for the sigma reduction; uses the image's native CRS if None.
        best_effort: If True, allows EE to use a coarser scale to avoid memory errors (default False).

    Returns:
        ee.ImageCollection with 'kNDVI_est' appended to each image.
    """

    def _add_band(image: ee.Image) -> ee.Image:
        red = image.select(red_band)
        nir = image.select(nir_band)
        sigma_image = nir.add(red).multiply(0.5).rename("sigma")

        reduce_kwargs = {
            "reducer": ee.Reducer.median(),
            "geometry": aoi,
            "scale": scale,
            "maxPixels": max_pixels,
            "tileScale": tile_scale,
            "bestEffort": best_effort,
        }
        if crs is not None:
            reduce_kwargs["crs"] = crs

        sigma_raw = sigma_image.reduceRegion(**reduce_kwargs).get("sigma")
        sigma = ee.Number(ee.Algorithms.If(sigma_raw, sigma_raw, SIGMA))

        kndvi_est = calc_kndvi_fixed_sigma(
            image=image,
            red_band=red_band,
            nir_band=nir_band,
            sigma=sigma,
            output_band="kNDVI_est",
        )
        return image.addBands(kndvi_est)

    return ee.ImageCollection(collection).map(_add_band)


def add_kndvi_temp_est_to_collection(
    collection: ee.ImageCollection,
    red_band: str,
    nir_band: str,
    reducer: ee.Reducer | None = None,
) -> ee.ImageCollection:
    """Add a temporally estimated-sigma kNDVI band to every image in a collection.

    Args:
        collection: ee.ImageCollection to process; used both to derive the temporal sigma and as the output base.
        red_band: Name of the red reflectance band.
        nir_band: Name of the near-infrared reflectance band.
        reducer: ee.Reducer used to collapse the sigma time series; defaults to ee.Reducer.mean().

    Returns:
        ee.ImageCollection with 'kNDVI_temp_est' appended to each image, computed using the collection-wide sigma image.
    """
    sigma_image = calc_kndvi_temp_est_sigma(
        collection=collection,
        red_band=red_band,
        nir_band=nir_band,
        reducer=reducer,
    )

    def _add_kndvi(image: ee.Image) -> ee.Image:
        kndvi = calc_kndvi_fixed_sigma(
            image=image,
            red_band=red_band,
            nir_band=nir_band,
            sigma=sigma_image,
            output_band="kNDVI_temp_est",
        )
        return image.addBands(kndvi)

    return collection.map(_add_kndvi)


##################### Additional ASI indices ####################
# Vegetation
def calc_gndvi(
    image: ee.Image,
    nir_band: str,
    green_band: str,
    output_band: str = "GNDVI",
) -> ee.Image:
    """Calculate the Green NDVI, a chlorophyll-sensitive variant of NDVI using green.

    Args:
        image: ee.Image containing at least nir_band and green_band.
        nir_band: Name of the near-infrared reflectance band.
        green_band: Name of the green reflectance band.
        output_band: Name of the output GNDVI band (default 'GNDVI').

    Returns:
        ee.Image with a single band named output_band containing GNDVI values in [-1, 1].

    ASI: GNDVI (doi:10.1016/S0034-4257(96)00072-7).
    """
    return image.normalizedDifference([nir_band, green_band]).rename(output_band)


def calc_evi2(
    image: ee.Image,
    nir_band: str,
    red_band: str,
    g: float = 2.5,
    L: float = 1.0,
    output_band: str = "EVI2",
) -> ee.Image:
    """Calculate the two-band EVI (EVI2), an atmosphere-robust EVI needing no blue band.

    Args:
        image: ee.Image containing at least nir_band and red_band.
        nir_band: Name of the near-infrared reflectance band.
        red_band: Name of the red reflectance band.
        g: Gain factor (default 2.5, the ASI default).
        L: Canopy background adjustment (default 1.0, the ASI default).
        output_band: Name of the output EVI2 band (default 'EVI2').

    Returns:
        ee.Image with a single band named output_band containing EVI2 values.

    ASI: EVI2; g=2.5 and L=1.0 are the ASI defaults (doi:10.1016/j.rse.2008.06.006).
    """
    return image.expression(
        "g * (nir - red) / (nir + 2.4 * red + L)",
        {
            "nir": image.select(nir_band),
            "red": image.select(red_band),
            "g": g,
            "L": L,
        },
    ).rename(output_band)


def calc_osavi(
    image: ee.Image,
    nir_band: str,
    red_band: str,
    output_band: str = "OSAVI",
) -> ee.Image:
    """Calculate the Optimized SAVI, a soil-robust vegetation index with a fixed 0.16
    adjustment (no tunable L).

    Args:
        image: ee.Image containing at least nir_band and red_band.
        nir_band: Name of the near-infrared reflectance band.
        red_band: Name of the red reflectance band.
        output_band: Name of the output OSAVI band (default 'OSAVI').

    Returns:
        ee.Image with a single band named output_band containing OSAVI values.

    ASI: OSAVI; the 0.16 soil-adjustment is baked in (doi:10.1016/0034-4257(95)00186-7).
    """
    return image.expression(
        "(nir - red) / (nir + red + 0.16)",
        {
            "nir": image.select(nir_band),
            "red": image.select(red_band),
        },
    ).rename(output_band)


# Moisture
def calc_gvmi(
    image: ee.Image,
    nir_band: str,
    swir2_band: str,
    output_band: str = "GVMI",
) -> ee.Image:
    """Calculate the Global Vegetation Moisture Index from NIR and SWIR2.

    Args:
        image: ee.Image containing at least nir_band and swir2_band.
        nir_band: Name of the near-infrared reflectance band.
        swir2_band: Name of the SWIR2 reflectance band.
        output_band: Name of the output GVMI band (default 'GVMI').

    Returns:
        ee.Image with a single band named output_band containing GVMI values.

    ASI: GVMI; the +0.1 / +0.02 offsets are baked in (doi:10.1016/S0034-4257(02)00037-8).
    """
    return image.expression(
        "((nir + 0.1) - (swir2 + 0.02)) / ((nir + 0.1) + (swir2 + 0.02))",
        {
            "nir": image.select(nir_band),
            "swir2": image.select(swir2_band),
        },
    ).rename(output_band)


# Urban / built-up
def calc_ndbi(
    image: ee.Image,
    swir1_band: str,
    nir_band: str,
    output_band: str = "NDBI",
) -> ee.Image:
    """Calculate the Normalized Difference Built-up Index (SWIR1 vs NIR).

    Args:
        image: ee.Image containing at least swir1_band and nir_band.
        swir1_band: Name of the SWIR1 reflectance band.
        nir_band: Name of the near-infrared reflectance band.
        output_band: Name of the output NDBI band (default 'NDBI').

    Returns:
        ee.Image with a single band named output_band containing NDBI values in [-1, 1].

    ASI: NDBI (doi:10.1080/01431160304987). Algebraically NDBI = -NDMI; exposed as its
        own named index for pipeline clarity.
    """
    return image.normalizedDifference([swir1_band, nir_band]).rename(output_band)


def calc_ui(
    image: ee.Image,
    swir2_band: str,
    nir_band: str,
    output_band: str = "UI",
) -> ee.Image:
    """Calculate the Urban Index (SWIR2 vs NIR).

    Args:
        image: ee.Image containing at least swir2_band and nir_band.
        swir2_band: Name of the SWIR2 reflectance band.
        nir_band: Name of the near-infrared reflectance band.
        output_band: Name of the output UI band (default 'UI').

    Returns:
        ee.Image with a single band named output_band containing UI values in [-1, 1].

    ASI: UI (Kawamura et al. 1996, ISPRS XXXI/B7:321).
    """
    return image.normalizedDifference([swir2_band, nir_band]).rename(output_band)


# Bare soil
def calc_mbi(
    image: ee.Image,
    swir1_band: str,
    swir2_band: str,
    nir_band: str,
    output_band: str = "MBI",
) -> ee.Image:
    """Calculate the Modified Bare Soil Index.

    Args:
        image: ee.Image containing at least swir1_band, swir2_band, and nir_band.
        swir1_band: Name of the SWIR1 reflectance band.
        swir2_band: Name of the SWIR2 reflectance band.
        nir_band: Name of the near-infrared reflectance band.
        output_band: Name of the output MBI band (default 'MBI').

    Returns:
        ee.Image with a single band named output_band containing MBI values.

    ASI: MBI; the +0.5 offset is baked in (doi:10.3390/land10030231).
    """
    return image.expression(
        "((swir1 - swir2 - nir) / (swir1 + swir2 + nir)) + 0.5",
        {
            "swir1": image.select(swir1_band),
            "swir2": image.select(swir2_band),
            "nir": image.select(nir_band),
        },
    ).rename(output_band)


def calc_embi(
    image: ee.Image,
    swir1_band: str,
    swir2_band: str,
    nir_band: str,
    green_band: str,
    output_band: str = "EMBI",
) -> ee.Image:
    """Calculate the Enhanced Modified Bare Soil Index, which suppresses built-up
    surfaces when isolating bare soil.

    Args:
        image: ee.Image containing at least swir1_band, swir2_band, nir_band, and green_band.
        swir1_band: Name of the SWIR1 reflectance band.
        swir2_band: Name of the SWIR2 reflectance band.
        nir_band: Name of the near-infrared reflectance band.
        green_band: Name of the green reflectance band.
        output_band: Name of the output EMBI band (default 'EMBI').

    Returns:
        ee.Image with a single band named output_band containing EMBI values.

    ASI: EMBI (doi:10.1016/j.jag.2022.102703). Built from MBI and the green/SWIR1 MNDWI;
        computed with band arithmetic (not normalizedDifference) so validly-negative
        intermediates are preserved.
    """
    swir1 = image.select(swir1_band)
    swir2 = image.select(swir2_band)
    nir = image.select(nir_band)
    green = image.select(green_band)
    mbi = swir1.subtract(swir2).subtract(nir).divide(swir1.add(swir2).add(nir)).add(0.5)
    mndwi = green.subtract(swir1).divide(green.add(swir1))
    embi = mbi.subtract(mndwi).subtract(0.5).divide(mbi.add(mndwi).add(1.5))
    return embi.rename(output_band)


def calc_dbsi(
    image: ee.Image,
    swir1_band: str,
    green_band: str,
    nir_band: str,
    red_band: str,
    output_band: str = "DBSI",
) -> ee.Image:
    """Calculate the Dry Bare Soil Index (a green/SWIR1 term minus NDVI).

    Args:
        image: ee.Image containing at least swir1_band, green_band, nir_band, and red_band.
        swir1_band: Name of the SWIR1 reflectance band.
        green_band: Name of the green reflectance band.
        nir_band: Name of the near-infrared reflectance band.
        red_band: Name of the red reflectance band.
        output_band: Name of the output DBSI band (default 'DBSI').

    Returns:
        ee.Image with a single band named output_band containing DBSI values.

    ASI: DBSI (doi:10.3390/land7030081).
    """
    swir1_green = image.normalizedDifference([swir1_band, green_band])
    nir_red = image.normalizedDifference([nir_band, red_band])
    return swir1_green.subtract(nir_red).rename(output_band)


# Burn
def calc_bai(
    image: ee.Image,
    red_band: str,
    nir_band: str,
    output_band: str = "BAI",
) -> ee.Image:
    """Calculate the Burned Area Index (spectral distance from a reference burn point).

    Args:
        image: ee.Image containing at least red_band and nir_band.
        red_band: Name of the red reflectance band.
        nir_band: Name of the near-infrared reflectance band.
        output_band: Name of the output BAI band (default 'BAI').

    Returns:
        ee.Image with a single band named output_band containing BAI values (higher = more
        likely burned).

    ASI: BAI; the 0.1 / 0.06 reference reflectances are baked in (Martin 1998, CSIC).
    """
    return image.expression(
        "1.0 / ((0.1 - red) ** 2 + (0.06 - nir) ** 2)",
        {
            "red": image.select(red_band),
            "nir": image.select(nir_band),
        },
    ).rename(output_band)


def calc_nbr2(
    image: ee.Image,
    swir1_band: str,
    swir2_band: str,
    output_band: str = "NBR2",
) -> ee.Image:
    """Calculate NBR2, a SWIR1/SWIR2 burn-severity ratio sensitive to post-fire moisture.

    Args:
        image: ee.Image containing at least swir1_band and swir2_band.
        swir1_band: Name of the SWIR1 reflectance band.
        swir2_band: Name of the SWIR2 reflectance band.
        output_band: Name of the output NBR2 band (default 'NBR2').

    Returns:
        ee.Image with a single band named output_band containing NBR2 values in [-1, 1].

    ASI: NBR2 (USGS Landsat NBR2 product definition).
    """
    return image.normalizedDifference([swir1_band, swir2_band]).rename(output_band)


# Sentinel-2 red-edge (require red_edge2/red_edge3/nir2 in the band map)
def calc_mtci(
    image: ee.Image,
    red_edge_band: str,
    red_edge2_band: str,
    red_band: str,
    output_band: str = "MTCI",
) -> ee.Image:
    """Calculate the MERIS Terrestrial Chlorophyll Index from the red-edge slope.

    Args:
        image: ee.Image containing at least red_edge_band, red_edge2_band, and red_band.
        red_edge_band: Name of the red-edge 1 (RE1) reflectance band.
        red_edge2_band: Name of the red-edge 2 (RE2) reflectance band.
        red_band: Name of the red reflectance band.
        output_band: Name of the output MTCI band (default 'MTCI').

    Returns:
        ee.Image with a single band named output_band containing MTCI values.

    ASI: MTCI (doi:10.1080/0143116042000274015). Sentinel-2 only (needs RE2).
    """
    return image.expression(
        "(re2 - re1) / (re1 - red)",
        {
            "re2": image.select(red_edge2_band),
            "re1": image.select(red_edge_band),
            "red": image.select(red_band),
        },
    ).rename(output_band)


def calc_ireci(
    image: ee.Image,
    red_edge_band: str,
    red_edge2_band: str,
    red_edge3_band: str,
    red_band: str,
    output_band: str = "IRECI",
) -> ee.Image:
    """Calculate the Inverted Red-Edge Chlorophyll Index.

    Args:
        image: ee.Image containing at least red_edge_band, red_edge2_band, red_edge3_band, and red_band.
        red_edge_band: Name of the red-edge 1 (RE1) reflectance band.
        red_edge2_band: Name of the red-edge 2 (RE2) reflectance band.
        red_edge3_band: Name of the red-edge 3 (RE3) reflectance band.
        red_band: Name of the red reflectance band.
        output_band: Name of the output IRECI band (default 'IRECI').

    Returns:
        ee.Image with a single band named output_band containing IRECI values.

    ASI: IRECI (doi:10.1016/j.isprsjprs.2013.04.007). Sentinel-2 only (needs RE2/RE3).
    """
    return image.expression(
        "(re3 - red) / (re1 / re2)",
        {
            "re3": image.select(red_edge3_band),
            "re2": image.select(red_edge2_band),
            "re1": image.select(red_edge_band),
            "red": image.select(red_band),
        },
    ).rename(output_band)


def calc_s2rep(
    image: ee.Image,
    red_edge_band: str,
    red_edge2_band: str,
    red_edge3_band: str,
    red_band: str,
    output_band: str = "S2REP",
) -> ee.Image:
    """Calculate the Sentinel-2 Red-Edge Position (in nanometres).

    Args:
        image: ee.Image containing at least red_edge_band, red_edge2_band, red_edge3_band, and red_band.
        red_edge_band: Name of the red-edge 1 (RE1) reflectance band.
        red_edge2_band: Name of the red-edge 2 (RE2) reflectance band.
        red_edge3_band: Name of the red-edge 3 (RE3) reflectance band.
        red_band: Name of the red reflectance band.
        output_band: Name of the output S2REP band (default 'S2REP').

    Returns:
        ee.Image with a single band named output_band containing the red-edge position in nm.

    ASI: S2REP (doi:10.1016/j.isprsjprs.2013.04.007). Sentinel-2 only (needs RE2/RE3).
    """
    return image.expression(
        "705.0 + 35.0 * ((((re3 + red) / 2.0) - re1) / (re2 - re1))",
        {
            "re3": image.select(red_edge3_band),
            "re2": image.select(red_edge2_band),
            "re1": image.select(red_edge_band),
            "red": image.select(red_band),
        },
    ).rename(output_band)


def calc_bais2(
    image: ee.Image,
    red_edge2_band: str,
    red_edge3_band: str,
    nir2_band: str,
    red_band: str,
    swir2_band: str,
    output_band: str = "BAIS2",
) -> ee.Image:
    """Calculate the Burned Area Index for Sentinel-2 (red-edge + SWIR2 burn index).

    Args:
        image: ee.Image containing at least red_edge2_band, red_edge3_band, nir2_band, red_band, and swir2_band.
        red_edge2_band: Name of the red-edge 2 (RE2) reflectance band.
        red_edge3_band: Name of the red-edge 3 (RE3) reflectance band.
        nir2_band: Name of the narrow-NIR (N2 / B8A) reflectance band.
        red_band: Name of the red reflectance band.
        swir2_band: Name of the SWIR2 reflectance band.
        output_band: Name of the output BAIS2 band (default 'BAIS2').

    Returns:
        ee.Image with a single band named output_band containing BAIS2 values.

    ASI: BAIS2 (doi:10.3390/ecrs-2-05177). Sentinel-2 only (needs RE2/RE3/N2). Expression
        parenthesisation is copied verbatim from the ASI catalogue.
    """
    return image.expression(
        "(1.0 - ((re2 * re3 * n2) / red) ** 0.5) "
        "* (((swir2 - n2) / (swir2 + n2) ** 0.5) + 1.0)",
        {
            "re2": image.select(red_edge2_band),
            "re3": image.select(red_edge3_band),
            "n2": image.select(nir2_band),
            "red": image.select(red_band),
            "swir2": image.select(swir2_band),
        },
    ).rename(output_band)


##################### Index registry ####################
@dataclass(frozen=True)
class IndexSpec:
    """Declarative spec for one spectral index, mirroring the ASI catalogue model.

    Attributes:
        name: Output band name (the ASI short_name).
        domain: Application domain ('vegetation', 'water', 'moisture', 'soil', 'burn', 'urban').
        func: The generic calc_* function that computes the index.
        band_args: Maps each band-name keyword of ``func`` to a logical band_map key, so the
            required band_map keys are derived (``bands``) and the function is invoked generically.
    """

    name: str
    domain: str
    func: Callable[..., ee.Image]
    band_args: dict[str, str]

    @property
    def bands(self) -> tuple[str, ...]:
        """Logical band_map keys this index requires."""
        return tuple(self.band_args.values())

    def compute(self, image: ee.Image, band_map: dict) -> ee.Image:
        """Compute the index on ``image`` using ``band_map`` to resolve band names."""
        kwargs = {arg: band_map[key] for arg, key in self.band_args.items()}
        return self.func(image, **kwargs)


# Single source of truth: name -> spec. Add a new index by adding a calc_* function
# above and one entry here; calc_indices, the default set, and band-availability
# filtering all derive from this. Insertion order is the output order for domain/default
# selections. kNDVI uses the sigma-RBF form (calc_kndvi_fixed_sigma); the data-driven
# sigma variants stay as separate collection-level helpers.
INDEX_REGISTRY: dict[str, IndexSpec] = {
    # Vegetation
    "NDVI": IndexSpec(
        "NDVI", "vegetation", calc_ndvi, {"nir_band": "nir", "red_band": "red"}
    ),
    "kNDVI_fixed": IndexSpec(
        "kNDVI_fixed",
        "vegetation",
        calc_kndvi_fixed_sigma,
        {"red_band": "red", "nir_band": "nir"},
    ),
    "Fpar": IndexSpec(
        "Fpar", "vegetation", calc_fpar, {"nir_band": "nir", "red_band": "red"}
    ),
    "EVI": IndexSpec(
        "EVI",
        "vegetation",
        calc_evi,
        {"nir_band": "nir", "red_band": "red", "blue_band": "blue"},
    ),
    "EVI2": IndexSpec(
        "EVI2", "vegetation", calc_evi2, {"nir_band": "nir", "red_band": "red"}
    ),
    "GNDVI": IndexSpec(
        "GNDVI", "vegetation", calc_gndvi, {"nir_band": "nir", "green_band": "green"}
    ),
    "OSAVI": IndexSpec(
        "OSAVI", "vegetation", calc_osavi, {"nir_band": "nir", "red_band": "red"}
    ),
    "SAVI": IndexSpec(
        "SAVI", "vegetation", calc_savi, {"nir_band": "nir", "red_band": "red"}
    ),
    "NIRv": IndexSpec(
        "NIRv", "vegetation", calc_nirv, {"nir_band": "nir", "red_band": "red"}
    ),
    "NDRE": IndexSpec(
        "NDRE",
        "vegetation",
        calc_ndre,
        {"nir_band": "nir", "red_edge_band": "red_edge"},
    ),
    "MTCI": IndexSpec(
        "MTCI",
        "vegetation",
        calc_mtci,
        {"red_edge_band": "red_edge", "red_edge2_band": "red_edge2", "red_band": "red"},
    ),
    "IRECI": IndexSpec(
        "IRECI",
        "vegetation",
        calc_ireci,
        {
            "red_edge_band": "red_edge",
            "red_edge2_band": "red_edge2",
            "red_edge3_band": "red_edge3",
            "red_band": "red",
        },
    ),
    "S2REP": IndexSpec(
        "S2REP",
        "vegetation",
        calc_s2rep,
        {
            "red_edge_band": "red_edge",
            "red_edge2_band": "red_edge2",
            "red_edge3_band": "red_edge3",
            "red_band": "red",
        },
    ),
    # Water
    "NDWI": IndexSpec(
        "NDWI", "water", calc_ndwi, {"green_band": "green", "nir_band": "nir"}
    ),
    "MNDWI": IndexSpec(
        "MNDWI", "water", calc_mndwi, {"green_band": "green", "swir1_band": "swir1"}
    ),
    # Moisture
    "NDMI": IndexSpec(
        "NDMI", "moisture", calc_ndmi, {"nir_band": "nir", "swir1_band": "swir1"}
    ),
    "GVMI": IndexSpec(
        "GVMI", "moisture", calc_gvmi, {"nir_band": "nir", "swir2_band": "swir2"}
    ),
    # Soil
    "BSI": IndexSpec(
        "BSI",
        "soil",
        calc_bsi,
        {
            "swir1_band": "swir1",
            "red_band": "red",
            "nir_band": "nir",
            "blue_band": "blue",
        },
    ),
    "MBI": IndexSpec(
        "MBI",
        "soil",
        calc_mbi,
        {"swir1_band": "swir1", "swir2_band": "swir2", "nir_band": "nir"},
    ),
    "EMBI": IndexSpec(
        "EMBI",
        "soil",
        calc_embi,
        {
            "swir1_band": "swir1",
            "swir2_band": "swir2",
            "nir_band": "nir",
            "green_band": "green",
        },
    ),
    "DBSI": IndexSpec(
        "DBSI",
        "soil",
        calc_dbsi,
        {
            "swir1_band": "swir1",
            "green_band": "green",
            "nir_band": "nir",
            "red_band": "red",
        },
    ),
    # Burn
    "NBR": IndexSpec(
        "NBR", "burn", calc_nbr, {"nir_band": "nir", "swir2_band": "swir2"}
    ),
    "NBR2": IndexSpec(
        "NBR2", "burn", calc_nbr2, {"swir1_band": "swir1", "swir2_band": "swir2"}
    ),
    "BAI": IndexSpec("BAI", "burn", calc_bai, {"red_band": "red", "nir_band": "nir"}),
    "BAIS2": IndexSpec(
        "BAIS2",
        "burn",
        calc_bais2,
        {
            "red_edge2_band": "red_edge2",
            "red_edge3_band": "red_edge3",
            "nir2_band": "nir2",
            "red_band": "red",
            "swir2_band": "swir2",
        },
    ),
    # Urban / built-up
    "NDBI": IndexSpec(
        "NDBI", "urban", calc_ndbi, {"swir1_band": "swir1", "nir_band": "nir"}
    ),
    "UI": IndexSpec("UI", "urban", calc_ui, {"swir2_band": "swir2", "nir_band": "nir"}),
}

# The set of valid domains, derived from the registry.
INDEX_DOMAINS: frozenset[str] = frozenset(
    spec.domain for spec in INDEX_REGISTRY.values()
)

# The harmonized core appended by default (preserves historical output + order). NDRE is
# in the list but only computes where the band map has 'red_edge' (Sentinel-2), so it is
# auto-included on S2 and auto-skipped on Landsat/HLS — no flag needed.
DEFAULT_INDICES: tuple[str, ...] = (
    "NDVI",
    "kNDVI_fixed",
    "Fpar",
    "EVI",
    "NDWI",
    "MNDWI",
    "SAVI",
    "NDMI",
    "NBR",
    "NIRv",
    "BSI",
    "NDRE",
)

# The smaller vegetation/water core used by calc_veg_indices.
VEG_CORE_INDICES: tuple[str, ...] = (
    "NDVI",
    "kNDVI_fixed",
    "Fpar",
    "EVI",
    "NDWI",
    "MNDWI",
    "SAVI",
)


##################### Generic multi-index helpers ####################
def resolve_index_names(
    band_map: dict,
    indices: list[str] | tuple[str, ...] | None = None,
    domains: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Resolve an index selection to the ordered list of names that can be computed.

    Selection semantics:
      * ``indices`` and ``domains`` both None -> the default harmonized core (DEFAULT_INDICES).
      * ``domains`` -> every registered index whose domain is requested (registry order).
      * ``indices`` -> those specific names (appended after any domain matches).
    Indices pulled in via the default set or via ``domains`` are silently skipped when the
    ``band_map`` lacks their required keys (so red-edge indices drop on non-S2 sensors).
    Indices named explicitly in ``indices`` must be computable: an unknown name, or one whose
    required band_map keys are absent, raises ValueError.

    Args:
        band_map: Dict mapping logical keys to band names; its keys determine availability.
        indices: Explicit index names to include (must be computable), or None.
        domains: Domains to include all registered indices for (skip-if-unavailable), or None.

    Returns:
        Ordered, de-duplicated list of index names to compute.

    Raises:
        ValueError: if a domain is unknown, an explicit index name is unknown, or an
            explicitly requested index requires band_map keys this sensor lacks.
    """
    available_keys = set(band_map)
    # (name, required) — required entries must compute or raise; others skip if unavailable.
    candidates: list[tuple[str, bool]] = []

    if indices is None and domains is None:
        candidates = [(name, False) for name in DEFAULT_INDICES]
    else:
        if domains is not None:
            wanted = set(domains)
            unknown = wanted - INDEX_DOMAINS
            if unknown:
                raise ValueError(
                    f"Unknown index domain(s): {sorted(unknown)}. "
                    f"Valid domains: {sorted(INDEX_DOMAINS)}."
                )
            candidates += [
                (name, False)
                for name, spec in INDEX_REGISTRY.items()
                if spec.domain in wanted
            ]
        if indices is not None:
            candidates += [(name, True) for name in indices]

    resolved: list[str] = []
    seen: set[str] = set()
    for name, required in candidates:
        if name in seen:
            continue
        spec = INDEX_REGISTRY.get(name)
        if spec is None:
            raise ValueError(
                f"Unknown index '{name}'. Available indices: {sorted(INDEX_REGISTRY)}."
            )
        missing = set(spec.bands) - available_keys
        if missing:
            if required:
                raise ValueError(
                    f"Index '{name}' requires band_map key(s) {sorted(missing)}, "
                    f"absent for this sensor (band_map keys: {sorted(available_keys)})."
                )
            continue
        seen.add(name)
        resolved.append(name)
    return resolved


def calc_indices(
    image: ee.Image,
    band_map: dict,
    indices: list[str] | tuple[str, ...] | None = None,
    domains: list[str] | tuple[str, ...] | None = None,
) -> ee.Image:
    """Append a selection of spectral indices to an image, driven by the index registry.

    With no selection the historical harmonized core is appended (NDVI, kNDVI_fixed, Fpar,
    EVI, NDWI, MNDWI, SAVI, NDMI, NBR, NIRv, BSI, and NDRE where the band map has a red
    edge). Pass ``indices`` to request specific indices by name, and/or ``domains`` to pull
    in whole families ('vegetation', 'water', 'moisture', 'soil', 'burn', 'urban'). See
    ``resolve_index_names`` for the exact selection and availability rules.

    Args:
        image: ee.Image containing the reflectance bands referenced by band_map.
        band_map: Dict mapping logical keys ('nir', 'red', 'blue', 'green', 'swir1', 'swir2',
            and for Sentinel-2 'red_edge', 'red_edge2', 'red_edge3', 'nir2') to band names.
        indices: Explicit index names to append (must be computable), or None for the default.
        domains: Index domains to append all registered members of, or None.

    Returns:
        ee.Image with the original bands plus the selected index bands appended.

    Raises:
        ValueError: propagated from resolve_index_names for unknown domains/indices or an
            explicitly requested index whose bands this sensor's band_map lacks.
    """
    names = resolve_index_names(band_map, indices=indices, domains=domains)
    index_bands = [INDEX_REGISTRY[name].compute(image, band_map) for name in names]
    # EE's addBands accepts a list of single-band images at runtime; the stub
    # only types a single image-like, so the list arg is flagged.
    return image.addBands(index_bands)  # type: ignore[arg-type]


def calc_veg_indices(image: ee.Image, band_map: dict) -> ee.Image:
    """Add the core vegetation and water indices (NDVI, kNDVI_fixed, Fpar, EVI, NDWI,
    MNDWI, SAVI) to an image.

    Args:
        image: ee.Image containing reflectance bands referenced by band_map.
        band_map: Dict mapping logical keys ('nir', 'red', 'blue', 'green', 'swir1') to band names in the image.

    Returns:
        ee.Image with the original bands plus NDVI, kNDVI_fixed, Fpar, EVI, NDWI, MNDWI, and SAVI appended.
    """
    return calc_indices(image, band_map, indices=VEG_CORE_INDICES)


def select_base_bands(
    image: ee.Image,
    input_bands: list[str],
    output_bands: list[str] | None = None,
) -> ee.Image:
    """Select base reflectance bands from an image, optionally renaming them for cross-
    sensor harmonization.

    Args:
        image: ee.Image from which to select bands.
        input_bands: List of band names to select.
        output_bands: Optional list of output names to rename the selected bands; must match length of input_bands if provided.

    Returns:
        ee.Image containing only the selected bands, renamed if output_bands was provided.
    """
    if output_bands is None:
        return image.select(input_bands)
    return image.select(input_bands, output_bands)
