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


##################### Generic multi-index helpers ####################
def calc_veg_indices(image: ee.Image, band_map: dict) -> ee.Image:
    """Add the core vegetation and water indices (NDVI, kNDVI_fixed, Fpar, EVI, NDWI,
    MNDWI, SAVI) to an image.

    Args:
        image: ee.Image containing reflectance bands referenced by band_map.
        band_map: Dict mapping logical keys ('nir', 'red', 'blue', 'green', 'swir1') to band names in the image.

    Returns:
        ee.Image with the original bands plus NDVI, kNDVI_fixed, Fpar, EVI, NDWI, MNDWI, and SAVI appended.
    """
    index_bands = [
        calc_ndvi(image, nir_band=band_map["nir"], red_band=band_map["red"]),
        calc_kndvi_fixed_sigma(
            image, red_band=band_map["red"], nir_band=band_map["nir"]
        ),
        calc_fpar(image, nir_band=band_map["nir"], red_band=band_map["red"]),
        calc_evi(
            image,
            nir_band=band_map["nir"],
            red_band=band_map["red"],
            blue_band=band_map["blue"],
        ),
        calc_ndwi(image, green_band=band_map["green"], nir_band=band_map["nir"]),
        calc_mndwi(image, green_band=band_map["green"], swir1_band=band_map["swir1"]),
        calc_savi(image, nir_band=band_map["nir"], red_band=band_map["red"]),
    ]
    # EE's addBands accepts a list of single-band images at runtime; the stub
    # only types a single image-like, so the list arg is flagged.
    return image.addBands(index_bands)  # type: ignore[arg-type]


def calc_indices(
    image: ee.Image,
    band_map: dict,
    include_ndre: bool = False,
) -> ee.Image:
    """Add the full index set (NDVI, kNDVI_fixed, Fpar, EVI, NDWI, MNDWI, SAVI, NDMI,
    NBR, NIRv, BSI, optionally NDRE) to an image.

    Args:
        image: ee.Image containing reflectance bands referenced by band_map.
        band_map: Dict mapping logical keys ('nir', 'red', 'blue', 'green', 'swir1', 'swir2', and optionally 'red_edge') to band names in the image.
        include_ndre: If True, also appends the NDRE band; requires 'red_edge' in band_map (default False).

    Returns:
        ee.Image with the original bands plus all computed index bands appended.
    """
    index_bands = [
        calc_ndvi(image, nir_band=band_map["nir"], red_band=band_map["red"]),
        calc_kndvi_fixed_sigma(
            image, red_band=band_map["red"], nir_band=band_map["nir"]
        ),
        calc_fpar(image, nir_band=band_map["nir"], red_band=band_map["red"]),
        calc_evi(
            image,
            nir_band=band_map["nir"],
            red_band=band_map["red"],
            blue_band=band_map["blue"],
        ),
        calc_ndwi(image, green_band=band_map["green"], nir_band=band_map["nir"]),
        calc_mndwi(image, green_band=band_map["green"], swir1_band=band_map["swir1"]),
        calc_savi(image, nir_band=band_map["nir"], red_band=band_map["red"]),
        calc_ndmi(image, nir_band=band_map["nir"], swir1_band=band_map["swir1"]),
        calc_nbr(image, nir_band=band_map["nir"], swir2_band=band_map["swir2"]),
        calc_nirv(image, nir_band=band_map["nir"], red_band=band_map["red"]),
        calc_bsi(
            image,
            swir1_band=band_map["swir1"],
            red_band=band_map["red"],
            nir_band=band_map["nir"],
            blue_band=band_map["blue"],
        ),
    ]

    if include_ndre:
        index_bands.append(
            calc_ndre(
                image, nir_band=band_map["nir"], red_edge_band=band_map["red_edge"]
            )
        )

    # EE's addBands accepts a list of single-band images at runtime; the stub
    # only types a single image-like, so the list arg is flagged.
    return image.addBands(index_bands)  # type: ignore[arg-type]


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
