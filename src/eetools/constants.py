# CHIRPS Precipitation Daily Reanalysis
CHIRPS_COLLECTION = "UCSB-CHC/CHIRPS/V3/DAILY_RNL"
CHIRPS_PRECIP_BAND = "precipitation"

# Sentinel-2 SR
S2_SR_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_PROB_COLLECTION = "COPERNICUS/S2_CLOUD_PROBABILITY"

# Landsat-8 SR
L8_SR_COLLECTION = "LANDSAT/LC08/C02/T1_L2"

# NASA HLS (Harmonized Landsat Sentinel)
HLS_L30_COLLECTION = "NASA/HLS/HLSL30/v002"  # Landsat OLI
HLS_S30_COLLECTION = "NASA/HLS/HLSS30/v002"  # Sentinel MSI

# MODIS LAI / Fpar
MODIS_LAI_FPAR_COLLECTION = "MODIS/061/MCD15A3H"

# ESA WorldCover
ESA_WC_COLLECTION = "ESA/WorldCover/v200"

#################### SENTINEL-2 ##########################
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_ALL_BANDS = [
    "B2",   # BLUE
    "B3",   # GREEN
    "B4",   # RED
    "B5",   # RE1
    "B6",   # RE2
    "B7",   # RE3
    "B8",   # NIR
    "B8A",  # NARROW_NIR
    "B11",  # SWIR1
    "B12",  # SWIR2
    "NDVI", "kNDVI_fixed", "Fpar", "EVI", "NDWI", "MNDWI", "SAVI",
    "NDMI", "NBR", "NIRv", "NDRE",
]
S2_INDEX_BANDS = [
    "NDVI", "kNDVI_fixed", "Fpar", "EVI", "NDWI", "MNDWI",
    "SAVI", "NDMI", "NBR", "NIRv", "NDRE",
]
S2_BAND_MAP = {
    "blue": "B2",
    "green": "B3",
    "red": "B4",
    "red_edge": "B5",
    "nir": "B8",
    "swir1": "B11",
    "swir2": "B12",
}
S2_SCALE_FACTOR = 0.0001
S2_SCALE = 10

# Sentinel-2 cloud masking parameters
CLOUD_FILTER = 50        # % max CLOUDY_PIXEL_PERCENTAGE per image
CLD_PRB_THRESH = 40      # % s2cloudless probability threshold
NIR_DRK_THRESH = 0.15    # reflectance threshold for dark (shadow candidate) pixels
CLD_PRJ_DIST_KM = 1.0   # max shadow search distance in km
BUFFER_M = 50            # dilation buffer for cloud+shadow mask in metres
ERODE_RADIUS_M = 40      # erosion radius to denoise speckle in metres
DDT_SCALE_M = 100        # scale for directionalDistanceTransform in metres
MORPH_SCALE_M = 20       # scale for focal morphology ops in metres

#################### HLS ##########################
HLS_SCALE_FACTOR = 0.0001
HLS_ADD_OFFSET = 0.0

HLS_COMMON_BANDS = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
HLS_L30_SOURCE_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]
HLS_S30_SOURCE_BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]

HLS_ALL_BANDS = [
    "BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2",
    "NDVI", "kNDVI_fixed", "Fpar", "EVI", "NDWI", "MNDWI",
    "SAVI", "NDMI", "NBR", "NIRv",
]
HLS_INDEX_BANDS = [
    "NDVI", "kNDVI_fixed", "Fpar", "EVI", "NDWI", "MNDWI",
    "SAVI", "NDMI", "NBR", "NIRv",
]
HLS_BAND_MAP = {
    "blue": "BLUE",
    "green": "GREEN",
    "red": "RED",
    "nir": "NIR",
    "swir1": "SWIR1",
    "swir2": "SWIR2",
}

# HLS Fmask QA controls
HLS_CLOUD_FILTER = 50
HLS_MASK_ADJACENT = True
HLS_MASK_SNOW = True
HLS_MASK_WATER_IN_QA = False   # keep False to use separate spectral water masking
HLS_MASK_MODERATE_AEROSOL = False
HLS_MASK_HIGH_AEROSOL = True

HLS_SCALE = 30

#################### LANDSAT-8 ##########################
L8_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
L8_ALL_BANDS: list[str] = []
L8_INDEX_BANDS = [
    "NDVI", "kNDVI_fixed", "Fpar", "EVI", "NDWI", "MNDWI",
    "SAVI", "NDMI", "NBR", "NIRv",
]
L8_BAND_MAP = {
    "blue": "SR_B2",
    "green": "SR_B3",
    "red": "SR_B4",
    "nir": "SR_B5",
    "swir1": "SR_B6",
    "swir2": "SR_B7",
}

# Landsat Collection 2 Level-2 SR scaling
L8_SCALE_FACTOR = 0.0000275
L8_ADD_OFFSET = -0.2
L8_CLOUD_FILTER = 50  # % max CLOUD_COVER per image
L8_SCALE = 30

#################### MODIS LAI/Fpar ##########################
MODIS_LAI_FPAR_BANDS = [
    "Fpar",
    "Lai",
    "FparStdDev",
    "LaiStdDev",
    "FparLai_QC",
    "FparExtra_QC",
]
FPAR_SCALE_FACTOR = 0.01
LAI_SCALE_FACTOR = 0.1
FPAR_STDDEV_SCALE_FACTOR = 0.01
LAI_STDDEV_SCALE_FACTOR = 0.1

#################### ESA WorldCover ##########################
ESA_BAND = "Map"

#################### GLOBAL DEFAULTS ##########################
SIGMA = 0.15            # default kNDVI sigma

DEFAULT_MAX_PIXELS = 1e13
DEFAULT_FILE_FORMAT = "GeoTIFF"
DEFAULT_CRS = "EPSG:4326"
