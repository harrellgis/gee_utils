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

# Copernicus DEM GLO-30 — canonical elevation source for all terrain work
COPERNICUS_DEM_COLLECTION = "COPERNICUS/DEM/GLO30"

# Biodiversity Intactness Index (BII) — sub-Saharan Africa (bii4africa, sat-io)
_BII_ROOT = "projects/earthengine-legacy/assets/projects/sat-io/open-datasets/BII"
BII_1KM_COLLECTION = f"{_BII_ROOT}/BII_1km"
BII_8KM_COLLECTION = f"{_BII_ROOT}/BII_8km"
BII_MASK_ASSET = f"{_BII_ROOT}/BII_Mask"

#################### SPECTRAL-INDEX BAND CONVENTION ##########################
# Awesome Spectral Indices (ASI; Montero et al. 2023, doi:10.1038/s41597-023-02096-0)
# standardized band letters for each logical band_map key. Every *_BAND_MAP below maps
# these same logical keys to a sensor's physical band names, so the generic index
# functions in sensors/indices.py are sensor-agnostic. Documented for cross-tool
# portability (e.g. spyndex/eemont use these letters directly).
ASI_BAND_LETTERS = {
    "blue": "B",
    "green": "G",
    "red": "R",
    "red_edge": "RE1",
    "nir": "N",
    "swir1": "S1",
    "swir2": "S2",
}

#################### SENTINEL-2 ##########################
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_ALL_BANDS = [
    "B2",  # BLUE
    "B3",  # GREEN
    "B4",  # RED
    "B5",  # RE1
    "B6",  # RE2
    "B7",  # RE3
    "B8",  # NIR
    "B8A",  # NARROW_NIR
    "B11",  # SWIR1
    "B12",  # SWIR2
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
]
S2_INDEX_BANDS = [
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
]
S2_BAND_MAP = {
    "blue": "B2",
    "green": "B3",
    "red": "B4",
    "red_edge": "B5",  # RE1
    "red_edge2": "B6",  # RE2 — enables MTCI/IRECI/S2REP/BAIS2 (Sentinel-2 only)
    "red_edge3": "B7",  # RE3
    "nir": "B8",
    "nir2": "B8A",  # N2 (narrow NIR) — used by BAIS2
    "swir1": "B11",
    "swir2": "B12",
}
S2_SCALE_FACTOR = 0.0001
S2_SCALE = 10

# Sentinel-2 cloud masking parameters
CLOUD_FILTER = 50  # % max CLOUDY_PIXEL_PERCENTAGE per image
CLD_PRB_THRESH = 40  # % s2cloudless probability threshold
NIR_DRK_THRESH = 0.15  # reflectance threshold for dark (shadow candidate) pixels
CLD_PRJ_DIST_KM = 1.0  # max shadow search distance in km
BUFFER_M = 50  # dilation buffer for cloud+shadow mask in metres
ERODE_RADIUS_M = 40  # erosion radius to denoise speckle in metres
DDT_SCALE_M = 100  # scale for directionalDistanceTransform in metres
MORPH_SCALE_M = 20  # scale for focal morphology ops in metres

#################### HLS ##########################
HLS_SCALE_FACTOR = 0.0001
HLS_ADD_OFFSET = 0.0

HLS_COMMON_BANDS = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
HLS_L30_SOURCE_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]
HLS_S30_SOURCE_BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]

HLS_ALL_BANDS = [
    "BLUE",
    "GREEN",
    "RED",
    "NIR",
    "SWIR1",
    "SWIR2",
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
]
HLS_INDEX_BANDS = [
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
HLS_MASK_WATER_IN_QA = False  # keep False to use separate spectral water masking
HLS_MASK_MODERATE_AEROSOL = False
HLS_MASK_HIGH_AEROSOL = True

HLS_SCALE = 30

#################### LANDSAT-8 ##########################
L8_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
L8_INDEX_BANDS = [
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
]
# Landsat keeps its native SR_B* band names (unlike HLS, which renames), so the
# full processed band set is the base bands followed by the appended indices.
L8_ALL_BANDS = L8_BANDS + L8_INDEX_BANDS
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

#################### LANDSAT 5 / 7 / 9 (C2 L2 SR) ##########################
# All Landsat Collection 2 Level-2 SR share the same scaling, offset, default cloud
# filter, QA_PIXEL/QA_RADSAT cloud masking, and 30 m resolution as Landsat 8.
L9_SR_COLLECTION = "LANDSAT/LC09/C02/T1_L2"  # OLI-2; schema-identical to Landsat 8
L7_SR_COLLECTION = "LANDSAT/LE07/C02/T1_L2"  # ETM+ (SLC-off after 2003-05-31)
L5_SR_COLLECTION = "LANDSAT/LT05/C02/T1_L2"  # TM (archive 1984-2012)

LANDSAT_C2_SCALE_FACTOR = L8_SCALE_FACTOR  # 2.75e-05, shared by all Landsat C2 L2 SR
LANDSAT_C2_ADD_OFFSET = L8_ADD_OFFSET  # -0.2 (non-zero offset!)
LANDSAT_C2_CLOUD_FILTER = L8_CLOUD_FILTER  # 50% max CLOUD_COVER per image
LANDSAT_SCALE = L8_SCALE  # 30 m

# Landsat 9 (OLI) is band-for-band identical to Landsat 8 (OLI).
L9_BANDS = L8_BANDS
L9_BAND_MAP = L8_BAND_MAP
L9_INDEX_BANDS = L8_INDEX_BANDS
L9_ALL_BANDS = L8_ALL_BANDS

# Landsat 5 (TM) and Landsat 7 (ETM+) share the TM/ETM+ band layout: reflective bands
# numbered 1-5,7 with NO coastal-aerosol band, so every band is SHIFTED down by one
# relative to OLI (TM/ETM+ NIR=SR_B4, Red=SR_B3, SWIR1=SR_B5 vs OLI NIR=SR_B5...).
TM_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
TM_BAND_MAP = {
    "blue": "SR_B1",
    "green": "SR_B2",
    "red": "SR_B3",
    "nir": "SR_B4",
    "swir1": "SR_B5",
    "swir2": "SR_B7",
}
TM_INDEX_BANDS = L8_INDEX_BANDS  # same index set (no red-edge band -> no NDRE)
TM_ALL_BANDS = TM_BANDS + TM_INDEX_BANDS

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

#################### Copernicus DEM GLO-30 ##########################
# Canonical elevation datasource for all Earth Engine terrain work. The GLO30
# product is a tiled ImageCollection of a Digital Surface Model; its source
# elevation band is "DEM" (metres above the EGM2008 geoid). It is mosaicked and
# pinned to its native projection before terrain analysis, and the elevation band
# is renamed to "elevation" so it can be passed directly to ee.Terrain.
COPERNICUS_DEM_BAND = "DEM"
ELEVATION_BAND = "elevation"
# Bands produced by ee.Terrain.products on the elevation image (elevation retained).
TERRAIN_BANDS = ["elevation", "slope", "aspect", "hillshade"]

#################### Biodiversity Intactness Index (BII) ##########################
# Per-resolution band order as the asset ingests (drives the toBands() rename).
# The 1km and 8km collections order their bands differently.
BII_1KM_BANDS = [
    "Land Use",
    "Land Use Intensity",
    "BII All",
    "BII Amphibians",
    "BII Birds",
    "BII Forbs",
    "BII Graminoids",
    "BII Mammals",
    "BII All Plants",
    "BII Reptiles",
    "BII Trees",
    "BII All Vertebrates",
]
BII_8KM_BANDS = [
    "BII All",
    "BII Amphibians",
    "BII Birds",
    "BII Forbs",
    "BII Graminoids",
    "BII Mammals",
    "BII All Plants",
    "BII Reptiles",
    "BII Trees",
    "BII All Vertebrates",
    "Land Use",
    "Land Use Intensity",
]
# The per-taxon BII bands (proportion of intact populations, 0-1).
BII_TAXON_BANDS = [
    "BII All",
    "BII Amphibians",
    "BII Birds",
    "BII Forbs",
    "BII Graminoids",
    "BII Mammals",
    "BII All Plants",
    "BII Reptiles",
    "BII Trees",
    "BII All Vertebrates",
]
BII_LAND_USE_BAND = "Land Use"
BII_LAND_USE_INTENSITY_BAND = "Land Use Intensity"
# Output band order after preprocessing: taxon BII bands then the land-use bands.
BII_PROCESSED_BANDS = BII_TAXON_BANDS + [BII_LAND_USE_BAND, BII_LAND_USE_INTENSITY_BAND]
# Land Use classes excluded from the Land Use Intensity mask (per the source script).
BII_EXCLUDED_LAND_USE_CLASSES = [2, 5]

#################### Hansen Global Forest Change ##########################
# Single static multiband ee.Image (Landsat-derived forest extent/change), 30.92 m.
HANSEN_GFC_COLLECTION = "UMD/hansen/global_forest_change_2025_v1_13"
HANSEN_TREECOVER_BAND = "treecover2000"  # canopy cover % for year 2000 (0-100)
HANSEN_LOSS_BAND = "loss"  # binary stand-replacement loss over the study period
HANSEN_GAIN_BAND = "gain"  # binary non-forest -> forest gain (2000-2012 only)
HANSEN_LOSSYEAR_BAND = "lossyear"  # year of loss, encoded 1-25 (see epoch below)
# Default canopy-cover % threshold defining "forest" when masking loss products.
HANSEN_TREE_COVER_THRESHOLD = 10
# lossyear encodes 1-25 for 2001-2025; absolute year = HANSEN_LOSSYEAR_EPOCH + value.
HANSEN_LOSSYEAR_EPOCH = 2000
HANSEN_LOSSYEAR_MAX = 25  # highest lossyear code in v1.13 (-> 2025)

#################### SENTINEL-1 SAR GRD ##########################
# C-band dual-pol Ground Range Detected, log-scaled to dB. Heterogeneous time
# series (mixed polarizations / modes / resolutions / orbit passes), so the
# collection builder always filters to a homogeneous set before use. No spectral
# indices (SAR backscatter, not reflectance); speckle filtering is not applied by
# the source and is added here.
S1_GRD_COLLECTION = "COPERNICUS/S1_GRD"
# Common IW land configuration: dual cross-pol VV+VH. Cross-pol (VH) is the most
# informative for vegetation/water discrimination.
S1_DEFAULT_POLARIZATIONS = ["VV", "VH"]
S1_DEFAULT_INSTRUMENT_MODE = "IW"  # Interferometric Wide — the standard land mode
# ASCENDING and DESCENDING look geometries differ and must never be mixed in a
# time series; the builder pins one pass (override per analysis).
S1_DEFAULT_ORBIT_PASS = "DESCENDING"
# Scene-edge pixels carry very low backscatter (terrain-correction border noise);
# pixels below this dB value are masked.
S1_EDGE_THRESHOLD_DB = -30.0
# Focal-median speckle-filter neighbourhood radius (median is unbiased in dB).
S1_SPECKLE_RADIUS_M = 50
S1_SCALE = 10

#################### OPERA DSWx (Dynamic Surface Water Extent) ##########################
# Pre-classified surface-water products (no spectral indexing). Shared
# WTR/BWTR/CONF/DIAG band schema on a 30 m MGRS grid; HLS (optical) and S1 (radar)
# differ in their invalid/mask class codes — see the per-product valid-max below.
DSWX_HLS_COLLECTION = "OPERA/DSWX/L3_V1/HLS"  # optical (Harmonized Landsat/Sentinel-2)
DSWX_S1_COLLECTION = "OPERA/DSWX/L3_V1/S1"  # radar (Sentinel-1), cloud-independent
DSWX_WTR_BAND = "WTR_Water_classification"  # primary water classification
DSWX_BWTR_BAND = "BWTR_Binary_water"  # binary water (1) / not-water (0)
DSWX_CONF_BAND = "CONF_Confidence"
DSWX_DIAG_BAND = "DIAG_diagnostic"
# Shared water-class codes.
DSWX_OPEN_WATER = 1
DSWX_PARTIAL_SURFACE_WATER = 2  # HLS only — subpixel inundation (wetlands/coastline)
DSWX_INUNDATED_VEGETATION = 3  # S1 only — high dual-pol ratio + wetland land cover
# First invalid class code per product; pixels with WTR >= this are sensor masks,
# not water. HLS: 252 snow / 253 cloud / 254 ocean. S1: 250 HAND / 251
# layover-shadow / 254 ocean (the catalog sample code wrongly reuses 252 for S1).
DSWX_HLS_VALID_MAX = 252
DSWX_S1_VALID_MAX = 250
DSWX_SCALE = 30

#################### DYNAMIC WORLD (near-real-time LULC) ##########################
# Per-Sentinel-2-acquisition 10 m LULC: nine class-probability bands (sum to 1 per pixel)
# plus a `label` argmax band (integer class code 0-8). Time series; cloud masking is built
# in (no extra mask needed). The per-acquisition `label` is unstable — composite the
# probability bands for stable LULC / landscape metrics rather than using a single label.
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
DW_LABEL_BAND = "label"
# Probability bands, ordered so list index == the `label` integer code for that class.
DW_PROBABILITY_BANDS = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]
# `label` integer code -> class name (argmax of the probability bands).
DW_CLASSES = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded_vegetation",
    4: "crops",
    5: "shrub_and_scrub",
    6: "built",
    7: "bare",
    8: "snow_and_ice",
}
DW_SCALE = 10

#################### WDPA (World Database on Protected Areas) ##########################
# Vector FeatureCollection of protected-area polygons (UNEP-WCMC / IUCN), updated monthly.
# `current` always resolves to the newest monthly release; pin a YYYYMM snapshot
# (e.g. "WCMC/WDPA/202606/polygons") for reproducible analysis.
# LICENSE: No Commercial Use without prior written UNEP-WCMC permission (commercial =
# any for-profit OR revenue-generating use, incl. by non-profits), and attribution is
# mandatory — flag for any paid/client deliverable.
WDPA_POLYGONS_COLLECTION = "WCMC/WDPA/current/polygons"
WDPA_COUNTRY_FIELD = "ISO3"  # ISO 3166-3 alpha-3 country code (e.g. "BWA", "KEN")
WDPA_ID_FIELD = (
    "SITE_ID"  # whole-site id in the live `current` asset (= WDPAID in older
)
# WDPA releases); SITE_PID is the per-parcel id (= the former WDPA_PID).
WDPA_GIS_AREA_FIELD = (
    "GIS_AREA"  # km^2, Mollweide-computed — use for area, NOT REP_AREA
)

#################### LANDTRENDR (temporal segmentation) ##########################
# LandTrendr runs on ONE annual image per year (medoid composite), band 1 = a
# loss-positive segmentation index, subsequent bands fit-to-vertices (FTV). All sensors
# are harmonized to a common band naming and the OLI family (L8/L9) is brought onto the
# TM/ETM+ (L5/L7) reflectance baseline before indexing.
LANDTRENDR_COMMON_BANDS = HLS_COMMON_BANDS  # BLUE, GREEN, RED, NIR, SWIR1, SWIR2

# Roy et al. 2016 OLI -> ETM+ harmonization, per common band (BLUE,GREEN,RED,NIR,SWIR1,
# SWIR2), replicated from the emaprlab LandTrendr.js `harmonizationRoy`. Applied in C2
# reflectance space as:  etm = (oli - intercept) / slope.
ROY_OLI_TO_ETM_SLOPES = [0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949]
ROY_OLI_TO_ETM_INTERCEPTS = [-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029]

# The 8 LandTrendr run parameters (Kennedy et al. 2010). `timeSeries` is attached at run
# time. Tune per landscape; these are sensible disturbance-mapping defaults.
LANDTRENDR_DEFAULT_RUN_PARAMS = {
    "maxSegments": 6,
    "spikeThreshold": 0.9,
    "vertexCountOvershoot": 3,
    "preventOneYearRecovery": True,
    "recoveryThreshold": 0.25,
    "pvalThreshold": 0.05,
    "bestModelProportion": 0.75,
    "minObservationsNeeded": 6,
}

# Per-sensor config the annual builder iterates over: (collection ID, reflective bands,
# is_oli). OLI sensors are Roy-harmonized; TM/ETM+ are the baseline.
LANDTRENDR_SENSORS = {
    "L5": (L5_SR_COLLECTION, TM_BANDS, False),
    "L7": (L7_SR_COLLECTION, TM_BANDS, False),
    "L8": (L8_SR_COLLECTION, L8_BANDS, True),
    "L9": (L9_SR_COLLECTION, L9_BANDS, True),
}

# Normalized indices supported as the segmentation index. Loss = NEGATIVE delta for these,
# so they are multiplied by -1 to make loss-positive for segmentation (dist_dir = -1).
LANDTRENDR_NORMALIZED_INDICES = ("NBR", "NDVI", "NDMI")

#################### GLOBAL DEFAULTS ##########################
SIGMA = 0.15  # default kNDVI sigma

DEFAULT_MAX_PIXELS = 1e13
DEFAULT_FILE_FORMAT = "GeoTIFF"
DEFAULT_CRS = "EPSG:4326"
