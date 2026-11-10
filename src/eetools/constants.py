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

# MODIS vegetation indices, burned area, NPP, and vegetation continuous fields
MODIS_VI_COLLECTION = "MODIS/061/MOD13Q1"  # 250 m, 16-day NDVI/EVI
MODIS_BURNED_AREA_COLLECTION = "MODIS/061/MCD64A1"  # 500 m, monthly
MODIS_NPP_COLLECTION = "MODIS/061/MOD17A3HGF"  # 500 m, annual (gap-filled)
MODIS_VCF_COLLECTION = "MODIS/061/MOD44B"  # 250 m, annual

# NASA SMAP L4 Global 3-hourly 9 km soil moisture (model-assimilated)
SMAP_L4_COLLECTION = "NASA/SMAP/SPL4SMGP/008"

# OpenLandMap SM2RAIN-based monthly precipitation climatology (single ee.Image)
SM2RAIN_CLIMATOLOGY_IMAGE = "OpenLandMap/CLM/CLM_PRECIPITATION_SM2RAIN_M/v01"

# NASA Black Marble VIIRS VNP46A2 v002 daily nighttime lights. Supersedes the
# NOAA/VIIRS/001/VNP46A1 / VNP46A2 v001 copies, which stopped updating in Jan 2025.
VIIRS_NTL_COLLECTION = "NASA/VIIRS/002/VNP46A2"

# ESA WorldCover
ESA_WC_COLLECTION = "ESA/WorldCover/v200"

# Copernicus DEM GLO-30 (2024_1) — canonical elevation source for all terrain work.
# COPERNICUS/DEM/GLO30 was deprecated in favor of this dataset.
COPERNICUS_DEM_COLLECTION = "COPERNICUS/DEM/GLO30_2024_1"

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
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_ALL_BANDS = [
    "B1",  # AEROSOL
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

#################### MODIS MOD13Q1 Vegetation Indices ##########################
# NDVI/EVI already carry the eetools index band names, so no rename is needed.
MODIS_VI_SCALED_BANDS = [
    "NDVI",
    "EVI",
    "sur_refl_b01",  # red
    "sur_refl_b02",  # nir
    "sur_refl_b03",  # blue
    "sur_refl_b07",  # swir2 (2130 nm)
]
MODIS_VI_QA_BANDS = ["SummaryQA", "DetailedQA"]
MODIS_VI_BANDS = MODIS_VI_SCALED_BANDS + MODIS_VI_QA_BANDS
MODIS_VI_SCALE_FACTOR = 0.0001
# SummaryQA pixel reliability: 0 good, 1 marginal, 2 snow/ice, 3 cloudy (-1 fill).
# Keeping <= 1 retains good + marginal retrievals.
MODIS_VI_MAX_SUMMARY_QA = 1

#################### MODIS MCD64A1 Burned Area ##########################
# BurnDate is the burn day-of-year (1-366); GEE masks it over unburned land.
MODIS_BURNED_AREA_BANDS = ["BurnDate", "Uncertainty", "QA", "FirstDay", "LastDay"]
MODIS_BURN_DATE_BAND = "BurnDate"
MODIS_BURNED_BAND = "burned"  # derived 0/1 flag
# QA bit 0 = land, bit 1 = valid data; both must be set for a usable observation.
MODIS_BURNED_AREA_VALID_QA_BITS = 0b11

#################### MODIS MOD17A3HGF Annual NPP ##########################
MODIS_NPP_BANDS = ["Npp", "Gpp", "Npp_QC"]
MODIS_NPP_SCALED_BANDS = ["Npp", "Gpp"]
MODIS_NPP_SCALE_FACTOR = 0.0001  # -> kg C / m^2 / year
# Upper valid-range limits; values above are fill codes (e.g. Gpp 65534 = water).
MODIS_NPP_VALID_MAX = 32700
MODIS_GPP_VALID_MAX = 30000

#################### MODIS MOD44B Vegetation Continuous Fields ##########################
MODIS_VCF_BANDS = [
    "Percent_Tree_Cover",
    "Percent_NonTree_Vegetation",
    "Percent_NonVegetated",
]
# Cover bands are percentages; 200 = water and 253 = fill are masked out.
MODIS_VCF_VALID_MAX = 100

#################### NASA SMAP L4 Soil Moisture ##########################
# SPL4SMGP is 3-hourly (8 images/day), ~9 km (EASE-Grid 2.0 M09, served in EPSG:4326),
# already in physical units (no scale factor) and masked over open water. It has
# no per-pixel QA band. Volumetric moisture is m^3/m^3; wetness is the fraction
# of saturation (0-1). Surface = 0-5 cm, rootzone = 0-100 cm, profile = surface to
# bedrock.
SMAP_SM_BANDS = [
    "sm_surface",
    "sm_rootzone",
    "sm_profile",
    "sm_surface_wetness",
    "sm_rootzone_wetness",
    "sm_profile_wetness",
]

#################### OpenLandMap SM2RAIN Precipitation Climatology ##########################
# Long-term (2007-2018) mean monthly precipitation in mm/month at 1 km, from an
# ensemble of SM2RAIN-ASCAT, GPM IMERG, WorldClim v2, and CHELSA. One band per
# calendar month (no time series, no scale factor, water not masked).
SM2RAIN_MONTH_BANDS = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]
SM2RAIN_PRECIP_BAND = "precipitation"  # matches CHIRPS_PRECIP_BAND

#################### NASA Black Marble VIIRS VNP46A2 Nighttime Lights ##########################
# Daily, 15 arc-second (~500 m), already in physical units (nW/cm^2/sr; no scale
# factor). The BRDF/lunar-corrected NTL band is masked by GEE where the retrieval
# is poor/cloudy; the gap-filled band fills those pixels from the latest
# high-quality retrieval, whose age in days is Latest_High_Quality_Retrieval.
VIIRS_NTL_BAND = "DNB_BRDF_Corrected_NTL"
VIIRS_NTL_GAP_FILLED_BAND = "Gap_Filled_DNB_BRDF_Corrected_NTL"
VIIRS_NTL_BANDS = [
    VIIRS_NTL_BAND,
    VIIRS_NTL_GAP_FILLED_BAND,
    "Latest_High_Quality_Retrieval",
    "Mandatory_Quality_Flag",
    "Snow_Flag",
    "QF_Cloud_Mask",
]
# Mandatory_Quality_Flag: 0 high-quality persistent, 1 high-quality ephemeral
# (e.g. fires, gas flares), 2 poor quality, 255 fill. <= 1 keeps both high-quality
# classes.
VIIRS_NTL_MAX_QUALITY_FLAG = 1

#################### ESA WorldCover ##########################
ESA_BAND = "Map"
# WorldCover v200 class codes -> clean lowercase names, for area-by-class summaries.
ESA_CLASS_MAP = {
    "tree_cover": 10,
    "shrubland": 20,
    "grassland": 30,
    "cropland": 40,
    "built_up": 50,
    "bare_sparse_vegetation": 60,
    "snow_ice": 70,
    "permanent_water_bodies": 80,
    "herbaceous_wetland": 90,
    "mangroves": 95,
    "moss_lichen": 100,
}

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
BII_PROCESSED_BANDS = BII_TAXON_BANDS + [
    BII_LAND_USE_BAND,
    BII_LAND_USE_INTENSITY_BAND,
]
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

#################### iSDA Africa Total Soil Carbon ##########################
# Single static multiband ee.Image (30m, Africa-only). "mean_0_20"/"mean_20_50" are
# predicted total carbon at 0-20cm/20-50cm depth; "stdev_*" are the paired uncertainty
# bands. Baseline assessments use the topsoil (0-20cm) mean band.
ISDA_CARBON_COLLECTION = "ISDASOIL/Africa/v1/carbon_total"
ISDA_CARBON_TOPSOIL_BAND = "mean_0_20"

#################### Meta/WRI 1m Global Canopy Height ##########################
# Tiled ee.ImageCollection (sat-io); mosaic() for wall-to-wall coverage. Single band,
# canopy height in whole metres (confirmed via .first().bandNames() -- undocumented in
# the catalog entry). Fused 2009-2020 baseline (~80% 2018-2020 imagery), not a dated
# annual layer -- do not difference against another epoch to infer change.
META_CANOPY_HEIGHT_COLLECTION = (
    "projects/sat-io/open-datasets/facebook/meta-canopy-height"
)
META_CANOPY_HEIGHT_BAND = "cover_code"

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
# Timescan: per-pixel temporal statistics over a (seasonal) VV+VH stack. Each input
# band is reduced with every statistic, giving "<band>_<stat>" output bands.
# VVVH_ratio / VVVH_ND_ratio are per-image VV/VH and (VV-VH)/(VV+VH) in linear power.
S1_TIMESCAN_INPUT_BANDS = ["VV", "VH", "VVVH_ratio", "VVVH_ND_ratio"]
S1_TIMESCAN_STATS = ["min", "max", "mean", "stdDev", "median"]
# Full timescan band set and order (matches the GEE book A3.2 mangrove timescans):
# the per-band statistics, then median VV-vs-VH comparisons (dB), then the
# coefficients of variation (linear power) and their normalized difference.
S1_TIMESCAN_BANDS = [
    f"{band}_{stat}" for band in S1_TIMESCAN_INPUT_BANDS for stat in S1_TIMESCAN_STATS
] + [
    "VV_median_VH_median_nat",  # VV_median - VH_median (dB)
    "VV_median_VH_median_db",  # VV_median / VH_median (dB)
    "VV_CV",  # stdDev / mean of VV in linear power
    "VH_CV",  # stdDev / mean of VH in linear power
    "NDCV",  # (VV_CV - VH_CV) / (VV_CV + VH_CV)
]

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

#################### GOOGLE SATELLITE EMBEDDING V1 (AlphaEarth Foundations) ##########################
# Annual, analysis-ready 10 m embeddings: each pixel is a 64-D unit-length vector
# (bands A00-A63) summarizing one calendar year of multi-sensor (optical+radar+LiDAR)
# surface trajectories, from the AlphaEarth Foundations model (Brown et al. 2025). A
# foundation-model OUTPUT, not spectral data: NO scale factor, cloud masking, or
# spectral indexing. The 64 bands are ONE coordinate — use them together; never compute
# per-band indices or select a subset for analysis (only for RGB visualization). Vectors
# are unit-length (no normalization); the space is consistent across years, so the dot
# product of two years' vectors is their cosine similarity (change detection). Each image
# is in its LOCAL UTM zone (UTM_ZONE property) — handle CRS for multi-zone AOIs; annual
# cadence (system:time_start = Jan 1). CC-BY-4.0 with REQUIRED attribution to Google +
# Google DeepMind. Pin DATASET_VERSION + MODEL_VERSION for reproducible deliverables.
SATELLITE_EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
# The 64 embedding axes, named A00..A63 (zero-padded two digits).
SATELLITE_EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]
SATELLITE_EMBEDDING_SCALE = 10

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
ROY_OLI_TO_ETM_INTERCEPTS = [
    -0.0095,
    -0.0016,
    -0.0022,
    -0.0021,
    -0.0030,
    0.0029,
]

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

# Normalized greenness/moisture indices historically offered as the segmentation index.
# Loss = NEGATIVE delta for these, so they are multiplied by -1 to make loss-positive for
# segmentation (dist_dir = -1). The segmentation/FTV builders are no longer limited to this
# tuple: any INDEX_REGISTRY index computable from LANDTRENDR_BAND_MAP is accepted, with its
# loss orientation taken from LANDTRENDR_DIST_DIR (default LANDTRENDR_DEFAULT_DIST_DIR).
LANDTRENDR_NORMALIZED_INDICES = ("NBR", "NDVI", "NDMI")

# Logical band-map keys -> LandTrendr common band names, so ANY INDEX_REGISTRY index can be
# computed on a common-band medoid composite via the generic index registry. There are no
# red-edge keys (Landsat has no red edge), so red-edge indices (NDRE, CIred_edge, MTCI, ...)
# are correctly reported as non-computable for LandTrendr rather than silently mis-banded.
LANDTRENDR_BAND_MAP = {
    "blue": "BLUE",
    "green": "GREEN",
    "red": "RED",
    "nir": "NIR",
    "swir1": "SWIR1",
    "swir2": "SWIR2",
}

# Disturbance orientation per segmentation index. LandTrendr requires the segmentation band
# to be LOSS-POSITIVE (disturbance = a rising edge). Greenness/moisture indices fall with
# degradation, so they are multiplied by -1 (the default); bare-soil / burned-area indices
# rise with degradation, so they are used as-is (+1). Indices not listed here fall back to
# LANDTRENDR_DEFAULT_DIST_DIR.
LANDTRENDR_DEFAULT_DIST_DIR = -1
LANDTRENDR_DIST_DIR = {
    "BSI": 1,
    "MBI": 1,
    "EMBI": 1,
    "DBSI": 1,
    "NDBI": 1,
    "UI": 1,
    "BAI": 1,
    "BAIS2": 1,
}

# Internal FTV band name build_landtrendr_collection always adds (a duplicate of the
# segmentation index, natural-signed) so get_fitted_stack(index=None) can recover the
# segmentation index's own DENSE annual fitted trajectory via the FTV mechanism, instead of
# the raw "LandTrendr" band's fitted-value row -- which is only populated for years a given
# pixel had a valid (unmasked) input observation, and is therefore genuinely ragged
# (per-pixel-variable column count) rather than a fixed-length array matching the requested
# year range.
LANDTRENDR_SEGMENTATION_SELF_BAND = "segmentation_self"

#################### Baseline Assessment (workflows.run_baseline_assessment) ####
# Layer keys used by build_baseline_layers / export_baseline_layers, in canonical order.
BASELINE_LAYER_NAMES = [
    "dem",
    "slope",
    "hillshade",
    "canopy_height",
    "land_cover",
    "soil_carbon",
    "bii_all",
    "forest_2000",
    "forest_loss",
]
# Continuous (non-categorical) layers combined into one stack for per-site
# mean/median/min/max/stdDev summaries. land_cover is summarized separately by
# class area (see ESA_CLASS_MAP); forest_2000/forest_loss are left out of the
# continuous summary, matching the source TGBS_Kwale baseline notebook.
BASELINE_CONTINUOUS_LAYERS = [
    "dem",
    "slope",
    "hillshade",
    "canopy_height",
    "soil_carbon",
    "bii_all",
]
# Default per-layer raster export scale (m) -- uniform 30m, including canopy_height
# downsampled from its native 1m, so every baseline layer shares one output grid.
DEFAULT_BASELINE_EXPORT_SCALE = {name: 30 for name in BASELINE_LAYER_NAMES}

#################### IMAGE TEXTURE (sensors/texture.py) ##########################
# Metric suffixes ee.Image.glcmTexture emits (output band = <band>_<metric>).
GLCM_METRICS = (
    "asm",
    "contrast",
    "corr",
    "var",
    "idm",
    "savg",
    "svar",
    "sent",
    "ent",
    "dvar",
    "dent",
    "imcorr1",
    "imcorr2",
    "diss",
    "inertia",
    "shade",
    "prom",
)
# Emitted by glcmTexture but always 0 ("not computed" per the API doc; confirmed 2026-10-14).
GLCM_UNCOMPUTED_METRICS = ("maxcorr",)
# Same quantity (bit-identical in GEE); never request both.
GLCM_EQUIVALENT_METRICS = (("contrast", "inertia"),)
DEFAULT_GLCM_METRICS = ("idm", "ent")  # homogeneity + entropy (Silveira et al. 2026)
# ent1 = first-order (histogram) entropy, distinct from the GLCM "ent".
FIRST_ORDER_STATS = ("mean", "sd", "ent1")

# Hall-Beyer 2017 / Silveira et al. 2026: bit depth barely matters (r 0.90-1.00); keep >= ~24
# levels; beyond 128 only compute cost rises. 256 = 8-bit -> toUint8.
TEXTURE_QUANT_LEVELS = 256
TEXTURE_MIN_QUANT_LEVELS = 24
TEXTURE_MAX_QUANT_LEVELS = 256

# FIXED stretch ranges shared by every site/date (quantization must not be per-image).
# Callers may override per band via value_ranges.
TEXTURE_VALUE_RANGES: dict[str, tuple[float, float]] = {
    "NDVI": (0.0, 1.0),
    "EVI": (0.0, 1.0),
    "SAVI": (0.0, 1.0),
    "MSAVI2": (0.0, 1.0),
    "kNDVI": (0.0, 1.0),
    "VV": (
        -25.0,
        0.0,
    ),  # dB; provisional -- not yet checked against East African histograms
    "VH": (-30.0, -5.0),  # dB; provisional
}

TEXTURE_MAX_RADIUS = 50  # 101x101 px, the largest window Silveira et al. tested

#################### OTSU THRESHOLDING ##########################
# Adaptive (per-scene) thresholds for bimodal bands such as NDWI/MNDWI or SAR dB
# (see thresholding.py). Edge-based Otsu samples only pixels near Canny edges so
# both classes are roughly balanced in the histogram (Donchyts et al. 2016).
OTSU_HISTOGRAM_BUCKETS = 100
CANNY_THRESHOLD = 0.9  # gradient magnitude threshold for CannyEdgeDetector
CANNY_SIGMA = 1.0  # Gaussian smoothing sigma (pixels) before edge detection

#################### ZONAL STATISTICS ##########################
# Site-level zonal functions take `scale` with no default: the caller chooses the
# accuracy/EECU trade-off. A region covering fewer than this many pixels at the
# chosen scale triggers a warning, because its statistics then come mostly from
# outside the region (e.g. a 6 ha site inside one 5566 m CHIRPS pixel).
DEFAULT_MIN_PIXELS_PER_REGION = 4
REGION_AREA_MAX_ERROR_M = 1  # maxError (m) for the geometry.area() size check

#################### GLOBAL DEFAULTS ##########################
SIGMA = 0.15  # default kNDVI sigma

DEFAULT_MAX_PIXELS = 1e13
DEFAULT_FILE_FORMAT = "GeoTIFF"
DEFAULT_CRS = "EPSG:4326"

# Representative wet/dry season month ranges for tropical East-African sites.
# Format: (start_month, end_month) inclusive; pass directly to build_seasonal_composites.
# Override per AOI and region — these are illustrative defaults, not universal.
SEASON_WET: tuple[int, int] = (3, 5)  # Mar–May  (long rains)
SEASON_DRY: tuple[int, int] = (7, 10)  # Jul–Oct  (boreal dry)

#################### CHIRPS v3 PENTAD (site selection, P1) ##########################
# Same band name as the daily product, but units are mm per pentad (72 pentads/year).
# Sums over a year or more equal the daily product to < 0.0001 mm (checked 2026-11-05),
# so annual / multi-year totals request the pentad cadence at ~5x fewer image reads.
# There is NO CHIRPS V3 monthly collection.
CHIRPS_PENTAD_COLLECTION = "UCSB-CHC/CHIRPS/V3/PENTAD"
CHIRPS_CADENCE_COLLECTIONS = {
    "daily": CHIRPS_COLLECTION,
    "pentad": CHIRPS_PENTAD_COLLECTION,
}
CHIRPS_NATIVE_SCALE_M = 5566
# Fixed normal window for rainfall climatologies (inclusive years).
CLIMATOLOGY_START_YEAR = 1991
CLIMATOLOGY_END_YEAR = 2020

#################### TerraClimate (IDAHO_EPSCOR/TERRACLIMATE) ##########################
# Monthly ~4.6 km climate and water balance. Scales differ PER BAND: def/pet/tmmn/tmmx
# are stored x10, but `pr` is NOT scaled (checked 2026-11-05: Nairobi monthly mean pr
# ~90 raw = mm). A blanket x0.1 makes aridity 10x wrong.
TERRACLIMATE_COLLECTION = "IDAHO_EPSCOR/TERRACLIMATE"
TERRACLIMATE_SCALES = {
    "pr": 1.0,
    "pet": 0.1,
    "def": 0.1,
    "tmmn": 0.1,
    "tmmx": 0.1,
}
TERRACLIMATE_NATIVE_SCALE_M = 4638
# Output band names of get_terraclimate_climatology().
TERRACLIMATE_CLIMATOLOGY_BANDS = ["cwd", "aridity", "tmin", "tmax"]

#################### WorldClim V1 BIO (static ~1 km bioclim) ##########################
WORLDCLIM_BIO_IMAGE = "WORLDCLIM/V1/BIO"
# Catalog scales: temperatures are stored x10 (bio04 x100); precipitation bands and
# bio15 (coefficient of variation, %) are unscaled.
WORLDCLIM_SCALES = {"bio01": 0.1, "bio04": 0.01, "bio12": 1.0, "bio15": 1.0}
WORLDCLIM_DEFAULT_BANDS = ["bio01", "bio04", "bio12", "bio15"]
WORLDCLIM_NATIVE_SCALE_M = 928

#################### RESOLVE Ecoregions 2017 ##########################
# Polygons are split, so always match on ECO_ID (never on the polygon).
RESOLVE_ECOREGIONS_COLLECTION = "RESOLVE/ECOREGIONS/2017"
RESOLVE_ECO_ID_FIELD = "ECO_ID"
RESOLVE_BIOME_FIELD = "BIOME_NUM"

#################### TNC Global Human Modification v3 ##########################
# ImageCollection holding one 2022 image; take .first() (keeps native 90 m projection).
HUMAN_MODIFICATION_COLLECTION = "TNC/HM/v3/90m_s"
HUMAN_MODIFICATION_BAND = "All_threats_combined"
HUMAN_MODIFICATION_YEAR = 2022

#################### OpenLandMap PNV biome (fallback national class) ##########################
PNV_BIOME_IMAGE = "OpenLandMap/PNV/PNV_BIOME-TYPE_BIOME00K_C/v01"
PNV_BIOME_BAND = "biome_type"
PNV_NATIVE_SCALE_M = 1000
# Biome codes (image property biome_type_class_values/_names, checked 2026-11-08) that
# count as the forest integrity stratum (canopy-height test, forest-only scoring
# extras) when the class gate falls back to PNV. Open woodland, shrubland and savanna
# (16 temperate sclerophyll woodland and shrubland, 17 temperate evergreen needleleaf
# open woodland, 18 tropical savanna, 20 xerophytic woods/scrub) use the non-forest
# path: "intactness must not mean more trees" (spec §1.2; Ecosia first-run review).
PNV_FOREST_STRATUM_BIOMES = frozenset({1, 2, 3, 4, 7, 8, 9, 13, 14, 15})

#################### MERIT Hydro (HAND) ##########################
MERIT_HYDRO_IMAGE = "MERIT/Hydro/v1_0_1"
MERIT_HAND_BAND = "hnd"
MERIT_NATIVE_SCALE_M = 90
# Height above nearest drainage separating riparian/wetland (< 5 m) from upland.
HAND_RIPARIAN_MAX_M = 5.0

#################### Copernicus DEM water-body mask ##########################
# GLO-30 WBM band: 0 no water, 1 ocean, 2 lake, 3 river (checked 2026-11-05). The DEM
# itself is NOT masked over the ocean, so coastal distance uses WBM == 1.
COPERNICUS_WBM_BAND = "WBM"
COPERNICUS_WBM_OCEAN = 1

#################### iSDAsoil texture / sand / clay ##########################
# Sand and clay content are integer % with no back-transform (checked 2026-11-05);
# texture_class is the USDA texture class code.
ISDA_TEXTURE_IMAGE = "ISDASOIL/Africa/v1/texture_class"
ISDA_SAND_IMAGE = "ISDASOIL/Africa/v1/sand_content"
ISDA_CLAY_IMAGE = "ISDASOIL/Africa/v1/clay_content"
ISDA_NATIVE_SCALE_M = 30

#################### Global Mangrove Watch 2020 (coastal stratum) ##########################
GMW_2020_COLLECTION = "projects/sat-io/open-datasets/GMW/annual-extent/GMW_MNG_2020"
GMW_BAND = "b1"

#################### MODIS MCD43A4 NBAR (NDMI / NBR) ##########################
# Daily 500 m nadir BRDF-adjusted reflectance; each value comes from a 16-day window,
# so consecutive days are near-duplicates and every 8th day is kept. system:index is
# "YYYY_MM_DD".
MODIS_NBAR_COLLECTION = "MODIS/061/MCD43A4"
MODIS_NBAR_BANDS = {
    "nir": "Nadir_Reflectance_Band2",  # 841-876 nm
    "swir1": "Nadir_Reflectance_Band6",  # 1628-1652 nm
    "swir2": "Nadir_Reflectance_Band7",  # 2105-2155 nm
}
MODIS_NBAR_QA_BANDS = {
    "nir": "BRDF_Albedo_Band_Mandatory_Quality_Band2",
    "swir1": "BRDF_Albedo_Band_Mandatory_Quality_Band6",
    "swir2": "BRDF_Albedo_Band_Mandatory_Quality_Band7",
}
MODIS_NBAR_MAX_QA = 1  # 0 full inversion, 1 magnitude inversion
MODIS_NBAR_SCALE_FACTOR = 0.0001
MODIS_NBAR_SUBSAMPLE_DAYS = 8
MODIS_NBAR_SCALE_M = 500
MODIS_VI_SCALE_M = 250

#################### Uploaded assets (VECEA, SoilGrids WRB) ##########################
# Loaders resolve the asset ID as: explicit asset_id= -> environment variable -> constant.
#
# VECEA v2.0 (van Breugel et al. 2015; Zenodo 10.5281/zenodo.11125645). LICENCE: the
# Zenodo deposit says CC BY 4.0 but the bundled 2015 terms forbid redistribution and
# commercial use without permission, so the asset stays PRIVATE until the VECEA team
# confirms. To reproduce it under another project, follow VECEA_REPRODUCE_STEPS.
VECEA_ASSET_ID = "projects/charrell-personal/assets/pnv_vecea_v2_0_Kenya"
VECEA_ASSET_ENV = "EETOOLS_VECEA_ASSET"
VECEA_BAND = "b1"
VECEA_NATIVE_SCALE_M = 92
VECEA_VERSION = "v2.0 (2024.05)"
VECEA_DOI = "10.5281/zenodo.11125645"
VECEA_REPRODUCE_STEPS = (
    "1) download pnv_vecea_v2_0_<Country>_raster.zip from Zenodo "
    "10.5281/zenodo.11125645; 2) upload pnv_vecea_v2_0_<Country>.tif as an image "
    "asset with pyramiding policy MODE; 3) set EETOOLS_VECEA_ASSET or "
    "asset_overrides={'vecea': '<asset id>'}."
)
# SoilGrids 2.0 WRB most-probable reference soil group (ISRIC, CC BY 4.0), Kenya clip.
SOILGRIDS_WRB_ASSET_ID = (
    "projects/charrell-personal/assets/soilgrids_wrb_mostprobable_kenya"
)
SOILGRIDS_WRB_ASSET_ENV = "EETOOLS_SOILGRIDS_WRB_ASSET"
SOILGRIDS_WRB_BAND = "b1"
SOILGRIDS_WRB_NATIVE_SCALE_M = 230
# Asset key -> (default ID, environment variable).
UPLOADED_ASSETS = {
    "vecea": (VECEA_ASSET_ID, VECEA_ASSET_ENV),
    "soilgrids_wrb": (SOILGRIDS_WRB_ASSET_ID, SOILGRIDS_WRB_ASSET_ENV),
}

# VECEA Kenya class table (MAP_ID -> (code, name)), from pnv_vecea_v2_0_Kenya.csv.
# All 8 VECEA countries share one MAP_ID space (83 codes, no conflicts), so other
# countries extend this dict.
VECEA_CLASS_TABLE: dict[int, tuple[str, str]] = {
    1: ("A", "Afroalpine vegetation"),
    2: ("Ad", "Afromontane desert"),
    3: ("B", "Afromontane bamboo"),
    4: ("Bd", "Somalia-Masai Acacia-Commiphora deciduous bushland and thicket"),
    5: (
        "Bd/Wcd",
        "Transitional zone Somalia-Masai Acacia-Commiphora deciduous bushland and "
        "thicket and Dry Combretum wooded grassland",
    ),
    8: ("Bds", "Acacia-Commiphora stunted bushland"),
    9: (
        "Bds/S",
        "Somalia-Masai Acacia-Commiphora shrubland + Somalia-Masai semi-desert "
        "grassland and shrubland",
    ),
    10: ("Be", "Evergreen and semi-evergreen bushland and thicket"),
    15: ("CM", "Coastal mosaic"),
    16: ("D", "Desert"),
    17: ("E", "Montane Ericaceous belt"),
    19: ("Fa", "Afromontane rain forest"),
    21: ("Fb", "Afromontane undifferentiated forest"),
    22: (
        "Fb/Be/wd",
        "Complex of Afromontane undifferentiated forest with wooded grasslands and "
        "evergreen or semi-evergreen bushland and thicket at lower margins",
    ),
    24: ("Fd", "Single-dominant Hagenia abyssinica forest"),
    26: ("Fe", "Afromontane moist transitional forest"),
    27: ("Ff", "Lake Victoria transitional rain forest"),
    29: ("Fh", "Afromontane dry transitional forest"),
    30: (
        "Fi",
        "Lake Victoria drier peripheral semi-evergreen Guineo-Congolian rain forest",
    ),
    33: ("Fo", "Zanzibar-Inhambane lowland rain forest"),
    35: ("G", "Climatic grasslands"),
    39: ("M", "Mangrove"),
    40: ("P", "Palm wooded grassland"),
    41: ("R", "Riverine wooded vegetation"),
    42: ("S", "Somalia-Masai semi-desert grassland and shrubland"),
    46: ("Wcd", "Dry combretum wooded grassland"),
    49: ("Wcm", "Moist Combretum wooded grassland"),
    51: (
        "Wd/Wc",
        "Acacia-Commiphora deciduous wooded grassland + Combretum wooded grassland",
    ),
    52: ("WdK", "Acacia tortilis wooded grassland and woodland"),
    53: ("We", "Upland Acacia wooded grassland"),
    67: (
        "Wn/g",
        "Catena of North Zambezian Undifferentiated woodland + edaphic grassland on "
        "drainage-impeded or seasonally flooded soils",
    ),
    73: ("X", "Freshwater swamp"),
    81: ("g", "Edaphic grassland on drainage-impeded or seasonally flooded soils"),
    85: (
        "g/X",
        "Edaphic grassland on drainage-impeded or seasonally flooded soils + "
        "freshwater swamp",
    ),
    86: ("gv", "Edaphic grassland on volcanic soils"),
    87: ("s", "Sand"),
    88: ("w", "Water bodies"),
    89: (
        "wd",
        "Edaphic wooded grassland on drainage-impeded or seasonally flooded soils",
    ),
    92: ("Z", "Halophytic vegetation"),
}
# MAP_ID -> ecological group (spec 2026-11-05 §4.4). Basis: VECEA main groups, with
# "Other types" split by ecology and three documented deviations: B (bamboo) and E
# (Ericaceous belt) are Afromontane belts, R (riverine) is azonal. Composite types take
# the group of their first component. 87 (sand) and 88 (water) are mapping classes, not
# PNV types, and are deliberately absent (never matched by a group).
VECEA_CLASS_GROUPS: dict[int, str] = {
    **dict.fromkeys([19, 21, 22, 24, 26, 27, 29, 30, 33], "forest"),
    **dict.fromkeys([40, 46, 49, 51, 52, 53, 67, 89], "woodland"),
    41: "riverine",
    **dict.fromkeys([4, 5, 8, 9, 10], "bushland"),
    3: "bamboo",
    **dict.fromkeys([1, 2, 17], "afroalpine_ericaceous"),
    35: "climatic_grassland",
    **dict.fromkeys([81, 85, 86], "edaphic_grassland"),
    73: "swamp",
    **dict.fromkeys([16, 42], "semi_desert"),
    39: "mangrove",
    92: "halophytic",
    15: "coastal_mosaic",
}
VECEA_EXCLUDED_CLASSES = frozenset({87, 88})  # sand, water bodies
VECEA_MOSAIC_CLASSES = frozenset({15})  # CM: too broad to discriminate coastal strata
# Groups that count as forest/woodland strata for the forest-only scoring extras.
# VECEA_CLASS_GROUPS groups scored with the forest integrity stratum (median canopy
# height >= FOREST_MIN_CANOPY_M, forest-only intactness extras). "woodland" is not one:
# in Kenya it is mostly wooded grassland (e.g. Wcd, Dry combretum wooded grassland),
# whose cells have 0-1 m median canopy, so it uses the non-forest path (1-gHM and
# natural cover). Changed 2026-11-08 after the TGBS_Ecosia first run.
VECEA_FOREST_STRATUM_GROUPS = frozenset({"forest", "riverine", "bamboo"})

# SoilGrids WRB 2006 reference soil group codes (MostProbable.rat.json).
WRB_CLASS_TABLE: dict[int, str] = {
    0: "Acrisols",
    1: "Albeluvisols",
    2: "Alisols",
    3: "Andosols",
    4: "Arenosols",
    5: "Calcisols",
    6: "Cambisols",
    7: "Chernozems",
    8: "Cryosols",
    9: "Durisols",
    10: "Ferralsols",
    11: "Fluvisols",
    12: "Gleysols",
    13: "Gypsisols",
    14: "Histosols",
    15: "Kastanozems",
    16: "Leptosols",
    17: "Lixisols",
    18: "Luvisols",
    19: "Nitisols",
    20: "Phaeozems",
    21: "Planosols",
    22: "Plinthosols",
    23: "Podzols",
    24: "Regosols",
    25: "Solonchaks",
    26: "Solonetz",
    27: "Stagnosols",
    28: "Umbrisols",
    29: "Vertisols",
}
# "Related" soil groups (matrix score 1): the FAO Major Soils of the World sets
# (Driessen et al. 2001) with three stated adjustments: Set #3 split into singletons
# (Andosols, Arenosols, Vertisols); Set #4 split into lowland (Fluvisols, Gleysols) and
# upland/erosional (Leptosols, Regosols); Stagnosols join Planosols in Set #9.
WRB_RELATED_GROUPS: dict[str, frozenset[int]] = {
    "1_organic": frozenset({14}),
    "3a_volcanic": frozenset({3}),
    "3b_sandy": frozenset({4}),
    "3c_shrink_swell_clay": frozenset({29}),
    "4a_lowland": frozenset({11, 12}),
    "4b_upland_erosional": frozenset({16, 24}),
    "5_limited_age": frozenset({6}),
    "6_wet_tropical": frozenset({22, 10, 2, 19, 0, 17}),
    "7_arid": frozenset({25, 26, 13, 9, 5}),
    "8_steppic": frozenset({7, 15, 20}),
    "9_subhumid_temperate": frozenset({23, 21, 27, 1, 18, 28}),
    "10_permafrost": frozenset({8}),
}

#################### SITE SELECTION (eetools.site_selection) ##########################
# Coarse-to-fine grid (spec §5.0): fine cell = screen / k with k restricted so every
# cell is a whole multiple of the 250 m fraction-image grid.
ALLOWED_K = (1, 2, 4, 8)
FRACTION_SCALE_M = 250
SCREEN_CELL_STEP_M = 500  # grown screening cells snap up to multiples of this
SITE_SELECTION_INDICATOR_SCALE_M = 90  # Stage 1 gate-indicator reduction
FINE_MIN_SHARE = 0.5  # fine-cell categorical gates: in-set share >= this
COARSE_MIN_SHARE_FLOOR = 0.05  # coarse gates: max(focal/cell area, this)
MIN_LAND_SHARE = 0.5  # cells with less land than this (ocean) are removed
DEFAULT_TIER_DISTANCES_KM = (25.0, 50.0, 150.0)
DEFAULT_SCREEN_CELL_M = 2000.0
DEFAULT_EXCLUDE_BUFFER_M = 500.0
DEFAULT_REFERENCE_BUFFER_M = 1000.0
DEFAULT_FOCAL_SAMPLES = 300
REDUCE_BATCH_SIZE = 1500  # cells per reduceRegions request
# Coastal mode: active when the focal site is within this distance of the ocean;
# candidates must then lie within max(2 x focal distance, COASTAL_MIN_LIMIT_M).
COASTAL_TRIGGER_M = 10_000.0
COASTAL_MIN_LIMIT_M = 2_000.0
COAST_DISTANCE_SCALE_M = 1000
COAST_DISTANCE_MAX_M = 50_000  # coastal mode only matters within ~10-20 km

# Search ladder (spec §5.1): (max_tier, relax_substrate, relax_class) per rung.
SEARCH_LADDERS: dict[str, tuple[tuple[int, bool, bool], ...]] = {
    "before_tier3": (
        (1, False, False),
        (2, False, False),
        (2, True, False),
        (2, True, True),
        (3, True, True),
    ),
    "after_tier3": (
        (1, False, False),
        (2, False, False),
        (3, False, False),
        (3, True, False),
        (3, True, True),
    ),
    "off": (
        (1, False, False),
        (2, False, False),
        (3, False, False),
    ),
}

# SOP envelope defaults (CER-K SOP; record what was used).
SOP_ELEVATION_TOL_M = 300.0
SOP_RAINFALL_TOL_FRAC = 0.25

# MESS predictors (condition-independent only), in interpretability priority order:
# drop_collinear keeps the earlier variable of a correlated pair (Porfirio et al. 2014).
ENV_PREDICTORS = [
    "elevation",
    "bio12",
    "bio01",
    "sand",
    "clay",
    "hand",
    "slope",
    "cwd",
    "aridity",
    "bio04",
    "bio15",
    "tpi",
]
COAST_PREDICTOR = "coast_dist"
# Climate predictors are reduced once at coarse level and inherited by fine cells.
CLIMATE_PREDICTORS = ["bio01", "bio04", "bio12", "bio15", "cwd", "aridity"]
TERRAIN_SOIL_PREDICTORS = ["elevation", "slope", "tpi", "sand", "clay", "hand"]
PREDICTOR_LABELS = {
    "elevation": "elevation",
    "slope": "slope",
    "tpi": "topographic position",
    "bio01": "mean temperature",
    "bio04": "temperature seasonality",
    "bio12": "annual rainfall",
    "bio15": "rainfall seasonality",
    "cwd": "climatic water deficit",
    "aridity": "aridity index",
    "sand": "sand content",
    "clay": "clay content",
    "hand": "height above drainage",
    "coast_dist": "distance to coast",
}
# Native sampling scale (m) of each predictor for the focal MESS reference sample.
PREDICTOR_NATIVE_SCALE_M = {
    "elevation": 30,
    "slope": 30,
    "tpi": 30,
    "bio01": WORLDCLIM_NATIVE_SCALE_M,
    "bio04": WORLDCLIM_NATIVE_SCALE_M,
    "bio12": WORLDCLIM_NATIVE_SCALE_M,
    "bio15": WORLDCLIM_NATIVE_SCALE_M,
    "cwd": TERRACLIMATE_NATIVE_SCALE_M,
    "aridity": TERRACLIMATE_NATIVE_SCALE_M,
    "sand": ISDA_NATIVE_SCALE_M,
    "clay": ISDA_NATIVE_SCALE_M,
    "hand": MERIT_NATIVE_SCALE_M,
    "coast_dist": COAST_DISTANCE_SCALE_M,
}
MESS_MIN_NATIVE_PX = 9
MESS_COLLINEARITY_R = 0.8
TPI_RADIUS_M = 500
# MESS range floors (spec §5.2): ("abs", total range) or ("rel", fraction of |focal
# median|). [L] defaults, written to layer_register.
MESS_RANGE_FLOORS: dict[str, tuple[str, float]] = {
    "elevation": ("abs", 50.0),
    "slope": ("abs", 2.0),
    "tpi": ("abs", 5.0),
    "bio01": ("abs", 0.5),
    "bio12": ("rel", 0.05),
    "bio04": ("rel", 0.05),
    "bio15": ("rel", 0.05),
    "cwd": ("rel", 0.05),
    "aridity": ("abs", 0.05),
    "sand": ("abs", 5.0),
    "clay": ("abs", 5.0),
    "hand": ("abs", 2.0),
    "coast_dist": ("abs", 1000.0),
}

# WorldCover classes counted as natural cover (tree, shrub, grass, herbaceous wetland,
# mangrove, moss/lichen) and the land-cover fraction classes.
NATURAL_WORLDCOVER_CLASSES = (10, 20, 30, 90, 95, 100)
WORLDCOVER_CROPLAND = 40
WORLDCOVER_BUILT = 50
WORLDCOVER_BARE = 60
WORLDCOVER_WATER = 80
WORLDCOVER_YEAR = 2021
CORE_FRAC_RADIUS_M = 1000
REDUCE_RESOLUTION_MAX_PIXELS = 1024

# WDPA protection weights by IUCN category (spec §5.3); STATUS must be "Designated".
WDPA_IUCN_FIELD = "IUCN_CAT"
WDPA_STATUS_FIELD = "STATUS"
WDPA_DESIGNATED = "Designated"
WDPA_IUCN_PROTECTION_WEIGHTS = {
    "Ia": 1.0,
    "Ib": 1.0,
    "II": 1.0,
    "III": 0.8,
    "IV": 0.8,
    "V": 0.5,
    "VI": 0.5,
    "Not Reported": 0.6,
    "Not Assigned": 0.6,
    "Not Applicable": 0.6,
}
WDPA_STRICT_CATEGORIES = ("Ia", "Ib", "II")  # WDPA-strict auto-seeds
WDPA_I_IV_CATEGORIES = ("Ia", "Ib", "II", "III", "IV")  # matrix protection = 2
WDPA_RASTER_SCALE_M = 30
WDPA_STRICT_MIN_SHARE = 0.5
WDPA_STRICT_MIN_CELLS = 2
WDPA_STRICT_MAX_SEED_CELLS = 20

# Productivity windows (Trends.Earth-style).
STATE_WINDOW_YEARS = 3
TREND_WINDOW_YEARS = 15
RECENT_LOSS_YEARS = 5
EMBEDDING_FIRST_YEAR = 2017
EMBEDDING_LATEST_YEAR = 2025
EMBEDDING_REDUCE_SCALE_M = 100
DYNAMIC_WORLD_FIRST_YEAR = 2018  # ranked-site DW context skipped before this end_year
DYNAMIC_WORLD_CONTEXT_SCALE_M = 30
DYNAMIC_WORLD_CONTEXT_BANDS = ["bare", "crops", "built"]
CANOPY_HEIGHT_REDUCE_SCALE_M = 30
FOREST_MIN_CANOPY_M = 5.0  # [L] forest integrity: canopy consistent with forest class
VIIRS_REDUCE_SCALE_M = 500

# Degraded workflow.
DEGRADED_EXCLUDE_MAX_FRAC = (
    0.5  # drop cells with >= this water (any cell) or built-up (coarse screen) fraction
)
# Degraded built-up filters on final candidate cells (not for degradation_type
# 'settlement'). GBS focal sites hold no built structures, so a built-up cell is the
# wrong 0% anchor; cropland stays acceptable. [L] defaults, set to tolerate scattered
# smallholder homesteads and tracks (2026-11-09 review: at 0.01 the TGBS_ITF degraded
# shortlist kept 2 of 34 farm cells; at 0.05, 13). A cell up to 5% built changes
# cell-mean NDVI/NDMI little compared with the anchor gaps D1 selects on; the sprawl
# cells the filter targets were 0.18-1.0 (TGBS_Ecosia). Field-check at the next
# smallholder site.
DEGRADED_MAX_BUILT_FRAC = 0.02  # ESA WorldCover (2021) built-up fraction
DEGRADED_MAX_DW_BUILT = 0.05  # Dynamic World built-label share (dw_landuse family)
DW_BUILT_LABEL = 6  # Dynamic World 'label' code for built

# Dynamic World land-use history (dw_landuse family, 2026-11-09 robustness review).
# One 30 m reduction of yearly DW label modes from DW_LANDUSE_FIRST_YEAR to end_year on
# the final cells of each rung; both workflows read the same values. Thresholds below
# are [L] tolerances applied in pandas, never part of the provenance key.
DW_LANDUSE_FIRST_YEAR = 2019  # first year of the land-use history (as tested)
DW_LANDUSE_SCALE_M = 30
DW_NATURAL_LABELS = (1, 2, 3, 5)  # trees, grass, flooded vegetation, shrub and scrub
DW_MODIFIED_LABELS = (4, 6, 7)  # crops, built, bare
DW_CROP_LABEL = 4
LANDUSE_MIN_NATURAL_PERSIST = 0.8  # reference gate: persistent-natural cell share
LANDUSE_MIN_MODIFIED_SHARE = 0.5  # converted / persistently modified cell share
LANDUSE_MIN_VALID_SHARE = 0.5  # below this share with DW observations -> 'no_dw'
LANDUSE_CLASSES = (
    "persistent_natural",
    "converted",
    "persistent_modified",
    "built",
    "mixed",
    "no_dw",
)
# Degradation pathways that convert land cover (absolute DW pathway test: modified at or
# before the end of the baseline window). Pathways inside natural cover (overgrazing,
# rangeland) keep the D2 signature gate and are not separated from references by class.
CONVERSION_DEGRADATION_TYPES = frozenset(
    {"generic", "cropland", "settlement", "mining", "bare_soil", "deforestation"}
)
PA_EXCLUDE_MIN_SHARE = 0.5  # exclude_protected: drop cells this share inside a PA
# Degraded anchor (2026-11-07 amendment): the degraded site is the 0% end of the TGBS
# reference = 100% / degraded = 0% rescale, so it must be at least as degraded as the
# focal site in the baseline and current windows on these indices.
ANCHOR_INDICES = ("ndvi", "ndmi")
DEGRADED_BASELINE_WINDOW_YEARS = 3  # baseline = the 3 complete years before start
MODIS_FIRST_FULL_YEAR = 2001  # first complete MOD13Q1 / MCD43A4 year
PATHWAY_GATE_PCT = 0.5  # [L] min pct(signature_sim) for the pathway gate
ANCHOR_REVIEW_PCT = 0.5  # [L] focal_baseline_pct above this -> anchor_check "review"
# Rules-mode signature profiles: terms are (column, orientation), +1 = higher means
# "more like this degradation type"; layers name the extra reductions a profile needs.
DEGRADATION_TYPE_PROFILES: dict[str, dict[str, list]] = {
    "mining": {
        "terms": [("bare_frac", 1), ("natural_frac", -1), ("ndvi_state", -1)],
        "layers": [],
    },
    "bare_soil": {
        "terms": [("bare_frac", 1), ("natural_frac", -1), ("ndvi_state", -1)],
        "layers": [],
    },
    "cropland": {"terms": [("crop_frac", 1)], "layers": []},
    "deforestation": {
        "terms": [("forest_loss_frac", 1), ("canopy_height_p50", -1)],
        "layers": ["canopy"],
    },
    "overgrazing": {
        "terms": [("rue_trend", -1), ("bare_frac", 1), ("natural_frac", -1)],
        "layers": ["rue"],
    },
    "rangeland": {
        "terms": [("rue_trend", -1), ("bare_frac", 1), ("natural_frac", -1)],
        "layers": ["rue"],
    },
    "settlement": {
        "terms": [("ghm", 1), ("built_frac", 1), ("viirs", 1)],
        "layers": ["viirs"],
    },
    "generic": {
        "terms": [
            ("ndvi_state", -1),
            ("ndmi_state", -1),
            ("natural_frac", -1),
            ("ghm", 1),
        ],
        "layers": [],
    },
}

# CER-K SOP selection-matrix thresholds ([L] defaults, reported in layer_register).
SOP_MATRIX_THRESHOLDS = {
    "integrity_loss_2": 0.01,  # loss since 2000 < 1% of cell for a 2
    "integrity_loss_1": 0.05,  # loss since 2000 <= 5% for a 1
    "recent_loss_frac": 0.005,  # recent (last 5 yr) loss fraction allowed
    # Non-forest integrity 2 also needs the reference land-use gate and gHM below the
    # moderate/high break of Kennedy et al. 2019 (low < 0.1, moderate 0.1-0.4, high
    # 0.4-0.7, very high > 0.7). Absolute, so it never moves with the pool.
    "integrity_max_ghm": 0.4,
    "size_partial": 0.5,
    "protection_share": 0.5,
    "degraded_loss_2": 0.10,
    "degraded_loss_1": 0.01,
    "pathway_2": 0.75,
    "pathway_1": 0.5,
    "elevation_band_m": 150.0,
    "rainfall_band_frac": 0.125,
}
INTACTNESS_WEAK_NATURALNESS = 0.7  # 1 - ghm below this -> intactness_flag "weak"
INTACTNESS_WEAK_NATURAL_FRAC = 0.6
ZERO_STAR_HIGH_PCT = 0.9
ZERO_STAR_MODERATE_PCT = 0.75
ZERO_STAR_MAX_PERFORMANCE = 0.5
SPAN_ENV_COVERAGE_SD = 1.0  # env_coverage: focal samples within this RMS z-distance
