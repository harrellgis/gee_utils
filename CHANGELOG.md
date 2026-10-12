# Changelog

All notable changes to `eetools` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Four MODIS products in `sensors.modis`, each with a `get_*_collection(aoi, start_date, end_date, ...)`
  builder that validates the date range against the collection:
  - **MOD13Q1** (`get_modis_vi_collection`): 250 m 16-day NDVI/EVI plus red/NIR/blue/SWIR2
    reflectance, scaled by 0.0001 and masked by `SummaryQA` (default keeps good + marginal,
    tunable via `max_summary_qa`).
  - **MCD64A1** (`get_modis_burned_area_collection`): 500 m monthly burned area with a derived
    0/1 `burned` band. GEE masks `BurnDate` over unburned land, so the flag unmasks it to 0 and
    restricts to valid land observations (QA bits 0 and 1), keeping "unburned" distinct from
    "no data".
  - **MOD17A3HGF** (`get_modis_npp_collection`): 500 m annual gap-filled NPP/GPP in
    kg C/m²/yr, with fill codes (e.g. `Gpp` 65534 over water, which GEE leaves unmasked) masked.
  - **MOD44B** (`get_modis_vcf_collection`): 250 m annual percent tree / non-tree vegetation /
    non-vegetated cover, with water (200) and fill (253) codes masked.
- `sensors.smap.preprocessing` module wrapping **NASA SMAP L4** global soil moisture
  (`NASA/SMAP/SPL4SMGP/008`, ~9 km, 3-hourly). `get_smap_collection(aoi, start_date, end_date,
  bands=None, daily=True)` returns surface / rootzone / profile volumetric soil moisture
  (m³/m³) and the matching 0-1 wetness bands by default (any SPL4SMGP band can be selected),
  averaged to one image per UTC day with `year`/`month`/`day`/`image_count` properties, or
  the native 3-hourly timesteps with `daily=False`. No scale factor or masking is applied —
  the product is already in physical units, water-masked, and has no per-pixel QA band.
- `sensors.sm2rain.preprocessing` module wrapping the **OpenLandMap SM2RAIN** monthly
  precipitation climatology (`OpenLandMap/CLM/CLM_PRECIPITATION_SM2RAIN_M/v01`, 1 km).
  The source is a single 12-band image of 2007-2018 mean mm/month (an ensemble of
  SM2RAIN-ASCAT, GPM IMERG, WorldClim v2, and CHELSA), not a time series.
  `get_sm2rain_climatology(months=None)` returns one image per requested calendar month with
  a `precipitation` band (matching CHIRPS) and `month` / `month_name` properties.
- `sensors.viirs` package wrapping **NASA Black Marble VIIRS VNP46A2 v002** daily ~500 m
  nighttime lights (`NASA/VIIRS/002/VNP46A2`), which supersedes the `NOAA/VIIRS/001`
  VNP46A1/VNP46A2 copies that stopped updating in January 2025.
  `get_viirs_ntl_collection(aoi, start_date, end_date, ...)` returns the BRDF/lunar-corrected
  and gap-filled radiance bands (nW/cm²/sr, no scale factor) plus quality/snow/cloud flags;
  `masking.mask_viirs_ntl_qa` masks the corrected band to high-quality (`Mandatory_Quality_Flag`
  ≤ 1, tunable) snow-free retrievals while leaving the gap-filled band untouched.
- New `sensors.modis.masking` module holding the per-product QA/fill masks. Collection IDs,
  band lists, scale factors, and valid-range thresholds live in `constants.py`.

## [0.3.0] - 2026-10-01

### Added
- `utils.get_date_window` (a season's start/end `ee.Date` window spanning a year range) and
  `utils.get_available_window` (actual first/last acquisition dates of a collection, or of
  merged collections).
- `io.upload_geotiff_to_asset` and `io.upload_shapefile_to_asset`: stage a local `.tif`, or a
  `.shp` plus its sidecar files, to Google Cloud Storage and ingest it as an Earth Engine
  asset. Adds `google-cloud-storage` as a runtime dependency.
- Baseline-layer datasets: `sensors.canopy_height.preprocessing.get_canopy_height` (Meta
  canopy height), `sensors.isda.preprocessing.get_soil_carbon` (iSDA Africa topsoil total
  carbon), and `sensors.esa.preprocessing.get_land_cover`, with their collection IDs, bands,
  and the ESA class map in `constants.py`.
- Baseline-assessment workflow in `workflows`: `build_baseline_layers`,
  `summarize_baseline_layers`, `export_baseline_layers`, and `run_baseline_assessment`, plus
  `zonal.summarize_class_areas` for per-class area sums.
- Two spectral indices in `sensors.indices` / `INDEX_REGISTRY`, selectable by name or via
  the `vegetation` domain: **MSAVI2** (`calc_msavi2`, self-adjusting soil-robust greenness
  for sparse-vegetation / bare-soil backgrounds; needs no tunable L) and **CIred_edge**
  (`calc_ci_red_edge`, red-edge chlorophyll / canopy-vigor `(NIR / RE1) - 1`; computes only
  where the band map has `red_edge`, i.e. Sentinel-2 / HLS S30, like NDRE).
- `sensors.satellite_embedding.preprocessing` module wrapping **Google Satellite
  Embedding V1** (AlphaEarth Foundations, `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`): annual
  10 m analysis-ready 64-D embedding vectors (bands `A00`–`A63`). `get_satellite_embedding_collection(aoi, start_date, end_date)`
  returns the full 64-band stack with no scale factor, masking, or spectral indexing (a
  foundation-model output, not spectral data), and `embedding_similarity(image_a, image_b)`
  computes per-pixel cosine similarity (dot product of the unit-length vectors) for
  cross-year change detection. Collection ID, band list, and scale live in `constants.py`.

### Fixed
- `landtrendr.collection.medoid_composite` no longer crashes on an empty year
  (`Image.select: Band pattern 'BLUE' was applied to an Image with no bands`). A LandTrendr
  year with zero source scenes after date/sensor filtering now returns a fully-masked
  float32 image with the requested band names (guarded by `ee.Algorithms.If` on collection
  size, mirroring `compositing.build_period_composites`), so that year contributes only
  masked pixels to the time series — the behavior LandTrendr's `minObservationsNeeded`
  already expects — instead of producing a bandless image.
- `landtrendr.collection.medoid_composite` no longer raises `Expected a homogeneous image
  collection … Mismatched type for band 'BLUE'` on multi-sensor years. Roy-harmonized OLI
  (L8/L9) scenes carry a different bounded float type than the TM/ETM+ (L5/L7) baseline for
  the same band, which the `median()` / `qualityMosaic` reductions rejected; every scene is
  now cast to a uniform float32 before reducing.
- `io.export_image_to_drive` now unmasks with `sameFootprint=False`, so fringe pixels outside
  the AOI are written as the -9999 nodata value instead of Earth Engine's fallback of 0.

### Changed
- Sentinel-2 preprocessing now keeps band `B1` (coastal aerosol); `constants.S2_BANDS`
  includes it, so S2 outputs carry one more band than in 0.2.0.
- `sensors.dem` now sources the Copernicus DEM GLO-30 from `COPERNICUS/DEM/GLO30_2024_1`
  instead of the deprecated `COPERNICUS/DEM/GLO30`. Same schema (tiled `ee.ImageCollection`,
  `DEM`/`EDM`/`FLM`/`HEM`/`WBM` bands, EGM2008 vertical datum), so `get_copernicus_dem` /
  `get_terrain` are unaffected beyond the new `constants.COPERNICUS_DEM_COLLECTION` value.
- LandTrendr's segmentation and fit-to-vertices (FTV) bands now accept **any
  `INDEX_REGISTRY` index computable from the Landsat common bands**, not just the former
  NBR/NDVI/NDMI allowlist. `landtrendr.collection._segmentation_band` / `_natural_index`
  drive the generic index registry through the new `constants.LANDTRENDR_BAND_MAP`, and the
  loss-positive orientation is taken per index from `constants.LANDTRENDR_DIST_DIR`
  (default −1 for greenness/moisture indices; +1 for bare-soil / burned-area indices such as
  BSI, so degradation stays a rising edge). Unknown or non-computable indices (e.g. red-edge
  indices, which Landsat lacks) raise `ValueError`.

## [0.2.0] - 2026-07-03

### Added
- 15 additional spectral indices, selectable by name or application domain: **GNDVI**,
  **EVI2**, **OSAVI** (vegetation), **GVMI** (moisture), **NDBI**, **UI** (urban), **MBI**,
  **EMBI**, **DBSI** (soil), **BAI**, **NBR2** (burn), and the Sentinel-2 red-edge indices
  **MTCI**, **IRECI**, **S2REP**, **BAIS2**. Formulas follow the Awesome Spectral Indices
  catalogue; each carries its ASI `short_name` and reference DOI in the docstring.
- Declarative index registry (`sensors.indices.INDEX_REGISTRY` / `IndexSpec`) driving
  `calc_indices(image, band_map, indices=None, domains=None)`: select indices by name or by
  domain (`vegetation`, `water`, `moisture`, `soil`, `burn`, `urban`), with automatic
  band-availability filtering (red-edge indices compute on Sentinel-2 and are skipped on
  sensors without a red edge). Adding an index is now a single registry entry.
- `indices` / `domains` selection arguments on every collection builder
  (`get_s2_sr_collection`, `get_l5/l7/l8/l9_sr_collection`, `get_hls_*_collection`) and the
  per-image `process_*` functions, so callers choose which indices a collection carries.
- Sentinel-2 band map extended with `red_edge2` (B6), `red_edge3` (B7), and `nir2` (B8A) to
  support the red-edge indices; `constants.ASI_BAND_LETTERS` documents the logical-key →
  ASI band-letter mapping.
- `sensors.bii.preprocessing` module for the **Biodiversity Intactness Index** (sat-io
  sub-Saharan Africa, 1 km & 8 km): `get_bii_image(resolution)` reproduces the published
  processing (self-masked per-taxon BII bands, land-use-class-masked Land Use Intensity,
  global validity mask), and `get_bii(aoi, bands, resolution="1km")` returns the selected
  band(s) clipped to the AOI (single-band for one band, multiband for several).
- `sensors.dem.preprocessing` module establishing **Copernicus DEM GLO-30**
  (`COPERNICUS/DEM/GLO30`) as the canonical elevation source: `get_copernicus_dem(aoi=None)`
  returns the mosaicked DEM in its native projection as a single `elevation` band, and
  `get_terrain(aoi=None, elevation=None, add_elevation=True)` derives slope/aspect/hillshade
  via `ee.Terrain.products`.
- `utils.vector_files_to_feature_collection(sites, layer=None)` combining multiple
  vector files into one `ee.FeatureCollection` with one dissolved feature per file.
  Identity is caller-supplied via `(path, site_id, site_name)` tuples (never inferred
  from filenames or file attributes); `source_file` is recorded automatically as
  provenance.
- `utils.vector_file_to_features(path, layer=None, keep_properties=True)` reading a
  single vector file into an `ee.FeatureCollection` with one feature per record and
  its attribute columns carried through as properties — the feature-preserving
  counterpart to `vector_file_to_ee_geometry` (which dissolves to a single geometry).
- `utils.clip_collection_to_geometry(collection, geometry, mask_outside=True)` clipping
  every image in a collection to a geometry, optionally hard-masking pixels outside an
  irregular boundary while preserving per-image properties.
- `utils.load_site_feature` gained a `layer` argument and now reads any
  geopandas-supported vector format (not only GeoPackage).
- Optional `notebook` extra (`pip install eetools[notebook]`) pulling in `geemap`
  for interactive Earth Engine map visualization in notebooks.
- `visualization.vis_params` module of ready-made geemap visualization dictionaries —
  single-band index palettes (NDVI, FPAR, EVI, NDWI, MNDWI, SAVI, NDMI, NBR, NIRv,
  NDRE), terrain palettes (elevation, slope, aspect, hillshade), RGB true-colour
  composites (Sentinel-2, Landsat 8, HLS), site/boundary vector styling, and
  Biodiversity Intactness Index palettes.
- `utils.generate_diff_image(image_initial, image_final, bands=None, output_suffix=None)`
  computing the per-band change (final - initial) between two images — e.g. the
  difference in median NDVI between two years.
- Test suite (`tests/`) covering every source module, split into pure/mock tests
  and Earth-Engine tests (`@pytest.mark.ee` / `@pytest.mark.slow`); EE tests skip
  cleanly when no session is available.
- GitHub Actions CI running ruff, `black --check`, mypy, and the non-EE tests on
  Python 3.11 and 3.12.
- `mypy` type checking with configuration in `pyproject.toml`.
- `py.typed` marker so downstream consumers receive the package's type information.
- Shared `eetools.sensors.masking` module (`build_non_water_mask`,
  `apply_water_mask`, `apply_cloud_mask`) backing the per-sensor wrappers.
- `README.md` with install, authentication, quickstart, and testing docs.

### Changed
- **Breaking:** Vector utilities (`vector_file_to_ee_geometry`, `vector_file_to_features`,
  `vector_files_to_feature_collection`, `merge_vector_files`, `build_site_feature`,
  `load_site_feature`, `get_sites_geometry`, `fc_select_properties`,
  `buffer_feature_collection`) have moved from `eetools.utils` to the new
  `eetools.vectors` module. Update imports to `from eetools.vectors import ...`.
- **Breaking:** Compositing functions (`build_composite`, `build_period_composites`,
  `build_seasonal_composites`) and zonal/sampling functions (`reduce_image_over_region`,
  `collection_to_region_timeseries`, `image_collection_to_region_stats_fc`,
  `image_collection_to_sample_fc`, `summarize_collection_histograms`) have moved from
  `eetools.visualization.summaries` (now deleted) to `eetools.compositing` and
  `eetools.zonal` respectively. Update imports accordingly.
- **Breaking:** `calc_indices` no longer accepts `include_ndre`. NDRE is now emitted whenever
  the band map exposes a `red_edge` key (i.e. Sentinel-2), via the registry's
  band-availability filtering. Remove `include_ndre=...` from call sites.
- Spectral water masking validates that a custom index selection still includes NDVI and
  MNDWI when `apply_water_masking=True`, raising `ValueError` otherwise.
- **Breaking:** `utils.gpkg_to_ee_geometry` is renamed to `utils.vector_file_to_ee_geometry`
  and now reads any geopandas-supported format (.gpkg, .geojson, .shp, ...), not only
  GeoPackage. Update call sites to the new name.
- `get_collection_min_max` and `summarize_collection_histograms` now compute
  entirely server-side (one `getInfo()` round-trip instead of one per image).
- Per-sensor water/cloud masking now delegates to the shared masking module.
- `L8_ALL_BANDS` is derived from `L8_BANDS + L8_INDEX_BANDS` (was an empty stub).
- Export status reporting in `io` uses `logging` instead of `print`.
- Package version is single-sourced from `eetools.__version__` via Hatchling.

### Fixed
- `summaries._time_windows` monthly window count: switched from
  `ee.Date.difference(unit="month")` (average-length, truncating) to a calendar
  month count, so spans containing 31-day months no longer drop the final window.
- `utils.temporal_reducer` no longer uses a mutable default argument.

## [0.1.0]

### Added
- Initial package structure: configuration, constants, IO/export helpers,
  geometry/collection utilities, per-sensor preprocessing and masking
  (Sentinel-2, Landsat 8, HLS L30/S30, MODIS LAI/FPAR, CHIRPS, ESA WorldCover),
  spectral indices, and visualization (plots, summaries, tables).
