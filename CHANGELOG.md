# Changelog

All notable changes to `eetools` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
  NDRE), RGB true-colour composites (Sentinel-2, Landsat 8, HLS), site/boundary vector
  styling, and Biodiversity Intactness Index palettes.
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
