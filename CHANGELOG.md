# Changelog

All notable changes to `eetools` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
