# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Session Start — Load This Context

At the start of every session, load context in this order:

1. **Vault overview** — `C:\Users\harre\Obsidian_Vaults\01_Work_Projects_Vault\context\agent-onboarding.md`
2. **Coding conventions** — `C:\Users\harre\Obsidian_Vaults\01_Work_Projects_Vault\context\conventions.md`
3. **Tool references** — read the wiki pages for the sensors this repo wraps:
   - `wiki/tools/google-earth-engine.md` (core API patterns and gotchas)
   - `wiki/tools/sentinel-2-sr.md`, `wiki/tools/sentinel-2-cloud-probability.md`
   - `wiki/tools/landsat-c2-sr.md`
   - `wiki/tools/hls-hlsl30.md`, `wiki/tools/hls-hlss30.md`
   - `wiki/tools/chirps.md`, `wiki/tools/dynamic-world.md`
   - (all under `C:\Users\harre\Obsidian_Vaults\01_Work_Projects_Vault\`)
4. **Decisions registry** — `C:\Users\harre\Obsidian_Vaults\01_Work_Projects_Vault\context\decisions-registry.md`
5. **Tag cluster** — `C:\Users\harre\Obsidian_Vaults\01_Work_Projects_Vault\context\tags.md` (read the **Optical EO & Cloud Masking** section only)

---

## What This Repo Does

`eetools` is a reusable Python library of Google Earth Engine utilities for environmental remote sensing. It wraps multi-sensor imagery (Sentinel-2 SR, Landsat 8 C2 L2 SR, NASA HLS L30/S30, MODIS LAI/Fpar, CHIRPS precipitation, ESA WorldCover) behind a consistent API: build cloud- and water-masked collections, append a harmonized set of spectral indices (NDVI, kNDVI, EVI, SAVI, NDWI, MNDWI, NDMI, NBR, NIRv, NDRE, Fpar), reduce over time, export to Drive or EE assets, and visualize results. It is installed as a dependency by downstream analysis projects/notebooks rather than run standalone.

---

## Setup

`uv`-managed package (see `uv.lock`). Requires Python ≥ 3.11.

```bash
# Create the environment and install with dev dependencies
uv sync

# Or install editable into an existing environment
uv pip install -e .
```

Earth Engine must be authenticated once on the machine (`earthengine authenticate`). In code, call `eetools.initialize(project="my-gee-project")` before any EE operations — this registers the project via `eetools.configure()` and calls `ee.Initialize()`.

---

## Commands

```bash
# Tests — see "Testing" below for the marker split
uv run pytest                  # everything (EE tests skip without a session)
uv run pytest -m "not ee"      # pure/mock tests only — fast, no credentials (what CI runs)

# Lint (ruff: E, F, I; E501 is deferred to the formatter)
uv run ruff check src/ tests/

# Format
uv run black src/ tests/
uv run black --check src/ tests/   # CI gate
uv run ruff check --fix src/ tests/   # applies import sorting (I)

# Type check
uv run mypy src/
```

CI (`.github/workflows/ci.yml`) runs ruff + `black --check` + mypy + `pytest -m "not ee"`
on Python 3.11 and 3.12.

---

## Testing

The suite is split into two tiers by pytest marker (registered in `pyproject.toml`):

- **Pure / mock tests** (no marker) — `_config`, `tables`, `plots`, `constants`,
  validation guards, and the `io` export layer (mocked so no real export tasks fire).
  These always run and need no Earth Engine session.
- **`@pytest.mark.ee`** — exercise real `ee.*` graph-building. Most run on *synthetic
  `ee.Image.constant` images* so index/QA math is checked against hand-computed values;
  a subset is also marked **`@pytest.mark.slow`** because it hits real datasets.

The `ee_session` fixture (in `tests/conftest.py`) initializes Earth Engine from
`EE_PROJECT` (or `GOOGLE_CLOUD_PROJECT`); if no authenticated session is available it
**skips** every `ee` test rather than failing — so CI stays green without credentials.

```bash
# Run the EE tests locally (requires `earthengine authenticate` + a registered project)
EE_PROJECT=<your-project> uv run pytest        # PowerShell: $env:EE_PROJECT="<project>"
uv run pytest -m "not slow"                    # synthetic EE tests, skip network datasets
uv run pytest -m "not ee"                      # pure tests only (CI mode)
```

A persisted default project (`earthengine set_project <project>`) lets the EE tests run
without setting `EE_PROJECT` each time.

---

## Architecture

A single installable package, `eetools`, under `src/` (hatchling build, `src`-layout).

```
gee_utils/
├── src/eetools/
│   ├── __init__.py          # initialize() entry point + re-exports configure/get_project
│   ├── _config.py           # module-level project_id state: configure() / get_project()
│   ├── constants.py         # ALL collection IDs, band lists, band maps, scale factors, mask thresholds
│   ├── io.py                # Export to Drive / EE assets; task status helpers
│   ├── utils.py             # date-range validation, AOI/GPKG → ee.Geometry, clipping, joins, temporal reducers, resampling
│   ├── sensors/
│   │   ├── indices.py       # all spectral-index functions + calc_indices()/calc_veg_indices() multi-index builders
│   │   ├── masking.py       # shared cloud/water mask helpers used by the optical sensors
│   │   ├── bii/             # Biodiversity Intactness Index preprocessing
│   │   ├── chirps/          # precipitation preprocessing (+ monthly/annual rainfall table export)
│   │   ├── dem/             # Copernicus DEM / terrain preprocessing
│   │   ├── dswx/            # OPERA DSWx surface water (HLS + S1) masking + preprocessing
│   │   ├── dynamicworld/    # Dynamic World LULC masking (cover-type filter) + preprocessing
│   │   ├── esa/             # WorldCover preprocessing
│   │   ├── hansen/          # Global Forest Change preprocessing
│   │   ├── hls/             # masking + preprocessing
│   │   ├── landsat/         # masking + preprocessing
│   │   ├── modis/           # preprocessing
│   │   ├── sentinel/        # Sentinel-2 SR masking + preprocessing
│   │   ├── sentinel1/       # Sentinel-1 SAR GRD masking (edge + speckle) + preprocessing
│   │   └── wdpa/            # World Database on Protected Areas (vector) preprocessing + Drive export
│   ├── landtrendr/          # METHOD subpackage (not a sensor): LandTrendr temporal segmentation
│   │   ├── collection.py    # medoid composite + multi-sensor (Roy-harmonized) annual builder
│   │   ├── segmentation.py  # run-param defaults + run_landtrendr / run_landtrendr_from_aoi
│   │   └── outputs.py       # get_change_map / get_fitted_stack / get_segment_data (array-image parsers)
│   └── visualization/       # plots.py, summaries.py, tables.py (matplotlib/seaborn/pandas)
└── tests/                   # one test_*.py per source module; conftest.py holds shared
                             # fixtures + the ee_session fixture (see "Testing")
```

**Per-sensor module pattern** (`sensors/<sensor>/`): a `masking.py` builds the cloud-free (and optional water-masked) collection, and a `preprocessing.py` exposes the public `get_<sensor>_collection(aoi, start_date, end_date, ...)` that validates the date range, applies scale factors/offsets, calls `calc_indices()` with the sensor's `*_BAND_MAP`, and returns an `ee.ImageCollection`. Follow this shape when adding a sensor.

Non-optical sensors deviate where appropriate: `sentinel1` (SAR backscatter) and the pre-classified products `dswx` / `dynamicworld` skip `calc_indices()` (no reflectance bands), and `wdpa` is a **vector** sensor — it returns an `ee.FeatureCollection`, has no `masking.py` or date validation, and its `get_*` functions filter by attribute/AOI instead.

`landtrendr/` is a **method** subpackage (not a sensor): it fuses the Landsat builders into a one-image-per-year, Roy-harmonized medoid series, runs `ee.Algorithms.TemporalSegmentation.LandTrendr`, and parses the array-image outputs. Its segmentation band must be **loss-positive** (NBR/NDVI/NDMI ×−1, tracked by `DIST_DIR`), and run-param defaults + Roy coefficients live in `constants.py`.

### Key Conventions

- **All collection IDs, band names, band maps, scale factors, offsets, and mask thresholds live in `constants.py`** — never hardcode them in sensor modules. Each sensor has a `*_BAND_MAP` dict (`{"nir": ..., "red": ...}`) that drives the generic index functions, so indices stay sensor-agnostic.
- Index functions in `indices.py` are generic (take explicit band-name args); sensor preprocessing wires them up via the band map. Add new indices there, then wire into `calc_indices`/`calc_veg_indices`.
- Public functions are fully type-hinted and carry Google-style docstrings with `Args:`/`Returns:` (including what is raised and what the EE return type is). Match this for new functions.
- Always call `eetools.initialize(project=...)` first; functions that need the project ID read it via `get_project()` and accept a `project_id` override.
- Export filenames are derived from per-image date/year properties; preserve `system:time_start` through `copyProperties` in any new map function.

### Critical Gotchas (Earth Engine)

- **Everything is lazy and server-side.** Operations build a computation graph; nothing executes until `.getInfo()` or an `Export` task. Never call `.getInfo()` inside a `.map()` — it breaks lazy evaluation and errors. Keep map functions purely server-side.
- **Use `ee.batch.Export.*`** (not `Export.*`) in Python. Always specify `scale`; add `tileScale=4` on large `reduceRegion` calls to avoid OOM. `maxPixels=1e13` is the project default (`DEFAULT_MAX_PIXELS`).
- **`reproject()` is expensive** — only at the very end of a pipeline (see `resample_pixel_resolution` in `utils.py`).
- **`normalizedDifference` masks any negative-input pixel** and names its output `'nd'` (always `.rename(...)`). Use `.expression(...)` when negatives are valid.
- **kNDVI:** the `tanh(NDVI²)` shortcut is a known oversimplification. Prefer a data-estimated σ — `calc_kndvi_est_sigma` (per-image, regional) or `calc_kndvi_temp_est_sigma` (per-pixel, temporal) — over the fixed-σ `SIGMA` constant for time-series work.
- Server-side types (`ee.Number`, `ee.String`, `ee.Dictionary`) need `.getInfo()` to read into Python; primitives auto-wrap.

---

## Write-back Protocol

If you make a significant architectural decision during this session, create a stub in the vault's `raw/notes/` directory for the wiki agent to ingest:

```markdown
---
repo: gee_utils
date: YYYY-MM-DD
type: decision | observation | bug | gap
decision: one-sentence description
rationale: why
affected_files: src/eetools/...
---
```

Path: `C:\Users\harre\Obsidian_Vaults\01_Work_Projects_Vault\raw\notes\`
