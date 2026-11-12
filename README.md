# eetools

Reusable [Google Earth Engine](https://earthengine.google.com/) utilities for
satellite data processing and analysis. `eetools` wraps multi-sensor imagery
behind a consistent API: build cloud- and water-masked image collections, append
a harmonized set of spectral indices, reduce over time, export to Drive or Earth
Engine assets, and summarize/visualize the results. On top of the sensor builders
it provides temporal compositing, zonal statistics over site polygons, local
vector-file loading, end-to-end site workflows, and automated selection of TGBS
reference and degraded sites.

Supported sources:

- **Optical and SAR:** Sentinel-2 SR, Landsat 5/7/8/9 C2 L2 SR, NASA HLS
  (L30/S30), Sentinel-1 SAR GRD.
- **MODIS:** MCD15A3H LAI/FPAR, MOD13Q1 NDVI/EVI, MCD43A4 NBAR (NDMI/NBR),
  MCD64A1 burned area, MOD17A3HGF NPP, MOD44B vegetation continuous fields.
- **Climate and water:** CHIRPS v3 precipitation (daily or pentad), TerraClimate,
  WorldClim V1 bioclim, OpenLandMap SM2RAIN precipitation climatology, NASA SMAP L4
  soil moisture, OPERA DSWx surface water, MERIT Hydro HAND.
- **Land cover and condition:** ESA WorldCover, Dynamic World, Hansen Global Forest
  Change, Meta/WRI 1 m canopy height, TNC Global Human Modification, BII
  (Biodiversity Intactness Index), Global Mangrove Watch,
  NASA Black Marble VIIRS (VNP46A2) nighttime lights, Google Satellite Embedding V1
  (AlphaEarth Foundations) annual 64-D embeddings.
- **Terrain, soil and ecology:** Copernicus DEM/terrain (incl. TPI), iSDA Africa soil
  carbon and texture, SoilGrids WRB reference soil groups, VECEA potential natural
  vegetation of eastern Africa, OpenLandMap PNV biomes, RESOLVE Ecoregions, WDPA
  protected-area vectors.

It also includes a **LandTrendr** temporal-segmentation method subpackage built on
the harmonized Landsat builders, and a `thresholding` module for adaptive
(edge-based Otsu) thresholds.

## Requirements

- Python ≥ 3.11
- A Google Earth Engine account, authenticated on the machine, and a registered
  Cloud project.

## Installation

This project is managed with [uv](https://docs.astral.sh/uv/).

```bash
# Clone, then create the environment and install (with dev dependencies)
uv sync
```

To use `eetools` as a dependency in another project, install it directly:

```bash
uv pip install -e /path/to/gee_utils      # editable
# or
uv pip install git+https://github.com/harrellgis/gee_utils.git
```

### Optional dependencies

Extra features live behind [optional extras](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#extras) so the base install stays lean:

| Extra | Adds | For |
| --- | --- | --- |
| `notebook` | [`geemap`](https://geemap.org/) | Interactive Earth Engine map visualization in Jupyter notebooks |

```bash
# Local development — include an extra (or use --all-extras)
uv sync --extra notebook

# As a dependency in another project
uv pip install -e "/path/to/gee_utils[notebook]"
# or
uv pip install "eetools[notebook] @ git+https://github.com/harrellgis/gee_utils.git"
```

## Authentication

Authenticate once per machine, then initialize in code with your project:

```bash
uv run earthengine authenticate
uv run earthengine set_project <your-project>   # optional: persists the default
```

```python
import eetools

eetools.initialize(project="your-project")   # registers the project + ee.Initialize()
```

`initialize()` stores the project via `eetools.configure()`; functions that need
it (e.g. asset exports) read it back through `eetools.get_project()`.

## Quickstart

```python
import ee
import eetools
from eetools.sensors.sentinel.preprocessing import get_s2_sr_collection
from eetools.io import export_image_to_drive

eetools.initialize(project="your-project")

aoi = ee.Geometry.Rectangle([39.20, -4.30, 39.25, -4.25])
start, end = ee.Date("2021-06-01"), ee.Date("2021-09-01")

# Cloud- and water-masked Sentinel-2 SR with spectral indices appended
collection = get_s2_sr_collection(aoi, start, end)

# Median composite, exported to Google Drive
composite = collection.select("NDVI").median().clip(aoi)
task = export_image_to_drive(
    image=composite,
    aoi=aoi,
    description="ndvi_median",
    folder="eetools_exports",
    file_prefix="ndvi_median",
    scale=10,
)
print(task.status())
```

By default each collection carries a harmonized core of indices. To choose exactly
which indices each image gets, pass `indices` (by name) and/or `domains` (whole
families: `vegetation`, `water`, `moisture`, `soil`, `burn`, `urban`). Selection is
sensor-aware — red-edge indices (MTCI, IRECI, S2REP, BAIS2) compute on Sentinel-2 and
are silently skipped on sensors without a red edge.

```python
# Whole domains — here without water masking, since these families omit MNDWI.
veg_burn = get_s2_sr_collection(
    aoi, start, end, domains=["vegetation", "burn"], apply_water_masking=False
)

# Specific indices by name. Water masking is on by default, so the selection must
# include NDVI and MNDWI (both are read by the spectral water mask).
named = get_s2_sr_collection(
    aoi, start, end, indices=["NDVI", "MNDWI", "NDBI", "BAIS2"]
)
```

Indices come from a central registry (`eetools.sensors.indices.INDEX_REGISTRY`); the
same `indices`/`domains` arguments are accepted by `calc_indices()` directly. Adding a
new index is a single registry entry.

Use `eetools.compositing.build_period_composites` to reduce a collection to one composite per year or
month, or `build_seasonal_composites` to produce one composite per year restricted
to a recurring window of months (e.g. a wet or dry season):

```python
from eetools.compositing import build_seasonal_composites
from eetools.constants import SEASON_WET  # (3, 5) — Mar–May; override per region

# One median composite per year covering the wet-season months (Mar–May)
wet_season = build_seasonal_composites(
    collection=collection,
    bands=["NDVI"],
    start_year=2018,
    end_year=2023,
    season_months=SEASON_WET,
    season_name="wet",
    composite_stat="median",
)
# Each image carries year, season, season_months, image_count, composite_stat properties
```

Each sensor exposes a `get_<sensor>_collection(aoi, start_date, end_date, ...)`
builder (`get_l8_sr_collection`, `get_hls_merged_collection`,
`get_chirps_collection`, `get_modis_lai_fpar_col`, …) that validates the date
range, applies scale factors, and appends indices via the sensor's band map.
Static layers (`get_terrain`, `get_canopy_height`, `get_soil_carbon`,
`get_land_cover`, `get_bii`, …) take an optional AOI and return a single
`ee.Image`.

### Sites, zonal statistics, and workflows

Site boundaries are usually local files. `eetools.vectors` turns any format
geopandas can read (`.gpkg`, `.geojson`, `.shp`, …) into Earth Engine features with
`site_id` / `site_name` metadata:

```python
from eetools.vectors import vector_files_to_feature_collection

sites = vector_files_to_feature_collection([
    ("boundaries/site_a.gpkg", "A", "Site A"),
    ("boundaries/site_b.geojson", "B", "Site B"),
])
```

`eetools.zonal` reduces images and collections over those regions
(`reduce_image_over_region`, `image_collection_to_region_stats_fc`,
`image_collection_to_sample_fc`, `summarize_class_areas`,
`summarize_collection_histograms`) and returns flat `ee.FeatureCollection`s that
export cleanly as tables.

`eetools.workflows` chains these steps into one call:

```python
from functools import partial
from eetools.workflows import run_site_timeseries, run_baseline_assessment

# Annual median NDVI/EVI per site -> CSV in Drive
builder = partial(get_s2_sr_collection, indices=["NDVI", "MNDWI", "EVI"])
stats = run_site_timeseries(
    sites_fc=sites,
    collection_builder=builder,          # (aoi, start, end) -> ee.ImageCollection
    bands=["NDVI", "EVI"],
    start_date="2020-01-01",
    end_date="2024-01-01",
    export_folder="eetools_exports",
    file_prefix="s2_annual",
    scale=10,
)

# Static baseline: terrain, canopy height, land cover, soil carbon, BII,
# Hansen forest -> per-site summary CSV plus one GeoTIFF per layer
layers, summary = run_baseline_assessment(sites, export_folder="eetools_baseline")
```

`run_seasonal_site_timeseries` does the same as `run_site_timeseries` with
seasonal composites. `scale` is required for all site timeseries functions: choose
the sensor's native resolution (e.g. 10 m for Sentinel-2, 30 m for HLS/Landsat).

`get_chirps_collection` takes `cadence="daily"` (default) or `"pentad"`. Pentad totals
over a year or more match the daily product at about 5x fewer image reads, so use
`"pentad"` for annual or multi-year sums and `"daily"` for short windows.

### Reference and degraded site selection

`eetools.site_selection` proposes rank-ordered TGBS **reference** sites (most intact
first) and **degraded** (zero-star) sites (most degraded first) for a focal site. The
criteria follow the TGBS Manual §2.5/§4.1–4.2 and the CER-K reference-site SOP. Both
entry points are also importable from `eetools.workflows`.

```python
from eetools.site_selection import (
    ScreeningTables,
    export_site_selection,
    identify_degraded_sites,
    identify_reference_sites,
    result_to_feature_collection,
)

ref = identify_reference_sites("boundaries/focal.gpkg", auto_seeds="wdpa_strict")
deg = identify_degraded_sites(
    "boundaries/focal.gpkg",
    project_start_year=2023,       # signature baseline 2022; anchor window 2020-2022
    screening=ref.screening,       # reuse the reference run's reductions
    reference_result=ref,          # 100% value for focal_baseline_pct
)

ref.ranked                # GeoDataFrame: rank, role, rank_score, intactness, ...
ref.pool                  # every final cell: pool_status, gate_reason, landuse_class;
                          #   eligible cells also carry mandatory_met + SOP <criterion>_score
ref.selection_matrix      # CER-K SOP 0/1/2 matrix with evidence and H/M/L confidence
ref.layer_register        # SOP §7 layer and query register
ref.top_candidates        # best 10 distinct eligible sites by condition + similarity, with
                          #   condition flags and SOP status shown (never filtered); set
                          #   top_candidates_n / top_candidates_condition_weight on the call
ref.warnings              # ladder expansion, relaxed gates, fallbacks, ...

ref.screening.to_parquet("outputs/screening")   # reload: ScreeningTables.from_parquet
export_site_selection(ref, "outputs/tables", project="TGBS", site="ITF",
                      vector_folder="outputs/vectors")
# -> TGBS_ITF_reference_candidates_<date>_v1.gpkg, _reference_pool_..., _reference_top10_...
#    (vectors, each with site_role) plus ranked / selection-matrix / layer-register CSVs
```

How it works:

- **Hard filters.** Candidates must match the focal site's national vegetation class
  (VECEA) and soil group (SoilGrids WRB), be within ±300 m elevation and ±25% rainfall
  (SOP defaults), and share its HAND class and coastal setting. Outside VECEA or WRB
  coverage the class gate falls back to PNV biome + RESOLVE ecoregion and the soil gate
  to iSDA texture, at lower confidence.
- **Environmental similarity.** MESS on condition-independent predictors (terrain,
  climate, soil texture, HAND).
- **Land-use history (absolute, shared by both workflows).** One Dynamic World
  reduction per final cell gives the yearly label history from 2019 to `end_year`, and
  each cell gets a `landuse_class`: `persistent_natural`, `converted` (with mean
  conversion year), `persistent_modified`, `built`, `mixed` or `no_dw`. **Reference
  cells must be persistently natural** (`tolerances.min_natural_persist` of the cell
  natural in all but one year of the history, or every year with
  `natural_persist_years="all"`).
  The gate runs inside each search-ladder rung, before scoring, thinning and the stop
  rule, so a thin gated pool widens the search. Natural cover on final cells is the
  pixel-level agreement of WorldCover natural and Dynamic World persistent natural
  (corrects cropland that WorldCover maps as shrub or grass).
- **Condition.** Within the eligible cells, intactness or degradation from MODIS
  NDVI/NDMI/NBR state and trends, a 250 m fraction image (natural cover, Hansen loss,
  core area), gHM, Dynamic World natural persistence and WDPA. Percentiles only order
  cells; eligibility and the SOP integrity criterion are absolute: forest, riverine and
  bamboo classes need median canopy height >= 5 m, other vegetation needs the land-use
  gate and gHM < 0.4, so intact does not mean "more trees".
- **Degraded sites are the 0% anchor** of the TGBS rescale (reference = 100%,
  degraded = 0%), not look-alikes of the focal baseline. Fine cells go through
  constrain → gate → rank:
  - **anchor constraint:** on NDVI and NDMI, a candidate must be no healthier than the
    focal site in the 3 years before `project_start_year`, and now no healthier than
    the focal site was at either time (`below_baseline`);
  - **pathway test:** for conversion pathways (`generic`, `cropland`, `settlement`,
    `mining`, `bare_soil`, `deforestation`) at least half the cell must have been
    converted (Dynamic World natural -> crops, built or bare) by the end of the baseline
    window, or be persistently modified. Such cells can never be references. For
    pathways inside natural cover (`overgrazing`, `rangeland`) the Satellite Embedding
    similarity to the focal site's pre-project year (or a rule-based profile) must
    reach `pathway_gate_pct`, and cells near the ranked references are dropped. The
    signature never ranks;
  - **no built-up land** (except for `degradation_type="settlement"`): WorldCover
    built-up <= `max_built_frac` (0.02) and a Dynamic World built-label share <=
    `max_dw_built` (0.05) on every final cell, before thinning. The limits tolerate
    scattered smallholder homesteads; cropland stays acceptable;
  - **ranking** on state + trajectory only. gHM pressure is scored only for
    `settlement`, so the ranking does not favour peri-urban cells.

  Each ranked site reports `focal_baseline_pct_ndvi` / `_ndmi` (where the focal
  baseline falls on that site's 0–100% scale) and `anchor_check="review"` above
  `anchor_review_pct`. Repeat the anchor check on the survey sensor before rescaling.
- **Search order.** A coarse screening grid in the focal UTM zone is refined into
  focal-sized cells, and the search widens through tiers (25 / 50 / 150 km) only when
  too few eligible candidates pass, or (references) when a management unit (focal
  class) has none. If the ladder is exhausted, the best gated-out cells fill the list
  with `fallback=True` and their `gate_reason`. Reference shortlists are chosen to span
  the focal site's environmental range.
- **Top candidates.** Besides the shortlist, each result holds the 10 best distinct
  eligible sites by `top_score`, the mean of the condition percentile (intactness or
  degradation) and `env_sim`, ties broken by `rank_score` and spaced like the shortlist.
  `proposed` flags the shortlisted ones. Use it to swap a proposed site that fails the
  basemap or field check.
- **Earth Engine use.** Every reduction is memoised in `ScreeningTables`, and gating,
  scoring and ranking run in pandas. Reusing the tables (in memory or from parquet)
  costs no further EECU and gives identical results. Batches are sent in spatial
  order and split automatically when Earth Engine reports "Object too large".

VECEA and SoilGrids WRB are uploaded assets, not catalog datasets. Their IDs resolve as
explicit argument → `EETOOLS_VECEA_ASSET` / `EETOOLS_SOILGRIDS_WRB_ASSET` environment
variable → the default in `constants.py`. Pass `asset_overrides={"vecea": ...}` to use
your own copy; `constants.VECEA_REPRODUCE_STEPS` explains how to rebuild VECEA (its
licence terms are unresolved, so the default asset is private).

Outputs are *proposed* sites for ecologist confirmation, not final designations.
Confirm them with the survey sensor by passing `result_to_feature_collection(result)`
to `run_site_timeseries` at the sensor's native scale.
`examples/site_selection_handoff.ipynb` walks through this hand-off, the baseline vs
end-year change terms, Dynamic World conversion pressure and the CHIRPS rain ratio.

### Exports and uploads

`eetools.io` wraps `ee.batch.Export` for single images, image lists, whole
collections (`export_image_collection_to_drive`, with filenames taken from each
image's date), per-site collections, assets, and tables. `check_ee_task_status`
polls a task by ID. `upload_geotiff_to_asset` and `upload_shapefile_to_asset`
ingest local files as Earth Engine assets. They stage the files in a Cloud Storage
bucket you name, using the credentials Earth Engine already holds.

### Texture

`eetools.sensors.texture` computes GLCM (Haralick) and first-order texture on any band,
typically a vegetation index on a same-season composite. Bands are quantized against a
fixed value range (`TEXTURE_VALUE_RANGES`, overridable per band), because `glcmTexture`
only accepts integers. Output bands are named `<band>_<metric>_r<radius>`.

```python
from eetools.sensors.texture import calc_texture

s2 = get_s2_sr_collection(aoi.buffer(1000), start, end, indices=["EVI"])  # buffer >= window half-width
evi = s2.select("EVI").median()

# Habitat-heterogeneity covariates: homogeneity + entropy at two ground windows
tex = calc_texture(evi, ["EVI"], window_m=[50, 250], scale_m=10, mask=habitat_mask)
# -> EVI_idm_r2, EVI_ent_r2, EVI_idm_r12, EVI_ent_r12

# Extra classification features
feats = stack.addBands(
    calc_texture(evi, ["EVI"], window_m=50, scale_m=10,
                 glcm_metrics=("idm", "contrast"), first_order=("sd",))
)
tex = tex.clip(aoi)  # clip AFTER texturing
```

The window is `radius` pixels of the *requested* scale, so export or sample at
`scale_m`. To view or sample at another scale, pass `pin_scale=True, crs=...`. This
reprojects the output so the window keeps its ground size, but costs about 55% more
compute. Texture within `radius` px of a masked area is computed from fewer pixels, so
treat it with care.

## Project layout

```
src/eetools/
├── __init__.py       # initialize() / configure() / get_project()
├── _config.py        # global project-id state
├── constants.py      # collection IDs, band maps, scale factors, mask thresholds, seasons
├── compositing.py    # single, period (annual/monthly), and seasonal composites
├── io.py             # exports to Drive / EE assets, task status, GeoTIFF/shapefile uploads
├── utils.py          # date validation/windows, clipping, joins, reducers, resampling
├── vectors.py        # local vector files -> ee.Geometry / site FeatureCollections
├── zonal.py          # region reductions, sampling, class areas, histograms
├── thresholding.py   # Otsu / edge-based Otsu adaptive thresholds
├── workflows.py      # end-to-end site timeseries, baseline assessment, site-selection entry points
├── sensors/          # per-sensor masking.py + preprocessing.py, plus indices.py,
│                     #   masking.py (shared helpers), texture.py, and assets.py
│                     #   (uploaded-asset ID resolution and pre-flight checks)
├── site_selection/   # TGBS reference/degraded site selection: domain, predictors,
│                     #   similarity, scoring, types, workflows
├── landtrendr/       # LandTrendr temporal-segmentation method (collection/segmentation/outputs)
└── visualization/    # plots.py, tables.py, vis_params.py
examples/
└── site_selection_handoff.ipynb   # site selection -> survey-sensor confirmation
```

Per-subpackage conventions for contributors live in `CLAUDE.md` files (root,
`sensors/`, `site_selection/`, `landtrendr/`).

## Development

```bash
uv run ruff check src/ tests/      # lint
uv run black --check src/ tests/   # format check
uv run mypy src/                   # type check
uv run pytest                      # full test suite
```

### Testing

Tests are split by pytest marker:

| Command | What runs |
|---|---|
| `uv run pytest -m "not ee"` | Pure/mock tests only — fast, **no credentials needed** (this is what CI runs) |
| `uv run pytest -m "not slow"` | Adds synthetic-image Earth Engine tests; skips real-dataset network tests |
| `uv run pytest` | Everything |

Earth-Engine-marked tests need an authenticated session. They read the project
from `EE_PROJECT` (or `GOOGLE_CLOUD_PROJECT`) and **skip cleanly** when no session
is available, so the pure tests still run anywhere:

```bash
EE_PROJECT=your-project uv run pytest          # bash
$env:EE_PROJECT="your-project"; uv run pytest  # PowerShell
```

Continuous integration (`.github/workflows/ci.yml`) runs ruff, `black --check`,
mypy, and `pytest -m "not ee"` on Python 3.11 and 3.12.

See [CHANGELOG.md](CHANGELOG.md) for release history.
