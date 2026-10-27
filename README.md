# eetools

Reusable [Google Earth Engine](https://earthengine.google.com/) utilities for
satellite data processing and analysis. `eetools` wraps multi-sensor imagery
behind a consistent API: build cloud- and water-masked image collections, append
a harmonized set of spectral indices, reduce over time, export to Drive or Earth
Engine assets, and summarize/visualize the results. On top of the sensor builders
it provides temporal compositing, zonal statistics over site polygons, local
vector-file loading, and end-to-end site workflows.

Supported sources: **Sentinel-2 SR**, **Landsat 5/7/8/9 C2 L2 SR**, **NASA HLS**
(L30/S30), **MODIS** (MCD15A3H LAI/FPAR, MOD13Q1 NDVI/EVI, MCD64A1 burned area,
MOD17A3HGF NPP, MOD44B vegetation continuous fields), **NASA SMAP** L4 soil
moisture, **CHIRPS** precipitation, **OpenLandMap SM2RAIN** monthly precipitation
climatology, **NASA Black Marble VIIRS** (VNP46A2) daily nighttime lights,
**ESA WorldCover**, **Sentinel-1 SAR GRD**, **OPERA DSWx** surface water,
**Dynamic World** LULC, **Copernicus DEM**/terrain, **Hansen** Global Forest
Change, **Meta/WRI 1 m canopy height**, **iSDA** Africa topsoil carbon, **BII**
(Biodiversity Intactness Index), **WDPA** protected-area vectors, and **Google Satellite Embedding V1** (AlphaEarth Foundations) annual
64-D embeddings. It also includes a **LandTrendr** temporal-segmentation method
subpackage built on the harmonized Landsat builders.

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
seasonal composites.

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
├── workflows.py      # end-to-end site timeseries and baseline assessment
├── sensors/          # per-sensor masking.py + preprocessing.py, plus indices.py,
│                     #   masking.py (shared helpers), and texture.py
├── landtrendr/       # LandTrendr temporal-segmentation method (collection/segmentation/outputs)
└── visualization/    # plots.py, tables.py, vis_params.py
```

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
