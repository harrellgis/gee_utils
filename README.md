# eetools

Reusable [Google Earth Engine](https://earthengine.google.com/) utilities for
satellite data processing and analysis. `eetools` wraps multi-sensor imagery
behind a consistent API: build cloud- and water-masked image collections, append
a harmonized set of spectral indices, reduce over time, export to Drive or Earth
Engine assets, and summarize/visualize the results.

Supported sources: **Sentinel-2 SR**, **Landsat 5/7/8/9 C2 L2 SR**, **NASA HLS**
(L30/S30), **MODIS** MCD15A3H LAI/FPAR, **CHIRPS** precipitation,
**ESA WorldCover**, **Sentinel-1 SAR GRD**, **OPERA DSWx** surface water,
**Dynamic World** LULC, **Copernicus DEM**/terrain, **Hansen** Global Forest
Change, **BII** (Biodiversity Intactness Index), **WDPA** protected-area
vectors, and **Google Satellite Embedding V1** (AlphaEarth Foundations) annual
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
uv pip install git+https://github.com/<owner>/gee_utils.git
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
uv pip install "eetools[notebook] @ git+https://github.com/<owner>/gee_utils.git"
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

Use `build_period_composites` to reduce a collection to one composite per year or
month, or `build_seasonal_composites` to produce one composite per year restricted
to a recurring window of months (e.g. a wet or dry season):

```python
from eetools.visualization.summaries import build_seasonal_composites
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

## Project layout

```
src/eetools/
├── _config.py        # global project-id state
├── constants.py      # collection IDs, band maps, scale factors, mask thresholds
├── io.py             # exports to Drive / EE assets, task status helpers
├── utils.py          # geometry/GPKG helpers, date validation, reducers, resampling
├── sensors/          # per-sensor masking.py + preprocessing.py, plus indices.py
├── landtrendr/       # LandTrendr temporal-segmentation method (collection/segmentation/outputs)
└── visualization/    # plots.py, summaries.py, tables.py, vis_params.py
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
