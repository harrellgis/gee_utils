# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Session Start — Load This Context

At the start of every session, load context in this order:

1. **Vault overview** — `C:\Users\harre\Repos\Obsidian\01_Work_Projects_Vault\context\agent-onboarding.md`
2. **Coding conventions** — `C:\Users\harre\Repos\Obsidian\01_Work_Projects_Vault\context\conventions.md`
3. **Tool references** — read the wiki pages for the sensors this repo wraps:
   - `wiki/tools/google-earth-engine.md` (core API patterns and gotchas)
   - `wiki/tools/sentinel-2-sr.md`, `wiki/tools/sentinel-2-cloud-probability.md`
   - `wiki/tools/landsat-c2-sr.md`
   - `wiki/tools/hls-hlsl30.md`, `wiki/tools/hls-hlss30.md`
   - `wiki/tools/chirps.md`, `wiki/tools/dynamic-world.md`
   - (all under `C:\Users\harre\Repos\Obsidian\01_Work_Projects_Vault\`)
4. **Decisions registry** — `C:\Users\harre\Repos\Obsidian\01_Work_Projects_Vault\context\decisions-registry.md`
5. **Tag cluster** — `C:\Users\harre\Repos\Obsidian\01_Work_Projects_Vault\context\tags.md` (read the **Optical EO & Cloud Masking** section only)

---

## What This Repo Does

`eetools`: reusable Google Earth Engine utilities for multi-sensor environmental remote sensing (masked collections, harmonized spectral indices, temporal reduction, zonal stats/site workflows, export, visualization). Installed as a dependency by downstream analysis projects/notebooks, not run standalone.

---

## Setup & Testing

`uv`-managed, src-layout package. Earth Engine must be authenticated once per machine (`earthengine authenticate`); in code call `eetools.initialize(project="my-gee-project")` before any EE operation.

Tests are split by pytest marker: unmarked = pure/mock (no EE session); `@pytest.mark.ee` = real `ee.*` graph-building, mostly on synthetic `ee.Image.constant` images checked against hand-computed values; `@pytest.mark.slow` = hits real datasets. The `ee_session` fixture (`tests/conftest.py`) initializes from `EE_PROJECT` (or `GOOGLE_CLOUD_PROJECT`, or a persisted `earthengine set_project`) and **skips** — not fails — every `ee` test when no session is available.

```bash
uv run pytest -m "not ee"                      # pure tests only — what CI runs
EE_PROJECT=<project> uv run pytest             # PowerShell: $env:EE_PROJECT="<project>"
uv run pytest -m "not slow"                    # synthetic EE tests, skip network datasets
uv run ruff check --fix src/ tests/            # ruff E,F,I (E501 deferred to black)
```

CI gates: ruff, `black --check`, mypy, `pytest -m "not ee"` on Python 3.11 and 3.12.

---

## Architecture

Single package `eetools` under `src/`. Per-sensor conventions live in `src/eetools/sensors/CLAUDE.md`; the LandTrendr method subpackage in `src/eetools/landtrendr/CLAUDE.md`; the TGBS reference/degraded site-selection workflows (`identify_reference_sites`, `identify_degraded_sites`) in `src/eetools/site_selection/CLAUDE.md`.

### Key Conventions

- **All collection IDs, band names, band maps, scale factors, offsets, and mask thresholds live in `constants.py`** — never hardcode them in sensor modules. Each sensor has a `*_BAND_MAP` dict (`{"nir": ..., "red": ...}`) that drives the generic index functions, so indices stay sensor-agnostic.
- The built-in water masks (`build_non_water_mask`) use fixed spectral thresholds; for a per-scene split of a bimodal band (NDWI/MNDWI, SAR dB, dNBR, index differences) use `thresholding.edge_otsu_threshold`. Its Canny defaults (`CANNY_THRESHOLD`, `CANNY_SIGMA`) are in the band's units and tuned for normalized-difference indices; set `fallback` per index (returned when no edges are found).
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

Path: `C:\Users\harre\Repos\Obsidian\01_Work_Projects_Vault\raw\notes\`
