"""Ready-made Earth Engine visualization parameter dictionaries for geemap.

Each constant is a plain dict suitable for passing straight to
``geemap.Map.add_layer(layer, vis_params, name)`` (or any EE visualization call that
takes ``visParams``), so notebooks can import a styling preset instead of
reconstructing it inline.

Flavours:

* **Single-band index layers** — ``{"min", "max", "palette"}`` for the indices the
  package appends (see ``S2_INDEX_BANDS`` / ``sensors/indices.py``); ranges are the
  index value ranges (mostly -1..1).
* **RGB composites** — ``{"bands", "min", "max", "gamma"}`` referencing raw EE band
  names from ``constants.py``. ``min``/``max`` assume the **scaled-reflectance**
  output of the ``get_*_collection`` builders (which apply each sensor's scale
  factor/offset, so reflectance is ~0-1) — not raw DN.
* **Terrain** — elevation/slope/aspect/hillshade single-band layers for the
  Copernicus DEM and ``ee.Terrain`` products (see ``sensors/dem``).
* **Vector styling** — ``{"color", "width", ...}`` for site/boundary FeatureCollections.
* **Derived products** — palettes for downstream metrics (e.g. Biodiversity
  Intactness Index).

Palettes and ranges are ported from the NIP Habitat Health Metric project. They are
starter defaults; copy and tweak per AOI/season as needed.
"""

import copy

# --------------------------------------------------------------------------- #
# Single-band spectral index layers ({"min", "max", "palette"})
# --------------------------------------------------------------------------- #
NDVI_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#0000FF",  # water / very low
        "#FFFFFF",  # bare / neutral
        "#CE7E45",
        "#DF923D",
        "#F1B555",
        "#FCD163",
        "#99B718",
        "#74A901",
        "#66A000",
        "#529400",
        "#3E8601",
        "#207401",
        "#056201",
        "#004C00",  # dense vegetation
    ],
}

DNDVI_VIS_PARAMS = {
    "min": -1,
    "max": 1,
    "palette": [
        "#006400",  # Vegetation Gain
        "#008000",
        "#32CD32",
        "#FFFFFF",  # No Change
        "#DEB887",
        "#A52A2A",
        "#800000",  # Vegetation Loss
    ],
}

FPAR_VIS_PARAMS = {
    "min": 0.0,
    "max": 0.95,
    "palette": [
        "#f7fcf5",
        "#e5f5e0",
        "#c7e9c0",
        "#a1d99b",
        "#74c476",
        "#31a354",
        "#006d2c",
    ],
}

EVI_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#000080",
        "#0000FF",
        "#FFFFFF",
        "#FDE725",
        "#5DC863",
        "#21918C",
        "#3B528B",
        "#004C00",
    ],
}

NDWI_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#8B4513",  # dry / non-water
        "#F5F5F5",
        "#B0E0E6",
        "#87CEFA",
        "#1E90FF",
        "#0000FF",  # strong water
    ],
}

MNDWI_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#654321",
        "#D2B48C",
        "#F7F7F7",
        "#BFEFFF",
        "#4DA6FF",
        "#0033CC",
    ],
}

SAVI_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#440154",
        "#3B528B",
        "#21918C",
        "#5DC863",
        "#FDE725",
        "#FFFFCC",
        "#006400",
    ],
}

NDMI_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#A52A2A",  # dry
        "#F5DEB3",
        "#FFFFBF",
        "#C7E9B4",
        "#7FCDBB",
        "#41B6C4",
        "#225EA8",  # moist
    ],
}

NBR_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#0000FF",
        "#FFFFFF",
        "#FFFFB2",
        "#FECC5C",
        "#FD8D3C",
        "#E31A1C",
        "#800026",
    ],
}

NIRV_VIS_PARAMS = {
    "min": 0.0,
    "max": 0.5,
    "palette": [
        "#F7FCF5",
        "#E5F5E0",
        "#C7E9C0",
        "#A1D99B",
        "#74C476",
        "#41AB5D",
        "#238B45",
        "#005A32",
    ],
}

NDRE_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#542788",
        "#998EC3",
        "#F7F7F7",
        "#F1A340",
        "#D73027",
        "#7F0000",
        "#004C00",
    ],
}

# kNDVI (RBF form): tanh output is always [0, 1].
KNDVI_FIXED_VIS_PARAMS = {
    "min": 0.0,
    "max": 1.0,
    "palette": [
        "#FFFFFF",
        "#CE7E45",
        "#DF923D",
        "#F1B555",
        "#FCD163",
        "#99B718",
        "#74A901",
        "#66A000",
        "#529400",
        "#3E8601",
        "#207401",
        "#056201",
        "#004C00",
    ],
}

EVI2_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#000080",
        "#0000FF",
        "#FFFFFF",
        "#FDE725",
        "#5DC863",
        "#21918C",
        "#3B528B",
        "#004C00",
    ],
}

GNDVI_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#0000FF",
        "#FFFFFF",
        "#FFFFB2",
        "#D9F0A3",
        "#78C679",
        "#31A354",
        "#006837",
        "#004C00",
    ],
}

OSAVI_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#440154",
        "#3B528B",
        "#21918C",
        "#5DC863",
        "#FDE725",
        "#FFFFCC",
        "#006400",
    ],
}

# MTCI and IRECI are chlorophyll-ratio indices with unbounded positive range;
# the caps below cover typical healthy-canopy values.
MTCI_VIS_PARAMS = {
    "min": 0.0,
    "max": 10.0,
    "palette": [
        "#FFFFE5",
        "#F7FCB9",
        "#D9F0A3",
        "#ADDD8E",
        "#78C679",
        "#41AB5D",
        "#238B45",
        "#006837",
    ],
}

IRECI_VIS_PARAMS = {
    "min": 0.0,
    "max": 15.0,
    "palette": [
        "#FFFFE5",
        "#F7FCB9",
        "#D9F0A3",
        "#ADDD8E",
        "#78C679",
        "#41AB5D",
        "#238B45",
        "#006837",
    ],
}

# S2REP is the red-edge inflection point in nanometres (~700–730 nm).
# Low values indicate stress; high values indicate healthy canopy.
S2REP_VIS_PARAMS = {
    "min": 700.0,
    "max": 730.0,
    "palette": [
        "#D73027",
        "#FC8D59",
        "#FEE090",
        "#FFFFBF",
        "#E0F3F8",
        "#91BFDB",
        "#4575B4",
    ],
}

GVMI_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#A52A2A",  # dry
        "#F5DEB3",
        "#FFFFBF",
        "#C7E9B4",
        "#7FCDBB",
        "#41B6C4",
        "#225EA8",  # moist
    ],
}

# Soil indices: positive values indicate bare/exposed soil; negative indicates
# vegetated or water surfaces.
BSI_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#006400",  # vegetated
        "#74A901",
        "#FFFFBF",
        "#D2B48C",
        "#A0522D",
        "#8B4513",
        "#4A2500",  # bare soil
    ],
}

# MBI has a +0.5 offset baked in; practical range is ~0–1 (bare soil ≈ 0.5+).
MBI_VIS_PARAMS = {
    "min": 0.0,
    "max": 1.0,
    "palette": [
        "#006400",
        "#74A901",
        "#FFFFBF",
        "#D2B48C",
        "#A0522D",
        "#8B4513",
    ],
}

EMBI_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#006400",
        "#74A901",
        "#FFFFBF",
        "#D2B48C",
        "#A0522D",
        "#8B4513",
    ],
}

DBSI_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#006400",
        "#74A901",
        "#FFFFBF",
        "#D2B48C",
        "#A0522D",
        "#8B4513",
        "#4A2500",
    ],
}

NBR2_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#0000FF",
        "#FFFFFF",
        "#FFFFB2",
        "#FECC5C",
        "#FD8D3C",
        "#E31A1C",
        "#800026",
    ],
}

# BAI is unbounded (1 / distance² from a reference burn point); cap at 500 for
# display — most fire-affected pixels fall in the 50–500 range.
BAI_VIS_PARAMS = {
    "min": 0.0,
    "max": 500.0,
    "palette": [
        "#FFFFFF",
        "#FFFFB2",
        "#FECC5C",
        "#FD8D3C",
        "#F03B20",
        "#BD0026",
        "#800026",
    ],
}

# BAIS2 output typically spans ~0–3 over burned/non-burned surfaces (S2 only).
BAIS2_VIS_PARAMS = {
    "min": 0.0,
    "max": 3.0,
    "palette": [
        "#FFFFFF",
        "#FFFFB2",
        "#FECC5C",
        "#FD8D3C",
        "#F03B20",
        "#BD0026",
        "#800026",
    ],
}

# Urban indices: positive values indicate built-up / impervious surfaces.
NDBI_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#006400",  # natural / non-urban
        "#90EE90",
        "#FFFFFF",
        "#C8C8C8",
        "#888888",
        "#444444",
        "#1A1A1A",  # dense urban
    ],
}

UI_VIS_PARAMS = {
    "min": -1.0,
    "max": 1.0,
    "palette": [
        "#006400",
        "#90EE90",
        "#FFFFFF",
        "#C8C8C8",
        "#888888",
        "#444444",
        "#1A1A1A",
    ],
}

# --------------------------------------------------------------------------- #
# Terrain (Copernicus DEM GLO-30 / ee.Terrain) — single-band layers
# --------------------------------------------------------------------------- #
# Elevation in metres; terrain ramp (low blue -> high white). Tune max per AOI.
ELEVATION_VIS_PARAMS = {
    "min": 0.0,
    "max": 3000.0,
    "palette": [
        "#333399",
        "#00a2e5",
        "#55dd77",
        "#ffff99",
        "#aa926b",
        "#aa928d",
        "#ffffff",
    ],
}

# Slope in degrees (0 flat -> steep); 0-60 covers most terrain.
SLOPE_VIS_PARAMS = {
    "min": 0.0,
    "max": 60.0,
    "palette": ["#ffffff", "#fdae61", "#f46d43", "#d73027", "#a50026"],
}

# Aspect in degrees (0-360, cyclic); rainbow with matching ends for wrap-around.
ASPECT_VIS_PARAMS = {
    "min": 0.0,
    "max": 360.0,
    "palette": [
        "#ff0000",
        "#ffff00",
        "#00ff00",
        "#00ffff",
        "#0000ff",
        "#ff00ff",
        "#ff0000",
    ],
}

# Hillshade is an unsigned byte (0-255); greyscale.
HILLSHADE_VIS_PARAMS = {
    "min": 0,
    "max": 255,
    "palette": ["#000000", "#ffffff"],
}

# --------------------------------------------------------------------------- #
# RGB composites ({"bands", "min", "max", "gamma"})
# --------------------------------------------------------------------------- #
# Sentinel-2 SR (band names from constants.S2_BANDS).
S2_TRUE_COLOR_VIS_PARAMS = {
    "bands": ["B4", "B3", "B2"],
    "min": 0.02,
    "max": 0.35,
    "gamma": 1.2,
}

# Landsat 8 C2 L2 SR (band names from constants.L8_BANDS).
L8_TRUE_COLOR_VIS_PARAMS = {
    "bands": ["SR_B4", "SR_B3", "SR_B2"],
    "min": 0.0,
    "max": 0.3,
    "gamma": 1.2,
}

# NASA HLS (renamed common band names from constants.HLS_COMMON_BANDS).
HLS_TRUE_COLOR_VIS_PARAMS = {
    "bands": ["RED", "GREEN", "BLUE"],
    "min": 0.02,
    "max": 0.35,
    "gamma": 1.2,
}

# --------------------------------------------------------------------------- #
# Vector / feature styling
# --------------------------------------------------------------------------- #
# Site / AOI boundary outlines (red outline, translucent red fill).
SITES_VIS_PARAMS = {
    "color": "#FF0000",
    "width": 2,
    "lineType": "solid",
    "fillColor": "#FF00006F",
}

# --------------------------------------------------------------------------- #
# Derived products
# --------------------------------------------------------------------------- #
# Biodiversity Intactness Index — continuous (0-100) and class (1-4) renderings.
BII_VIS_PARAMS = {
    "min": 0,
    "max": 100,
    "palette": ["#d73027", "#fee08b", "#d9ef8b", "#1a9850"],
}

BII_CLASS_VIS_PARAMS = {
    "min": 1,
    "max": 4,
    "palette": ["#d73027", "#fee08b", "#d9ef8b", "#1a9850"],
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def change_vis_min_max(
    vis_params: dict,
    new_min: float,
    new_max: float,
) -> dict:
    """Return a copy of a visualization-params dict with new min/max stretch values.

    Use this to re-stretch a preset (e.g. NDVI_VIS_PARAMS) for a specific AOI/index
    without authoring a new dict. The input dict is not modified — the shared module-level
    presets are safe to pass in.

    Args:
        vis_params: A visualization-params dict (e.g. one of the *_VIS_PARAMS presets), typically containing 'min', 'max', and 'palette'.
        new_min: New minimum stretch value.
        new_max: New maximum stretch value.

    Returns:
        A new dict identical to vis_params but with 'min' and 'max' set to the new values.

    Raises:
        ValueError: If new_min is not less than new_max.
    """
    if new_min >= new_max:
        raise ValueError(f"new_min ({new_min}) must be less than new_max ({new_max})")

    updated = copy.deepcopy(vis_params)
    updated["min"] = new_min
    updated["max"] = new_max
    return updated
