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
* **Vector styling** — ``{"color", "width", ...}`` for site/boundary FeatureCollections.
* **Derived products** — palettes for downstream metrics (e.g. Biodiversity
  Intactness Index).

Palettes and ranges are ported from the NIP Habitat Health Metric project. They are
starter defaults; copy and tweak per AOI/season as needed.
"""

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
