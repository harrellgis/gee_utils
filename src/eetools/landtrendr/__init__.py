"""LandTrendr annual spectral-temporal segmentation for Landsat (LT-GEE).

A method subpackage (not a sensor): build a one-image-per-year, multi-sensor harmonized
medoid collection, run ``ee.Algorithms.TemporalSegmentation.LandTrendr``, and unpack the
array-image outputs into change maps, fitted annual stacks, and per-segment attributes.
"""

from eetools.landtrendr.collection import (
    build_annual_composite,
    build_landtrendr_collection,
    medoid_composite,
)
from eetools.landtrendr.outputs import (
    get_change_map,
    get_fitted_stack,
    get_segment_count,
    get_segment_data,
)
from eetools.landtrendr.segmentation import (
    resolve_run_params,
    run_landtrendr,
    run_landtrendr_from_aoi,
)

__all__ = [
    "medoid_composite",
    "build_annual_composite",
    "build_landtrendr_collection",
    "resolve_run_params",
    "run_landtrendr",
    "run_landtrendr_from_aoi",
    "get_segment_data",
    "get_segment_count",
    "get_change_map",
    "get_fitted_stack",
]
