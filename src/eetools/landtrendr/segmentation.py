import ee

from eetools.constants import LANDTRENDR_DEFAULT_RUN_PARAMS
from eetools.landtrendr.collection import build_landtrendr_collection


def resolve_run_params(run_params: dict | None = None) -> dict:
    """Merge user-supplied run parameters over the LandTrendr defaults.

    Pure dict merge (no Earth Engine) so it is cheap to unit-test.

    Args:
        run_params: Optional overrides for any of the 8 LandTrendr run parameters.

    Returns:
        A new dict of run parameters: LANDTRENDR_DEFAULT_RUN_PARAMS updated with run_params.
    """
    resolved = dict(LANDTRENDR_DEFAULT_RUN_PARAMS)
    if run_params:
        resolved.update(run_params)
    return resolved


def run_landtrendr(
    annual_collection: ee.ImageCollection, run_params: dict | None = None
) -> ee.Image:
    """Run the LandTrendr temporal-segmentation algorithm on an annual collection.

    Args:
        annual_collection: One-image-per-year collection (output of build_landtrendr_collection); band 1 is the loss-positive segmentation index.
        run_params: Optional overrides for the 8 run parameters (merged over the defaults).

    Returns:
        ee.Image LandTrendr output (an array image with the 'LandTrendr' segmentation band, 'rmse', and any '<band>_fit' FTV bands).
    """
    params = resolve_run_params(run_params)
    params["timeSeries"] = annual_collection
    return ee.Algorithms.TemporalSegmentation.LandTrendr(**params)


def run_landtrendr_from_aoi(
    aoi: ee.Geometry,
    start_year: int,
    end_year: int,
    segmentation_index: str = "NBR",
    ftv_indices: list[str] | None = None,
    start_day: str = "01-01",
    end_day: str = "12-31",
    sensors: tuple[str, ...] = ("L5", "L7", "L8", "L9"),
    run_params: dict | None = None,
) -> ee.Image:
    """Build the annual collection for an AOI and run LandTrendr in one call.

    Args:
        aoi: Area of interest as ee.Geometry.
        start_year: First composite year (inclusive).
        end_year: Last composite year (inclusive).
        segmentation_index: Index LandTrendr segments on (default 'NBR').
        ftv_indices: Additional indices carried as fit-to-vertices bands (default none).
        start_day: Growing-season window start as 'MM-DD' (default full year).
        end_day: Growing-season window end as 'MM-DD' (crosses the new year if before start_day).
        sensors: Landsat sensors to fuse (default all of L5/L7/L8/L9).
        run_params: Optional overrides for the 8 run parameters.

    Returns:
        ee.Image LandTrendr output array image.
    """
    collection = build_landtrendr_collection(
        aoi,
        start_year,
        end_year,
        segmentation_index=segmentation_index,
        ftv_indices=ftv_indices,
        start_day=start_day,
        end_day=end_day,
        sensors=sensors,
    )
    return run_landtrendr(collection, run_params)
