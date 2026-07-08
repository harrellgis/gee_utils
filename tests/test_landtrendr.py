"""Tests for the LandTrendr subpackage.

resolve_run_params is pure Python. The medoid/harmonization/orientation building blocks
are checked deterministically on synthetic constant images. The end-to-end builder, run,
and output parsers hit the real Landsat archive and are marked slow.
"""

import pytest


# --------------------------------------------------------------------------- #
# Pure (no Earth Engine)
# --------------------------------------------------------------------------- #
def test_resolve_run_params_merges_over_defaults():
    from eetools.constants import LANDTRENDR_DEFAULT_RUN_PARAMS
    from eetools.landtrendr.segmentation import resolve_run_params

    out = resolve_run_params({"maxSegments": 8, "pvalThreshold": 0.1})
    assert out["maxSegments"] == 8  # overridden
    assert out["pvalThreshold"] == 0.1  # overridden
    assert out["minObservationsNeeded"] == 6  # default retained
    # The defaults dict itself is not mutated.
    assert LANDTRENDR_DEFAULT_RUN_PARAMS["maxSegments"] == 6


def test_resolve_run_params_none_returns_defaults():
    from eetools.constants import LANDTRENDR_DEFAULT_RUN_PARAMS
    from eetools.landtrendr.segmentation import resolve_run_params

    assert resolve_run_params() == LANDTRENDR_DEFAULT_RUN_PARAMS


def test_build_landtrendr_collection_rejects_reversed_years():
    from eetools.landtrendr.collection import build_landtrendr_collection

    with pytest.raises(ValueError, match="must not exceed"):
        build_landtrendr_collection(aoi=None, start_year=2017, end_year=2008)


def test_build_landtrendr_collection_rejects_ftv_equal_to_seg_index():
    from eetools.landtrendr.collection import build_landtrendr_collection

    with pytest.raises(ValueError, match="must not repeat"):
        build_landtrendr_collection(
            aoi=None,
            start_year=2008,
            end_year=2017,
            segmentation_index="NBR",
            ftv_indices=["NBR"],
        )


def test_natural_index_rejects_unknown_index():
    from eetools.landtrendr.collection import _natural_index

    # Validation happens (pure) before any EE op, so a None composite never gets touched.
    with pytest.raises(ValueError, match="Unknown index"):
        _natural_index(None, "NOPE")


def test_segmentation_band_rejects_red_edge_index():
    from eetools.landtrendr.collection import _segmentation_band

    # CIred_edge/NDRE need 'red_edge', which the Landsat common bands cannot provide.
    with pytest.raises(ValueError, match="requires band_map key"):
        _segmentation_band(None, "CIred_edge")


# --------------------------------------------------------------------------- #
# Building blocks (synthetic constant images)
# --------------------------------------------------------------------------- #
@pytest.mark.ee
def test_medoid_composite_picks_median_closest(ee_session, first_value):
    from eetools.landtrendr.collection import medoid_composite

    # toFloat() so all three constant images share one band type (real imagery is
    # uniformly typed; mixed ee.Image.constant widths would break qualityMosaic).
    col = ee_session.ImageCollection(
        [ee_session.Image.constant(v).rename("x").toFloat() for v in (1, 5, 100)]
    )
    out = medoid_composite(col, ["x"])
    # median is 5; the observation closest to it (value 5) wins.
    assert first_value(out, "x") == pytest.approx(5)


@pytest.mark.ee
def test_medoid_composite_empty_collection_returns_masked_bands(ee_session):
    from eetools.constants import LANDTRENDR_COMMON_BANDS
    from eetools.landtrendr.collection import medoid_composite

    # A year with zero scenes after filtering: the guard returns a fully-masked image with
    # the requested band names (not a 0-band image that would crash the next .select).
    empty = ee_session.ImageCollection([])
    out = medoid_composite(empty, LANDTRENDR_COMMON_BANDS)
    assert out.bandNames().getInfo() == LANDTRENDR_COMMON_BANDS
    # Downstream index math over the common bands still builds without error...
    from eetools.landtrendr.collection import _segmentation_band

    seg = _segmentation_band(out, "NBR")
    assert seg.bandNames().getInfo() == ["NBR"]
    # ...and every pixel is masked, so the year contributes nothing to the series.
    masked_count = (
        out.select("BLUE")
        .mask()
        .reduceRegion(
            reducer=ee_session.Reducer.sum(),
            geometry=ee_session.Geometry.Point([0, 0]).buffer(30),
            scale=30,
        )
        .get("BLUE")
        .getInfo()
    )
    assert masked_count == 0


@pytest.mark.ee
def test_medoid_composite_tolerates_mismatched_band_types(ee_session, first_value):
    from eetools.constants import LANDTRENDR_COMMON_BANDS
    from eetools.landtrendr.collection import medoid_composite

    # Reproduce the multi-sensor case: the same band carries DIFFERENT bounded float types
    # across scenes (a TM/ETM+ scaled scene vs a Roy-harmonized OLI scene), which makes the
    # median()/qualityMosaic reductions reject the collection as non-homogeneous unless
    # medoid_composite casts to a uniform type first.
    raw = (
        ee_session.Image.constant([1000, 2000, 3000, 4000, 5000, 6000])
        .toInt16()
        .rename(LANDTRENDR_COMMON_BANDS)
    )
    tm = raw.multiply(0.0000275).add(-0.2)  # scaled reflectance (one bounded range)
    oli = tm.subtract(-0.0095).divide(0.9785)  # Roy-like transform (a different range)
    col = ee_session.ImageCollection([tm, oli])

    out = medoid_composite(col, LANDTRENDR_COMMON_BANDS)
    assert out.bandNames().getInfo() == LANDTRENDR_COMMON_BANDS
    # Evaluating forces the reduction that used to raise the homogeneity error.
    blue = first_value(out, "BLUE")
    assert blue is not None


@pytest.mark.ee
def test_harmonize_oli_to_etm_matches_roy_formula(ee_session, first_value):
    from eetools.constants import (
        LANDTRENDR_COMMON_BANDS,
        ROY_OLI_TO_ETM_INTERCEPTS,
        ROY_OLI_TO_ETM_SLOPES,
    )
    from eetools.landtrendr.collection import _harmonize_oli_to_etm

    vals = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    img = ee_session.Image.constant(vals).rename(LANDTRENDR_COMMON_BANDS)
    out = _harmonize_oli_to_etm(img)
    # etm = (oli - intercept) / slope, per band.
    expected_red = (0.30 - ROY_OLI_TO_ETM_INTERCEPTS[2]) / ROY_OLI_TO_ETM_SLOPES[2]
    assert first_value(out, "RED") == pytest.approx(expected_red, rel=1e-6)


@pytest.mark.ee
def test_segmentation_band_is_loss_positive(ee_session, first_value):
    from eetools.landtrendr.collection import _segmentation_band

    # NBR = (NIR - SWIR2)/(NIR + SWIR2) = (0.4 - 0.1)/(0.4 + 0.1) = 0.6
    comp = ee_session.Image.constant([0.4, 0.1]).rename(["NIR", "SWIR2"])
    seg = _segmentation_band(comp, "NBR")
    assert seg.bandNames().getInfo() == ["NBR"]
    # Oriented loss-positive: the natural NBR is negated.
    assert first_value(seg, "NBR") == pytest.approx(-0.6)


@pytest.mark.ee
def test_segmentation_band_msavi2_is_negated(ee_session, first_value):
    import math

    from eetools.landtrendr.collection import _segmentation_band

    # A generalized INDEX_REGISTRY index (dist_dir defaults to -1: greenness falls with loss).
    comp = ee_session.Image.constant([0.5, 0.3]).rename(["NIR", "RED"])
    seg = _segmentation_band(comp, "MSAVI2")
    assert seg.bandNames().getInfo() == ["MSAVI2"]
    natural = (2 * 0.5 + 1 - math.sqrt((2 * 0.5 + 1) ** 2 - 8 * (0.5 - 0.3))) / 2
    assert first_value(seg, "MSAVI2") == pytest.approx(-natural)


@pytest.mark.ee
def test_segmentation_band_bsi_used_as_is(ee_session, first_value):
    from eetools.landtrendr.collection import _segmentation_band

    # BSI rises with degradation (LANDTRENDR_DIST_DIR = +1), so it is NOT negated.
    comp = ee_session.Image.constant([0.1, 0.3, 0.5, 0.25]).rename(
        ["BLUE", "RED", "NIR", "SWIR1"]
    )
    seg = _segmentation_band(comp, "BSI")
    assert seg.bandNames().getInfo() == ["BSI"]
    num = (0.25 + 0.30) - (0.50 + 0.10)
    den = (0.25 + 0.30) + (0.50 + 0.10)
    assert first_value(seg, "BSI") == pytest.approx(num / den)


# --------------------------------------------------------------------------- #
# End-to-end against the real Landsat archive
# --------------------------------------------------------------------------- #
@pytest.mark.ee
@pytest.mark.slow
def test_build_landtrendr_collection_real(ee_session, small_aoi):
    from eetools.landtrendr.collection import build_landtrendr_collection

    col = build_landtrendr_collection(small_aoi, 2008, 2017, segmentation_index="NBR")
    assert isinstance(col, ee_session.ImageCollection)
    assert col.size().getInfo() == 10  # one image per year, inclusive
    assert col.first().bandNames().getInfo() == ["NBR"]


@pytest.mark.ee
@pytest.mark.slow
def test_run_landtrendr_outputs_segmentation_and_rmse(ee_session, small_aoi):
    from eetools.landtrendr.segmentation import run_landtrendr_from_aoi

    lt = run_landtrendr_from_aoi(small_aoi, 2008, 2017, segmentation_index="NBR")
    names = lt.bandNames().getInfo()
    assert "LandTrendr" in names
    assert "rmse" in names


@pytest.mark.ee
@pytest.mark.slow
def test_get_change_map_and_fitted_stack(ee_session, small_aoi):
    from eetools.landtrendr.outputs import (
        get_change_map,
        get_fitted_stack,
        get_segment_data,
    )
    from eetools.landtrendr.segmentation import run_landtrendr_from_aoi

    lt = run_landtrendr_from_aoi(small_aoi, 2008, 2017, segmentation_index="NBR")

    change = get_change_map(lt)
    change_names = change.bandNames().getInfo()
    for band in ("yod", "mag", "dur"):
        assert band in change_names

    stack = get_fitted_stack(lt, 2008, 2017)
    assert len(stack.bandNames().getInfo()) == 10  # one band per year

    seg = get_segment_data(lt)
    assert isinstance(seg, ee_session.Image)
    assert len(seg.bandNames().getInfo()) == 1  # single array band
