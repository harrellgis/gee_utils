"""Sanity tests for the band lists, band maps, and scale factors in constants.py.

These guard against the kind of typo that would otherwise only surface as an
opaque Earth Engine error deep inside a collection build.
"""

import pytest

from eetools import constants as C

# (band_map, available_source_bands) pairs. The mapped band name must exist in
# the corresponding source-band list for the index functions to be able to
# .select() it.
BAND_MAP_CASES = [
    pytest.param(C.S2_BAND_MAP, C.S2_BANDS, id="sentinel2"),
    pytest.param(C.L8_BAND_MAP, C.L8_BANDS, id="landsat8"),
    pytest.param(C.HLS_BAND_MAP, C.HLS_COMMON_BANDS, id="hls"),
]


@pytest.mark.parametrize("band_map, source_bands", BAND_MAP_CASES)
def test_band_map_targets_exist_in_source_bands(band_map, source_bands):
    for logical, actual in band_map.items():
        assert actual in source_bands, (
            f"{logical!r} -> {actual!r} is not a known source band"
        )


@pytest.mark.parametrize("band_map", [C.S2_BAND_MAP, C.L8_BAND_MAP, C.HLS_BAND_MAP])
def test_band_maps_expose_core_logical_keys(band_map):
    # calc_indices() reads these keys; every sensor must define them.
    for key in ("blue", "green", "red", "nir", "swir1"):
        assert key in band_map


def test_sentinel_band_map_has_swir2_and_red_edge():
    # S2 is the only sensor calc_indices is called on with include_ndre=True.
    assert "swir2" in C.S2_BAND_MAP
    assert "red_edge" in C.S2_BAND_MAP


@pytest.mark.parametrize(
    "scale_factor",
    [
        C.S2_SCALE_FACTOR,
        C.L8_SCALE_FACTOR,
        C.HLS_SCALE_FACTOR,
        C.FPAR_SCALE_FACTOR,
        C.LAI_SCALE_FACTOR,
    ],
)
def test_scale_factors_are_positive(scale_factor):
    assert scale_factor > 0


@pytest.mark.parametrize(
    "collection_id",
    [
        C.CHIRPS_COLLECTION,
        C.S2_SR_COLLECTION,
        C.S2_CLOUD_PROB_COLLECTION,
        C.L8_SR_COLLECTION,
        C.HLS_L30_COLLECTION,
        C.HLS_S30_COLLECTION,
        C.MODIS_LAI_FPAR_COLLECTION,
        C.ESA_WC_COLLECTION,
    ],
)
def test_collection_ids_are_non_empty_strings(collection_id):
    assert isinstance(collection_id, str) and collection_id.strip()


def test_index_bands_are_subset_of_all_bands():
    assert set(C.S2_INDEX_BANDS).issubset(set(C.S2_ALL_BANDS))
    assert set(C.HLS_INDEX_BANDS).issubset(set(C.HLS_ALL_BANDS))


def test_cloud_filter_thresholds_are_percentages():
    thresholds = (
        C.CLOUD_FILTER,
        C.L8_CLOUD_FILTER,
        C.HLS_CLOUD_FILTER,
        C.CLD_PRB_THRESH,
    )
    for pct in thresholds:
        assert 0 <= pct <= 100


def test_hls_source_band_counts_match_common_bands():
    # Each HLS sensor renames its source bands onto the 6 common bands.
    assert len(C.HLS_L30_SOURCE_BANDS) == len(C.HLS_COMMON_BANDS)
    assert len(C.HLS_S30_SOURCE_BANDS) == len(C.HLS_COMMON_BANDS)
