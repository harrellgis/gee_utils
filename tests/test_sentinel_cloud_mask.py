"""Pure tests for the Sentinel-2 cloud-mask method option (no EE session)."""

from unittest.mock import patch

import pytest

from eetools.constants import (
    CS_PLUS_BAND,
    CS_PLUS_BANDS,
    CS_PLUS_THRESH,
    S2_CLOUD_MASK_METHODS,
    S2_CLOUD_PROB_COLLECTION,
    S2_CLOUD_SCORE_PLUS_COLLECTION,
    S2_DEFAULT_CLOUD_MASK,
)
from eetools.sensors.sentinel import masking, preprocessing


def test_default_is_cloud_score_plus():
    assert S2_DEFAULT_CLOUD_MASK == "cloud_score_plus"
    assert S2_DEFAULT_CLOUD_MASK in S2_CLOUD_MASK_METHODS
    assert "s2cloudless" in S2_CLOUD_MASK_METHODS
    assert CS_PLUS_BAND in CS_PLUS_BANDS
    assert 0 <= CS_PLUS_THRESH <= 1


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"cloud_mask": "qa60"}, "Unknown cloud_mask"),
        ({"cloud_mask": "cloud_score_plus", "cs_band": "probability"}, "band"),
        ({"cloud_mask": "cloud_score_plus", "cs_threshold": 60}, r"\[0, 1\]"),
        ({"cloud_mask": "cloud_score_plus", "cs_threshold": -0.1}, r"\[0, 1\]"),
    ],
)
def test_validate_rejects_bad_args(kwargs, match):
    with pytest.raises(ValueError, match=match):
        masking.validate_s2_cloud_mask_args(**kwargs)


def test_validate_ignores_cs_args_for_s2cloudless():
    masking.validate_s2_cloud_mask_args("s2cloudless", "bogus", 99)


@pytest.mark.parametrize("cs_band", CS_PLUS_BANDS)
def test_validate_accepts_cs_bands(cs_band):
    masking.validate_s2_cloud_mask_args("cloud_score_plus", cs_band, 0.5)


@pytest.mark.parametrize(
    "fn, collection_id, prop",
    [
        (masking.get_s2_sr_cs_col, S2_CLOUD_SCORE_PLUS_COLLECTION, "cloud_score_plus"),
        (masking.get_s2_sr_cld_col, S2_CLOUD_PROB_COLLECTION, "s2cloudless"),
    ],
)
def test_join_uses_matching_collection_and_property(fn, collection_id, prop):
    with patch.object(masking, "ee") as mock_ee:
        fn("AOI", "S", "E")
    mock_ee.ImageCollection.assert_any_call(collection_id)
    mock_ee.Join.saveFirst.assert_called_once_with(prop)
    mock_ee.Filter.equals.assert_called_once_with(
        leftField="system:index", rightField="system:index"
    )


def test_build_bad_method_raises_before_graph():
    with (
        patch.object(masking, "get_s2_sr_cs_col") as cs,
        patch.object(masking, "get_s2_sr_cld_col") as cld,
        pytest.raises(ValueError),
    ):
        masking.build_cloudfree_s2sr_col("AOI", "S", "E", cloud_mask="nope")
    cs.assert_not_called()
    cld.assert_not_called()


def test_get_s2_sr_collection_validates_before_date_check():
    with (
        patch.object(preprocessing, "validate_s2_sr_date_range") as dates,
        pytest.raises(ValueError, match="cs_threshold"),
    ):
        preprocessing.get_s2_sr_collection("AOI", "S", "E", cs_threshold=1.5)
    dates.assert_not_called()


def test_get_s2_sr_collection_passes_s2cloudless():
    with (
        patch.object(preprocessing, "validate_s2_sr_date_range"),
        patch.object(preprocessing, "build_cloudfree_s2sr_col") as build,
    ):
        preprocessing.get_s2_sr_collection(
            "AOI", "S", "E", apply_water_masking=False, cloud_mask="s2cloudless"
        )
    assert build.call_args.kwargs["cloud_mask"] == "s2cloudless"
