"""Tests for the Google Satellite Embedding V1 sensor module.

The band-list constant is pure Python; ``embedding_similarity`` builds a server-side
graph and is checked on synthetic constant images; the real-collection builder is slow.
"""

import pytest


# --------------------------------------------------------------------------- #
# Pure (no Earth Engine)
# --------------------------------------------------------------------------- #
def test_satellite_embedding_bands_are_a00_to_a63():
    from eetools.constants import SATELLITE_EMBEDDING_BANDS

    assert len(SATELLITE_EMBEDDING_BANDS) == 64
    assert SATELLITE_EMBEDDING_BANDS[0] == "A00"
    assert SATELLITE_EMBEDDING_BANDS[9] == "A09"
    assert SATELLITE_EMBEDDING_BANDS[-1] == "A63"


# --------------------------------------------------------------------------- #
# Earth Engine behaviour (synthetic constant images)
# --------------------------------------------------------------------------- #
@pytest.mark.ee
def test_embedding_similarity_identical_vectors(ee_session, first_value):
    from eetools.sensors.satellite_embedding.preprocessing import embedding_similarity

    # Two identical unit vectors -> dot product 0.6^2 + 0.8^2 = 1.0.
    a = ee_session.Image.constant([0.6, 0.8]).toFloat().rename(["A00", "A01"])
    sim = embedding_similarity(a, a)
    assert first_value(sim, "similarity") == pytest.approx(1.0)


@pytest.mark.ee
def test_embedding_similarity_orthogonal_vectors(ee_session, first_value):
    from eetools.sensors.satellite_embedding.preprocessing import embedding_similarity

    # Orthogonal vectors -> dot product 0.
    a = ee_session.Image.constant([1.0, 0.0]).toFloat().rename(["A00", "A01"])
    b = ee_session.Image.constant([0.0, 1.0]).toFloat().rename(["A00", "A01"])
    sim = embedding_similarity(a, b)
    assert first_value(sim, "similarity") == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Real dataset (network)
# --------------------------------------------------------------------------- #
@pytest.mark.ee
@pytest.mark.slow
def test_get_satellite_embedding_collection_real(ee_session, small_aoi):
    from eetools.sensors.satellite_embedding.preprocessing import (
        get_satellite_embedding_collection,
    )

    col = get_satellite_embedding_collection(
        small_aoi,
        ee_session.Date("2023-01-01"),
        ee_session.Date("2024-01-01"),
    )
    assert isinstance(col, ee_session.ImageCollection)
    if col.size().getInfo() == 0:
        pytest.skip("no Satellite Embedding scenes for the AOI/window")
    bands = col.first().bandNames().getInfo()
    assert "A00" in bands and "A63" in bands
    assert len(bands) == 64
