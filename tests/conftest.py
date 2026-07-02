"""Shared pytest fixtures for the eetools test suite.

Two tiers of tests live in this suite:

* **Pure tests** — exercise the pandas / matplotlib / plain-Python code paths
  (config state, tables, plots, constants, validation helpers). They need no
  Earth Engine session and always run.
* **EE tests** — marked ``@pytest.mark.ee`` and depend on the ``ee_session``
  fixture. They are skipped unless an Earth Engine session can be initialized.
  Set the ``EE_PROJECT`` environment variable (or ``GOOGLE_CLOUD_PROJECT``) to
  a project you have authenticated against to run them locally.

Most EE tests operate on *synthetic constant images* rather than real
collections, so they are deterministic and fast — the index/QA math is checked
against hand-computed values reduced over a tiny point geometry. A handful of
``@pytest.mark.slow`` tests hit real datasets to confirm the collection
builders wire together end to end.
"""

import os

import matplotlib
import pandas as pd
import pytest

# Force a non-interactive backend before anything imports pyplot so the plot
# tests never try to open a window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# Pure-Python fixtures (no Earth Engine required)
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _close_figures():
    """Close any matplotlib figures a test created to avoid state leaking."""
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def reset_eetools_config():
    """Snapshot and restore the module-level eetools config between tests."""
    from eetools import _config

    saved = dict(_config._state)
    try:
        yield _config
    finally:
        _config._state.clear()
        _config._state.update(saved)


@pytest.fixture
def monthly_timeseries_df():
    """A small long-format monthly timeseries with a single metric column."""
    return pd.DataFrame(
        {
            "date": [
                "2023-03-01",
                "2023-01-01",
                "2023-02-01",
                "2023-04-01",
            ],
            "metric": [0.41, 0.30, 0.35, 0.52],
            "temporal_scale": ["monthly"] * 4,
            "site_name": ["site_a"] * 4,
            "year": [2023, 2023, 2023, 2023],
            "month": [3, 1, 2, 4],
        }
    )


@pytest.fixture
def multi_metric_df():
    """A timeseries carrying several metric columns for multi-line plots."""
    return pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-02-01", "2023-03-01"],
            "NDVI": [0.40, 0.45, 0.50],
            "EVI": [0.30, 0.33, 0.36],
            "SAVI": [0.20, 0.22, 0.25],
        }
    )


@pytest.fixture
def stats_feature_collection():
    """A fake ee.FeatureCollection exposing only the ``getInfo`` contract used
    by ``stats_fc_to_df`` — one ``properties`` dict per feature."""

    class _FakeFC:
        def __init__(self, rows):
            self._rows = rows

        def getInfo(self):
            return {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": dict(row)} for row in self._rows
                ],
            }

    return _FakeFC(
        [
            {"site_name": "site_a", "year": 2023, "NDVI_mean": 0.41},
            {"site_name": "site_b", "year": 2023, "NDVI_mean": 0.55},
        ]
    )


# --------------------------------------------------------------------------- #
# Earth Engine fixtures (skipped unless a session can be initialized)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def ee_session():
    """Initialize Earth Engine once per session, or skip every EE test.

    Resolves the project from ``EE_PROJECT`` / ``GOOGLE_CLOUD_PROJECT`` and
    skips (rather than errors) when no authenticated session is available, so
    the pure tests still run in CI without credentials.
    """
    ee = pytest.importorskip("ee")
    project = os.environ.get("EE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as exc:  # noqa: BLE001 - any init failure means "skip"
        pytest.skip(f"Earth Engine session unavailable: {exc}")
    return ee


@pytest.fixture
def point(ee_session):
    """A 1 m point-buffer geometry to reduce constant images over cheaply."""
    return ee_session.Geometry.Point([36.8, -3.4]).buffer(1)


@pytest.fixture
def first_value(ee_session, point):
    """Return a helper that pulls a single band value off an image.

    Because the EE tests build ``ee.Image.constant`` images, ``Reducer.first``
    over the tiny ``point`` geometry yields the exact constant — letting tests
    assert hand-computed index/mask values.
    """

    def _first_value(image, band, scale=1):
        return (
            ee_session.Image(image)
            .select(band)
            .reduceRegion(
                reducer=ee_session.Reducer.first(),
                geometry=point,
                scale=scale,
            )
            .get(band)
            .getInfo()
        )

    return _first_value


@pytest.fixture
def reflectance_band_map():
    """Logical -> band-name map matching the synthetic reflectance image."""
    return {
        "blue": "blue",
        "green": "green",
        "red": "red",
        "red_edge": "red_edge",
        "nir": "nir",
        "swir1": "swir1",
        "swir2": "swir2",
    }


@pytest.fixture
def synthetic_reflectance_image(ee_session, reflectance_band_map):
    """A constant multi-band reflectance image with known per-band values.

    Values are chosen so the derived indices are easy to verify by hand, e.g.
    NDVI = (nir - red) / (nir + red) = (0.5 - 0.3) / (0.5 + 0.3) = 0.25.
    """
    bands = ["blue", "green", "red", "red_edge", "nir", "swir1", "swir2"]
    values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.25, 0.15]
    return ee_session.Image.constant(values).rename(bands)


@pytest.fixture
def small_aoi(ee_session):
    """A small land AOI (coastal Kwale, Kenya) for real-dataset integration."""
    return ee_session.Geometry.Rectangle([39.20, -4.30, 39.25, -4.25])


@pytest.fixture
def timed_collection(ee_session):
    """A 2-image constant collection with one image in 2020 and one in 2021.

    The constants are cast to a common float type — otherwise each
    ``ee.Image.constant`` carries a distinct typed value-range and EE rejects
    reductions across the collection as "inhomogeneous".
    """
    return ee_session.ImageCollection(
        [
            ee_session.Image.constant(0.2)
            .toFloat()
            .rename("b")
            .set("system:time_start", ee_session.Date("2020-06-01").millis()),
            ee_session.Image.constant(0.6)
            .toFloat()
            .rename("b")
            .set("system:time_start", ee_session.Date("2021-06-01").millis()),
        ]
    )
