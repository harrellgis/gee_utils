"""Tests for the Otsu thresholding module.

otsu_threshold is checked against a plain-Python reference on a hand-built
bimodal histogram. The edge-based threshold runs on a synthetic two-level step
image (longitude split), so the expected split is known; the no-edge and
min_value paths are checked on constant images. The real-data run is slow.
"""

import pytest

# Synthetic step image: -0.5 west of STEP_LON, +0.5 east of it.
STEP_LON = 36.8
LOW, HIGH = -0.5, 0.5


def _reference_otsu(counts: list[float], means: list[float]) -> float:
    total = sum(counts)
    mean = sum(c * m for c, m in zip(counts, means)) / total
    best_bss, best_mean = None, None
    for i in range(1, len(means) + 1):
        a_count = sum(counts[:i])
        b_count = total - a_count
        if a_count == 0 or b_count == 0:
            continue
        a_mean = sum(c * m for c, m in zip(counts[:i], means[:i])) / a_count
        b_mean = (mean * total - a_count * a_mean) / b_count
        bss = a_count * (a_mean - mean) ** 2 + b_count * (b_mean - mean) ** 2
        if best_bss is None or bss > best_bss:
            best_bss, best_mean = bss, means[i - 1]
    assert best_mean is not None
    return best_mean


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def step_region(ee_session):
    """A ~2 km box centred on the step."""
    return ee_session.Geometry.Rectangle(
        [STEP_LON - 0.01, -3.41, STEP_LON + 0.01, -3.39]
    )


@pytest.fixture
def step_image(ee_session):
    """Single-band image that is LOW west of STEP_LON and HIGH east of it."""
    lon = ee_session.Image.pixelLonLat().select("longitude")
    return lon.gt(STEP_LON).multiply(HIGH - LOW).add(LOW).toFloat().rename("index")


# --------------------------------------------------------------------------- #
# Pure tests                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("buckets", [0, 1])
def test_edge_otsu_threshold_rejects_too_few_buckets(buckets):
    from eetools.thresholding import edge_otsu_threshold, otsu_threshold_image

    with pytest.raises(ValueError, match="buckets"):
        edge_otsu_threshold(None, None, 10, buckets=buckets)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="buckets"):
        otsu_threshold_image(None, None, 10, buckets=buckets)  # type: ignore[arg-type]


def test_otsu_constants():
    from eetools.constants import CANNY_SIGMA, CANNY_THRESHOLD, OTSU_HISTOGRAM_BUCKETS

    assert OTSU_HISTOGRAM_BUCKETS >= 2
    assert CANNY_THRESHOLD > 0
    assert CANNY_SIGMA > 0


# --------------------------------------------------------------------------- #
# Otsu                                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.ee
def test_otsu_threshold_matches_reference(ee_session):
    from eetools.thresholding import otsu_threshold

    # Bimodal: a large mode around -0.6 and a smaller one around +0.5.
    means = [round(-0.8 + 0.1 * i, 1) for i in range(17)]  # -0.8 ... 0.8
    counts = [5, 40, 90, 40, 10, 3, 1, 0, 1, 2, 4, 15, 35, 70, 30, 6, 1]
    histogram = ee_session.Dictionary({"histogram": counts, "bucketMeans": means})

    threshold = otsu_threshold(histogram).getInfo()
    assert threshold == pytest.approx(_reference_otsu(counts, means))
    assert -0.6 < threshold < 0.5


@pytest.mark.ee
def test_otsu_threshold_image_splits_step(ee_session, step_image, step_region):
    from eetools.thresholding import otsu_threshold_image

    # fallback=99 so the assertion can only pass on a real Otsu split.
    threshold = otsu_threshold_image(
        step_image, step_region, scale=30, fallback=99
    ).getInfo()
    assert LOW <= threshold < HIGH


# --------------------------------------------------------------------------- #
# Edge sampling                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.ee
def test_canny_edge_buffer_keeps_only_pixels_near_step(
    ee_session, step_image, step_region
):
    from eetools.thresholding import canny_edge_buffer

    # The +/-0.5 step is gentler than an NDWI land/water edge, so lower the Canny
    # gradient threshold.
    image_edge = canny_edge_buffer(step_image, scale=30, canny_threshold=0.1)
    assert image_edge.bandNames().getInfo() == ["index"]

    def value_at(lon: float):
        return (
            image_edge.reduceRegion(
                reducer=ee_session.Reducer.first(),
                geometry=ee_session.Geometry.Point([lon, -3.4]),
                scale=30,
            )
            .get("index")
            .getInfo()
        )

    # ~10 m east of the step is kept; ~550 m away (well beyond the buffer) is masked.
    assert value_at(STEP_LON + 0.0001) == pytest.approx(HIGH)
    assert value_at(STEP_LON + 0.005) is None


# --------------------------------------------------------------------------- #
# Edge-based threshold                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.ee
def test_edge_otsu_threshold_splits_step(ee_session, step_image, step_region):
    from eetools.thresholding import edge_otsu_threshold

    # fallback=99 so the assertion can only pass if edges were found.
    threshold = edge_otsu_threshold(
        step_image, step_region, scale=30, canny_threshold=0.1, fallback=99
    ).getInfo()
    assert LOW <= threshold < HIGH


@pytest.mark.ee
def test_edge_otsu_threshold_min_value(ee_session, step_image, step_region):
    from eetools.thresholding import edge_otsu_threshold

    threshold = edge_otsu_threshold(
        step_image, step_region, scale=30, canny_threshold=0.1, min_value=0.6
    ).getInfo()
    assert threshold == pytest.approx(0.6)


@pytest.mark.ee
def test_edge_otsu_threshold_fallback_without_edges(ee_session, step_region):
    from eetools.thresholding import edge_otsu_threshold

    flat = ee_session.Image.constant(0.2).toFloat().rename("index")
    threshold = edge_otsu_threshold(flat, step_region, scale=30, fallback=0.3).getInfo()
    assert threshold == pytest.approx(0.3)


@pytest.mark.ee
@pytest.mark.slow
def test_edge_otsu_threshold_real_ndwi(ee_session, small_aoi):
    from eetools.sensors.sentinel.preprocessing import get_s2_sr_collection
    from eetools.thresholding import edge_otsu_threshold

    col = get_s2_sr_collection(
        small_aoi,
        ee_session.Date("2024-01-01"),
        ee_session.Date("2024-04-01"),
        apply_water_masking=False,
        indices=["NDWI"],
    )
    if col.size().getInfo() == 0:
        pytest.skip("no Sentinel-2 scenes for the AOI/window")
    ndwi = col.select("NDWI").median()
    threshold = edge_otsu_threshold(ndwi, small_aoi, scale=30, fallback=0.3).getInfo()
    assert -1 <= threshold <= 1
