# landtrendr/ — conventions

`landtrendr/` is a **method** subpackage (not a sensor): it fuses the Landsat builders into a one-image-per-year, Roy-harmonized medoid series, runs `ee.Algorithms.TemporalSegmentation.LandTrendr`, and parses the array-image outputs. Its segmentation band must be **loss-positive** (NBR/NDVI/NDMI ×−1, tracked by `DIST_DIR`), and run-param defaults + Roy coefficients live in `constants.py`.
