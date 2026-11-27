# landtrendr/ — conventions

`landtrendr/` is a **method** subpackage (not a sensor): it fuses the Landsat builders into a one-image-per-year, Roy-harmonized medoid series, runs `ee.Algorithms.TemporalSegmentation.LandTrendr`, and parses the array-image outputs. Its segmentation band must be **loss-positive** (NBR/NDVI/NDMI ×−1, tracked by `DIST_DIR`), and run-param defaults + Roy coefficients live in `constants.py`.

**Change maps:** `get_change_map` picks one segment per pixel by `sort`. By default the mag/dur/preval/year filters run AFTER that choice, as in LT-GEE, so they can only mask a pixel. Pass `filter_before_sort=True` to drop non-qualifying segments first. For example, `{sort: greatest, dur < 5, mag >= 0.1}` then gives the greatest abrupt loss even where a larger slow decline spans the record. That decline would otherwise win with `yod` = the second year of the run. Keep the default False where existing outputs or export hashes must not change.
