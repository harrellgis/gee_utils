import pandas as pd


def stats_fc_to_df(stats_fc) -> pd.DataFrame:
    """Convert an ee.FeatureCollection of area statistics to a pandas DataFrame.

    Args:
        stats_fc: ee.FeatureCollection whose features each carry numeric stat properties.

    Returns:
        pd.DataFrame with one row per feature and one column per property.
    """
    fc_info = stats_fc.getInfo()
    rows = [feature["properties"] for feature in fc_info["features"]]
    return pd.DataFrame(rows)


def combine_site_season_tables(
    dfs: list[pd.DataFrame] | tuple[pd.DataFrame, ...],
    expected_temporal_scale: str | None = "monthly",
) -> pd.DataFrame:
    """Concatenate per-site or per-season DataFrames, optionally filtering to one
    temporal scale.

    Args:
        dfs: List or tuple of pd.DataFrames to concatenate; must contain at least one DataFrame.
        expected_temporal_scale: If provided, retains only rows where the 'temporal_scale' column equals this value; pass None to skip filtering (default 'monthly').

    Returns:
        pd.DataFrame with all rows concatenated, dates parsed, filtered by temporal_scale if specified, and sorted by site_name, season, year, month, and date where present.
    """
    if not dfs:
        raise ValueError("dfs must contain at least one DataFrame.")

    combined = pd.concat([df.copy() for df in dfs], ignore_index=True)

    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"])

    if expected_temporal_scale is not None and "temporal_scale" in combined.columns:
        combined = combined[
            combined["temporal_scale"].eq(expected_temporal_scale)
        ].copy()

    sort_cols = [
        col
        for col in ["site_name", "season", "year", "month", "date"]
        if col in combined.columns
    ]
    if sort_cols:
        combined = combined.sort_values(sort_cols)

    return combined.reset_index(drop=True)
