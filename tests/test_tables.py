"""Tests for eetools.visualization.tables (pure pandas, no Earth Engine)."""

import pandas as pd
import pytest

from eetools.visualization.tables import (
    combine_site_season_tables,
    stats_fc_to_df,
)


def test_stats_fc_to_df_flattens_properties(stats_feature_collection):
    df = stats_fc_to_df(stats_feature_collection)

    assert list(df.columns) == ["site_name", "year", "NDVI_mean"]
    assert len(df) == 2
    assert df.loc[df["site_name"] == "site_b", "NDVI_mean"].iloc[0] == 0.55


def test_combine_site_season_tables_concatenates_and_sorts():
    df1 = pd.DataFrame(
        {
            "site_name": ["b", "b"],
            "year": [2023, 2023],
            "month": [2, 1],
            "date": ["2023-02-01", "2023-01-01"],
            "temporal_scale": ["monthly", "monthly"],
        }
    )
    df2 = pd.DataFrame(
        {
            "site_name": ["a"],
            "year": [2023],
            "month": [1],
            "date": ["2023-01-01"],
            "temporal_scale": ["monthly"],
        }
    )

    combined = combine_site_season_tables([df1, df2])

    assert len(combined) == 3
    # Sorted by site_name then month: a/Jan, b/Jan, b/Feb.
    assert combined["site_name"].tolist() == ["a", "b", "b"]
    assert combined["month"].tolist() == [1, 1, 2]
    # Index reset after sorting.
    assert combined.index.tolist() == [0, 1, 2]
    # The date column is parsed to datetime.
    assert pd.api.types.is_datetime64_any_dtype(combined["date"])


def test_combine_site_season_tables_filters_by_temporal_scale():
    df = pd.DataFrame(
        {
            "site_name": ["a", "a"],
            "temporal_scale": ["monthly", "annual"],
            "date": ["2023-01-01", "2023-01-01"],
        }
    )

    combined = combine_site_season_tables([df], expected_temporal_scale="monthly")

    assert combined["temporal_scale"].unique().tolist() == ["monthly"]
    assert len(combined) == 1


def test_combine_site_season_tables_skip_filter_with_none():
    df = pd.DataFrame(
        {
            "site_name": ["a", "a"],
            "temporal_scale": ["monthly", "annual"],
            "date": ["2023-01-01", "2023-01-01"],
        }
    )

    combined = combine_site_season_tables([df], expected_temporal_scale=None)

    assert len(combined) == 2


def test_combine_site_season_tables_does_not_mutate_inputs():
    df = pd.DataFrame({"site_name": ["a"], "date": ["2023-01-01"]})
    original_dtype = df["date"].dtype

    combine_site_season_tables([df], expected_temporal_scale=None)

    # The function copies before parsing dates, so the caller's frame is intact.
    assert df["date"].dtype == original_dtype


def test_combine_site_season_tables_empty_raises():
    with pytest.raises(ValueError, match="at least one DataFrame"):
        combine_site_season_tables([])
