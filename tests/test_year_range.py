"""Pure tests for utils.validate_collection_year_range (EE lookups mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from eetools import utils


def _window(first_year, last_year):
    first, last = MagicMock(), MagicMock()
    first.get.return_value.getInfo.return_value = first_year
    last.get.return_value.getInfo.return_value = last_year
    return first, last


@pytest.mark.parametrize("years", [(2001, 2024), (2001, 2001), (2024, 2024)])
def test_year_range_inside_record_passes(years):
    with patch.object(utils, "get_available_window", return_value=_window(2001, 2024)):
        utils.validate_collection_year_range("C", "AOI", *years, "X")


@pytest.mark.parametrize("years", [(2000, 2010), (2010, 2025)])
def test_year_range_outside_record_raises(years):
    with patch.object(utils, "get_available_window", return_value=_window(2001, 2024)):
        with pytest.raises(ValueError, match="outside the X record.*2001-2024"):
            utils.validate_collection_year_range("C", "AOI", *years, "X")


def test_year_range_reversed_raises_before_lookup():
    with patch.object(utils, "get_available_window") as window:
        with pytest.raises(ValueError, match="end_year"):
            utils.validate_collection_year_range("C", "AOI", 2021, 2020)
    window.assert_not_called()
