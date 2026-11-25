"""validate_collection_date_range without an EE session (mocked dates)."""

from __future__ import annotations

import pytest


def _fake_date(ms: int, day: str):
    from unittest.mock import MagicMock

    d = MagicMock()
    d.millis.return_value.getInfo.return_value = ms
    d.format.return_value.getInfo.return_value = day
    return d


def test_validate_collection_date_range_start_on_first_image_day():
    from unittest.mock import patch

    from eetools import utils

    day = 86_400_000
    first = _fake_date(10 * day + 8 * 3_600_000, "1970-01-11")  # 08:00 UTC
    last = _fake_date(40 * day, "1970-02-10")
    with patch.object(utils, "get_available_window", return_value=(first, last)):
        # same calendar day as the first image, earlier in the day: accepted
        utils.validate_collection_date_range(
            "C",
            "AOI",
            _fake_date(10 * day, "1970-01-11"),
            _fake_date(20 * day, "1970-01-21"),
        )
        with pytest.raises(ValueError, match="earlier than the first"):
            utils.validate_collection_date_range(
                "C",
                "AOI",
                _fake_date(9 * day, "1970-01-10"),
                _fake_date(20 * day, "1970-01-21"),
            )
