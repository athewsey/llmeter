# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from datetime import datetime, UTC
import pytest

from llmeter.utils import RunningStats


@pytest.fixture
def rs() -> RunningStats:
    return RunningStats(
        metrics=[
            "time_to_first_token",
            "time_to_last_token",
            "time_per_output_token",
            "num_tokens_input",
            "num_tokens_output",
        ]
    )


@pytest.fixture
def populated_rs(rs) -> RunningStats:
    """A RunningStats with 3 responses recorded."""
    responses = [
        {
            "time_to_first_token": 0.3,
            "time_to_last_token": 0.8,
            "time_per_output_token": 0.02,
            "num_tokens_input": 100,
            "num_tokens_output": 25,
            "error": None,
            "request_time": datetime(
                year=2026, month=4, day=16, hour=1, minute=0, second=0, tzinfo=UTC
            ),
        },
        {
            "time_to_first_token": 0.5,
            "time_to_last_token": 1.2,
            "time_per_output_token": 0.03,
            "num_tokens_input": 120,
            "num_tokens_output": 30,
            "error": None,
            "request_time": datetime(
                year=2026, month=4, day=16, hour=1, minute=0, second=30, tzinfo=UTC
            ),
        },
        {
            "time_to_first_token": 0.4,
            "time_to_last_token": 1.0,
            "time_per_output_token": 0.025,
            "num_tokens_input": 110,
            "num_tokens_output": 28,
            "error": "timeout",
            "request_time": datetime(
                year=2026, month=4, day=16, hour=1, minute=1, second=0, tzinfo=UTC
            ),
        },
    ]
    for r in responses:
        rs.update(r)
    return rs


# ── update ───────────────────────────────────────────────────────────────────


class TestUpdate:
    def test_first_update_sets_times(self, rs):
        assert rs._first_send_time is None
        assert rs._last_send_time is None
        request_time = datetime(
            year=2026, month=4, day=16, hour=12, minute=34, second=56
        )
        rs.update(
            {
                "error": None,
                "request_time": request_time,
            }
        )
        assert rs._first_send_time == request_time
        assert rs._last_send_time == request_time

    def test_subsequent_sends_update_last_time(self, rs):
        assert rs._first_send_time is None
        assert rs._last_send_time is None
        time_1 = datetime(
            year=2026, month=4, day=16, hour=12, minute=0, second=30, tzinfo=UTC
        )
        time_2 = datetime(
            year=2026, month=4, day=16, hour=12, minute=0, second=31, tzinfo=UTC
        )
        time_3 = datetime(
            year=2026, month=4, day=16, hour=12, minute=0, second=29, tzinfo=UTC
        )
        rs.update({"error": None, "request_time": time_1})
        rs.update({"error": None, "request_time": time_2})
        rs.update({"error": None, "request_time": time_3})
        assert rs._first_send_time == time_3
        assert rs._last_send_time == time_2

    def test_count_increments(self, rs):
        rs.update({"time_to_first_token": 0.3, "error": None})
        assert rs._count == 1
        rs.update({"time_to_first_token": 0.5, "error": None})
        assert rs._count == 2

    def test_failed_count(self, rs):
        rs.update({"error": "timeout"})
        rs.update({"error": None})
        rs.update({"error": "connection refused"})
        assert rs._failed == 2

    def test_none_values_skipped(self, rs):
        rs.update({"time_to_first_token": None, "error": None})
        assert len(rs._values["time_to_first_token"]) == 0

    def test_nan_values_skipped(self, rs):
        rs.update({"time_to_first_token": float("nan"), "error": None})
        assert len(rs._values["time_to_first_token"]) == 0

    def test_sums_accumulated(self, rs):
        rs.update({"num_tokens_output": 10, "error": None})
        rs.update({"num_tokens_output": 20, "error": None})
        assert rs._sums["num_tokens_output"] == 30

    def test_values_sorted(self, rs):
        rs.update({"time_to_first_token": 0.5, "error": None})
        rs.update({"time_to_first_token": 0.1, "error": None})
        rs.update({"time_to_first_token": 0.3, "error": None})
        assert rs._values["time_to_first_token"] == [0.1, 0.3, 0.5]


# ── to_stats ─────────────────────────────────────────────────────────────────


class TestToStats:
    def test_basic_stats(self, populated_rs):
        stats = populated_rs.to_stats(
            end_time=datetime(
                year=2026, month=4, day=16, hour=1, minute=1, second=0, tzinfo=UTC
            )
        )
        assert stats["failed_requests"] == 1
        assert "time_to_first_token-p50" in stats
        assert "time_to_last_token-average" in stats
        assert "num_tokens_output-p90" in stats

    def test_with_run_context(self, populated_rs):
        stats = populated_rs.to_stats(
            end_time=datetime(
                year=2026, month=4, day=16, hour=1, minute=1, second=0, tzinfo=UTC
            ),
            result_dict={"model_id": "test"},
        )
        assert stats["model_id"] == "test"
        # 3 reqs in 60sec = 3 rpm
        assert stats["requests_per_minute"] == pytest.approx(3.0)
        assert stats["failed_requests_rate"] == pytest.approx(1 / 3)
        assert stats["total_output_tokens"] == 83

    def test_without_run_context(self, populated_rs):
        stats = populated_rs.to_stats(end_time=datetime.now(tz=UTC))
        assert stats["failed_requests"] == 1
        assert stats["total_input_tokens"] == 330
        assert stats["total_output_tokens"] == 83

    def test_empty_stats(self, rs):
        stats = rs.to_stats(end_time=datetime.now(tz=UTC))
        assert stats["failed_requests"] == 0
        assert stats["total_input_tokens"] == 0

    def test_requests_per_minute(self, populated_rs):
        stats = populated_rs.to_stats(end_time=datetime.now(tz=UTC))
        # 3 reqs in 60 seconds = 3 rpm (Invariant of end time)
        assert stats["requests_per_minute"] == pytest.approx(3.0)

    def test_output_tps_(self, populated_rs):
        # (25 + 30 + 28) over 120 seconds - end-time dependent
        stats = populated_rs.to_stats(
            end_time=datetime(
                year=2026, month=4, day=16, hour=1, minute=2, second=0, tzinfo=UTC
            ),
        )
        assert stats["output_tps"] == pytest.approx((25 + 30 + 28) / 120)

    def test_no_input_rate_stats_when_single_request(self, rs):
        """With only one request, first == last, no window to compute rates."""
        rs.update(
            {
                "time_to_first_token": 0.3,
                "time_to_last_token": 0.8,
                "time_per_output_token": 0.02,
                "num_tokens_input": 100,
                "num_tokens_output": 25,
                "error": None,
                "request_time": datetime(
                    year=2026, month=4, day=16, hour=1, minute=0, second=0, tzinfo=UTC
                ),
            }
        )
        stats = rs.to_stats(end_time=datetime.now(tz=UTC))
        assert "requests_per_minute" not in stats

    def test_no_rate_stats_when_no_sends(self, rs):
        stats = rs.to_stats(end_time=datetime.now(tz=UTC))
        assert "requests_per_minute" not in stats
        assert "output_tps" not in stats

    def test_send_window_helper(self, rs):
        assert rs._send_window() is None
        rs.update(
            {
                "request_time": datetime(
                    year=2026, month=4, day=16, hour=1, minute=0, second=0, tzinfo=UTC
                ),
            }
        )
        assert rs._send_window() is None
        rs.update(
            {
                # Same timestamp
                "request_time": datetime(
                    year=2026, month=4, day=16, hour=1, minute=0, second=0, tzinfo=UTC
                ),
            }
        )
        assert rs._send_window() is None
        assert rs._send_window() is None
        rs.update(
            {
                # Later timestamp
                "request_time": datetime(
                    year=2026, month=4, day=16, hour=1, minute=0, second=23, tzinfo=UTC
                ),
            }
        )
        assert rs._send_window() == pytest.approx(23.0)
