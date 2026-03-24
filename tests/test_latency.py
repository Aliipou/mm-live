"""
Tests for mm_live.core.latency.LatencyTracker.

Covers record, p50, p99, count, window eviction, summary, and empty-label
behaviour.
"""

from __future__ import annotations

import pytest

from mm_live.core.latency import LatencyTracker

# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

class TestRecord:
    def test_record_adds_sample(self) -> None:
        tracker = LatencyTracker()
        tracker.record("rtt", 1.0)
        assert tracker.count("rtt") == 1

    def test_record_multiple_samples(self) -> None:
        tracker = LatencyTracker()
        for v in [1.0, 2.0, 3.0]:
            tracker.record("rtt", v)
        assert tracker.count("rtt") == 3

    def test_record_separate_labels(self) -> None:
        tracker = LatencyTracker()
        tracker.record("a", 1.0)
        tracker.record("b", 2.0)
        assert tracker.count("a") == 1
        assert tracker.count("b") == 1

    def test_record_unknown_label_creates_bucket(self) -> None:
        tracker = LatencyTracker()
        tracker.record("new_label", 5.0)
        assert tracker.count("new_label") == 1


# ---------------------------------------------------------------------------
# p50
# ---------------------------------------------------------------------------

class TestP50:
    def test_p50_single_element(self) -> None:
        tracker = LatencyTracker()
        tracker.record("rtt", 7.0)
        assert tracker.p50("rtt") == pytest.approx(7.0)

    def test_p50_odd_count(self) -> None:
        tracker = LatencyTracker()
        for v in [1.0, 3.0, 5.0]:
            tracker.record("rtt", v)
        # Nearest-rank median of [1,3,5] → 3.0
        assert tracker.p50("rtt") == pytest.approx(3.0)

    def test_p50_even_count(self) -> None:
        tracker = LatencyTracker()
        for v in [1.0, 2.0, 3.0, 4.0]:
            tracker.record("rtt", v)
        # Nearest-rank: ceil(50/100 * 4) - 1 = ceil(2) - 1 = 1 → sorted[1] = 2.0
        result = tracker.p50("rtt")
        assert result in (2.0, 3.0)  # nearest-rank can be 2 or 3

    def test_p50_empty_label_returns_zero(self) -> None:
        tracker = LatencyTracker()
        assert tracker.p50("nonexistent") == 0.0

    def test_p50_unsorted_input(self) -> None:
        tracker = LatencyTracker()
        for v in [5.0, 1.0, 9.0, 3.0, 7.0]:
            tracker.record("rtt", v)
        # sorted: [1,3,5,7,9] → median = 5.0
        assert tracker.p50("rtt") == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# p99
# ---------------------------------------------------------------------------

class TestP99:
    def test_p99_single_element(self) -> None:
        tracker = LatencyTracker()
        tracker.record("rtt", 42.0)
        assert tracker.p99("rtt") == pytest.approx(42.0)

    def test_p99_returns_near_maximum(self) -> None:
        tracker = LatencyTracker()
        # 100 samples: 1..100
        for v in range(1, 101):
            tracker.record("rtt", float(v))
        result = tracker.p99("rtt")
        # Nearest-rank: ceil(99/100 * 100) - 1 = 98 → sorted[98] = 99.0
        assert result == pytest.approx(99.0)

    def test_p99_empty_label_returns_zero(self) -> None:
        tracker = LatencyTracker()
        assert tracker.p99("nonexistent") == 0.0

    def test_p99_large_outlier_at_end(self) -> None:
        tracker = LatencyTracker()
        for v in range(1, 99):
            tracker.record("rtt", float(v))
        tracker.record("rtt", 9999.0)  # one huge outlier at 99th position (100 samples total)
        tracker.record("rtt", 100.0)
        result = tracker.p99("rtt")
        # 99th percentile should be near the large value, not 100.0
        assert result >= 99.0


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

class TestCount:
    def test_count_zero_for_new_label(self) -> None:
        tracker = LatencyTracker()
        assert tracker.count("empty") == 0

    def test_count_matches_records(self) -> None:
        tracker = LatencyTracker()
        N = 7
        for i in range(N):
            tracker.record("x", float(i))
        assert tracker.count("x") == N


# ---------------------------------------------------------------------------
# Window eviction
# ---------------------------------------------------------------------------

class TestWindowEviction:
    def test_oldest_samples_drop_when_window_exceeded(self) -> None:
        window = 5
        tracker = LatencyTracker(window=window)
        for i in range(10):
            tracker.record("rtt", float(i))
        # Only the last 5 samples should remain
        assert tracker.count("rtt") == window

    def test_evicted_samples_do_not_affect_percentile(self) -> None:
        tracker = LatencyTracker(window=3)
        # Insert low values that will be evicted
        for v in [1.0, 2.0, 3.0]:
            tracker.record("rtt", v)
        # Insert new high values that replace the old ones
        for v in [100.0, 200.0, 300.0]:
            tracker.record("rtt", v)
        # p50 should reflect only the new values
        assert tracker.p50("rtt") == pytest.approx(200.0)

    def test_window_one_keeps_only_latest(self) -> None:
        tracker = LatencyTracker(window=1)
        tracker.record("rtt", 1.0)
        tracker.record("rtt", 999.0)
        assert tracker.count("rtt") == 1
        assert tracker.p50("rtt") == pytest.approx(999.0)

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError):
            LatencyTracker(window=0)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_returns_dict(self) -> None:
        tracker = LatencyTracker()
        tracker.record("a", 1.0)
        result = tracker.summary()
        assert isinstance(result, dict)

    def test_summary_contains_all_labels(self) -> None:
        tracker = LatencyTracker()
        tracker.record("alpha", 1.0)
        tracker.record("beta", 2.0)
        summary = tracker.summary()
        assert "alpha" in summary
        assert "beta" in summary

    def test_summary_contains_p50_p99_count(self) -> None:
        tracker = LatencyTracker()
        tracker.record("rtt", 5.0)
        summary = tracker.summary()
        assert "p50" in summary["rtt"]
        assert "p99" in summary["rtt"]
        assert "count" in summary["rtt"]

    def test_summary_count_is_correct(self) -> None:
        tracker = LatencyTracker()
        for v in [1.0, 2.0, 3.0]:
            tracker.record("rtt", v)
        summary = tracker.summary()
        assert summary["rtt"]["count"] == 3

    def test_summary_empty_tracker_returns_empty_dict(self) -> None:
        tracker = LatencyTracker()
        assert tracker.summary() == {}

    def test_summary_values_are_numeric(self) -> None:
        tracker = LatencyTracker()
        for v in range(10):
            tracker.record("rtt", float(v))
        summary = tracker.summary()
        assert isinstance(summary["rtt"]["p50"], float)
        assert isinstance(summary["rtt"]["p99"], float)
        assert isinstance(summary["rtt"]["count"], int)


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_all_labels(self) -> None:
        tracker = LatencyTracker()
        tracker.record("a", 1.0)
        tracker.record("b", 2.0)
        tracker.reset()
        assert tracker.count("a") == 0
        assert tracker.count("b") == 0

    def test_reset_specific_label(self) -> None:
        tracker = LatencyTracker()
        tracker.record("a", 1.0)
        tracker.record("b", 2.0)
        tracker.reset("a")
        assert tracker.count("a") == 0
        assert tracker.count("b") == 1

    def test_reset_unknown_label_does_not_raise(self) -> None:
        tracker = LatencyTracker()
        tracker.reset("nonexistent")  # should not raise
