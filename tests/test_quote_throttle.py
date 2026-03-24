"""
Tests for mm_live.execution.quote_throttle.QuoteThrottle.

Tests cover first-call behaviour, time gate, price-move gate, record_update,
and reset.
"""

from __future__ import annotations

import math
import time

import pytest

from mm_live.execution.quote_throttle import QuoteThrottle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monotonic_ns_from_ms(ms: float) -> int:
    return int(ms * 1_000_000)


# ---------------------------------------------------------------------------
# First call
# ---------------------------------------------------------------------------

class TestFirstCall:
    def test_should_update_returns_true_on_first_call(self) -> None:
        throttle = QuoteThrottle()
        assert throttle.should_update(50000.0, 50100.0) is True

    def test_should_update_true_after_reset(self) -> None:
        throttle = QuoteThrottle()
        throttle.record_update(50000.0, 50100.0)
        throttle.reset()
        assert throttle.should_update(50000.0, 50100.0) is True

    def test_should_update_true_when_last_bid_is_nan(self) -> None:
        throttle = QuoteThrottle()
        assert math.isnan(throttle._last_bid)
        assert throttle.should_update(1.0, 2.0) is True

    def test_should_update_true_when_last_ask_is_nan(self) -> None:
        throttle = QuoteThrottle()
        # Manually set bid but leave ask as NaN
        throttle._last_bid = 50000.0
        assert throttle.should_update(50000.0, 50100.0) is True


# ---------------------------------------------------------------------------
# Time gate
# ---------------------------------------------------------------------------

class TestTimeGate:
    def test_returns_false_immediately_after_record(self) -> None:
        throttle = QuoteThrottle(min_interval_ms=50.0, min_price_move=0.10)
        throttle.record_update(50000.0, 50100.0)
        # Without waiting, time gate fires
        assert throttle.should_update(50001.0, 50101.0) is False

    def test_returns_true_after_interval_and_price_move(self) -> None:
        throttle = QuoteThrottle(min_interval_ms=50.0, min_price_move=0.10)
        t0_ns = time.monotonic_ns()
        throttle._last_bid = 50000.0
        throttle._last_ask = 50100.0
        throttle._last_update_ns = t0_ns - _monotonic_ns_from_ms(100)  # 100ms ago

        # Price moved enough
        assert throttle.should_update(50000.5, 50100.5) is True

    def test_still_false_if_only_time_passed_but_no_price_move(self) -> None:
        throttle = QuoteThrottle(min_interval_ms=50.0, min_price_move=1.0)
        t0_ns = time.monotonic_ns()
        throttle._last_bid = 50000.0
        throttle._last_ask = 50100.0
        throttle._last_update_ns = t0_ns - _monotonic_ns_from_ms(200)  # 200ms ago

        # Price barely moved
        assert throttle.should_update(50000.05, 50100.05) is False

    def test_ms_since_last_update_is_inf_before_any_record(self) -> None:
        throttle = QuoteThrottle()
        assert throttle.ms_since_last_update == float("inf")

    def test_ms_since_last_update_is_positive_after_record(self) -> None:
        throttle = QuoteThrottle()
        throttle.record_update(50000.0, 50100.0)
        assert throttle.ms_since_last_update >= 0.0


# ---------------------------------------------------------------------------
# Price move gate
# ---------------------------------------------------------------------------

class TestPriceMoveGate:
    def test_returns_false_if_bid_not_moved_enough(self) -> None:
        throttle = QuoteThrottle(min_interval_ms=0.0, min_price_move=1.0)
        t0_ns = time.monotonic_ns()
        throttle._last_bid = 50000.0
        throttle._last_ask = 50100.0
        throttle._last_update_ns = t0_ns - _monotonic_ns_from_ms(500)

        # Bid moves 0.5 (below threshold), ask moves 0.5 (below threshold)
        assert throttle.should_update(50000.5, 50100.5) is False

    def test_returns_true_if_bid_moved_enough(self) -> None:
        throttle = QuoteThrottle(min_interval_ms=0.0, min_price_move=1.0)
        t0_ns = time.monotonic_ns()
        throttle._last_bid = 50000.0
        throttle._last_ask = 50100.0
        throttle._last_update_ns = t0_ns - _monotonic_ns_from_ms(500)

        # Bid moves 2.0 (above threshold)
        assert throttle.should_update(50002.0, 50100.0) is True

    def test_returns_true_if_ask_moved_enough(self) -> None:
        throttle = QuoteThrottle(min_interval_ms=0.0, min_price_move=1.0)
        t0_ns = time.monotonic_ns()
        throttle._last_bid = 50000.0
        throttle._last_ask = 50100.0
        throttle._last_update_ns = t0_ns - _monotonic_ns_from_ms(500)

        # Ask moves 2.0 (above threshold)
        assert throttle.should_update(50000.0, 50102.0) is True

    def test_returns_false_if_neither_leg_moved(self) -> None:
        throttle = QuoteThrottle(min_interval_ms=0.0, min_price_move=0.10)
        t0_ns = time.monotonic_ns()
        throttle._last_bid = 50000.0
        throttle._last_ask = 50100.0
        throttle._last_update_ns = t0_ns - _monotonic_ns_from_ms(500)

        assert throttle.should_update(50000.0, 50100.0) is False


# ---------------------------------------------------------------------------
# record_update
# ---------------------------------------------------------------------------

class TestRecordUpdate:
    def test_record_update_stores_bid(self) -> None:
        throttle = QuoteThrottle()
        throttle.record_update(49999.0, 50001.0)
        assert throttle.last_bid == pytest.approx(49999.0)

    def test_record_update_stores_ask(self) -> None:
        throttle = QuoteThrottle()
        throttle.record_update(49999.0, 50001.0)
        assert throttle.last_ask == pytest.approx(50001.0)

    def test_record_update_sets_timestamp(self) -> None:
        throttle = QuoteThrottle()
        before = time.monotonic_ns()
        throttle.record_update(50000.0, 50100.0)
        after = time.monotonic_ns()
        assert before <= throttle._last_update_ns <= after

    def test_record_update_overwrites_previous(self) -> None:
        throttle = QuoteThrottle()
        throttle.record_update(50000.0, 50100.0)
        throttle.record_update(51000.0, 51100.0)
        assert throttle.last_bid == pytest.approx(51000.0)
        assert throttle.last_ask == pytest.approx(51100.0)


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_bid(self) -> None:
        throttle = QuoteThrottle()
        throttle.record_update(50000.0, 50100.0)
        throttle.reset()
        assert math.isnan(throttle._last_bid)

    def test_reset_clears_ask(self) -> None:
        throttle = QuoteThrottle()
        throttle.record_update(50000.0, 50100.0)
        throttle.reset()
        assert math.isnan(throttle._last_ask)

    def test_reset_clears_timestamp(self) -> None:
        throttle = QuoteThrottle()
        throttle.record_update(50000.0, 50100.0)
        throttle.reset()
        assert throttle._last_update_ns == 0

    def test_reset_makes_next_should_update_return_true(self) -> None:
        throttle = QuoteThrottle(min_interval_ms=10000.0, min_price_move=1000.0)
        throttle.record_update(50000.0, 50100.0)
        # Should be False normally (time gate and price gate both block)
        assert throttle.should_update(50000.0, 50100.0) is False
        throttle.reset()
        # After reset, always True
        assert throttle.should_update(50000.0, 50100.0) is True
