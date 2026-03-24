"""
Tests for mm_live.execution.rate_limiter.TokenBucket and BinanceRateLimiter.

Time-sensitive tests mock time.monotonic to avoid real sleeps.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from mm_live.execution.rate_limiter import BinanceRateLimiter, TokenBucket


# ---------------------------------------------------------------------------
# TokenBucket — basic token consumption
# ---------------------------------------------------------------------------

class TestTokenBucketAcquire:
    async def test_acquire_consumes_tokens(self) -> None:
        bucket = TokenBucket(rate=10.0, capacity=10.0)
        initial = bucket.available()
        await bucket.acquire(1.0)
        assert bucket.available() < initial

    async def test_acquire_consumes_exact_amount(self) -> None:
        bucket = TokenBucket(rate=10.0, capacity=10.0)
        # Drain all; the bucket starts full
        await bucket.acquire(10.0)
        # Available should now be ~0 (may have slightly refilled during await)
        assert bucket.available() < 1.0

    async def test_acquire_raises_when_tokens_exceed_capacity(self) -> None:
        bucket = TokenBucket(rate=10.0, capacity=5.0)
        with pytest.raises(ValueError, match="exceeds capacity"):
            await bucket.acquire(6.0)

    async def test_acquire_default_one_token(self) -> None:
        bucket = TokenBucket(rate=100.0, capacity=10.0)
        before = bucket._tokens
        await bucket.acquire()  # default = 1.0
        # tokens should have decreased by ~1 (before any refill in the tight loop)
        assert bucket._tokens <= before

    def test_constructor_rejects_zero_rate(self) -> None:
        with pytest.raises(ValueError):
            TokenBucket(rate=0.0, capacity=10.0)

    def test_constructor_rejects_negative_rate(self) -> None:
        with pytest.raises(ValueError):
            TokenBucket(rate=-1.0, capacity=10.0)

    def test_constructor_rejects_zero_capacity(self) -> None:
        with pytest.raises(ValueError):
            TokenBucket(rate=10.0, capacity=0.0)


# ---------------------------------------------------------------------------
# TokenBucket — capacity limit
# ---------------------------------------------------------------------------

class TestTokenBucketCapacity:
    def test_available_does_not_exceed_capacity(self) -> None:
        capacity = 5.0
        bucket = TokenBucket(rate=100.0, capacity=capacity)
        # Force a large elapsed time by manipulating _last_refill
        bucket._last_refill -= 1000.0  # pretend 1000 seconds passed
        assert bucket.available() <= capacity

    def test_refill_clamps_to_capacity(self) -> None:
        bucket = TokenBucket(rate=100.0, capacity=5.0)
        bucket._tokens = 0.0
        bucket._last_refill -= 1000.0  # pretend lots of time passed
        bucket._refill()
        assert bucket._tokens == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# TokenBucket — refill over time (mocked)
# ---------------------------------------------------------------------------

class TestTokenBucketRefill:
    def test_refill_adds_correct_tokens(self) -> None:
        bucket = TokenBucket(rate=10.0, capacity=100.0)
        bucket._tokens = 0.0
        t0 = bucket._last_refill

        # Simulate 2 seconds passing
        with patch("time.monotonic", return_value=t0 + 2.0):
            bucket._refill()

        assert bucket._tokens == pytest.approx(20.0)

    def test_refill_does_not_overfill(self) -> None:
        bucket = TokenBucket(rate=10.0, capacity=5.0)
        bucket._tokens = 4.5
        t0 = bucket._last_refill

        # Simulate 10 seconds passing — would produce 100 tokens, but capped at 5
        with patch("time.monotonic", return_value=t0 + 10.0):
            bucket._refill()

        assert bucket._tokens == pytest.approx(5.0)

    async def test_acquire_waits_for_refill(self) -> None:
        """Bucket is empty; acquire should complete after tokens refill."""
        # Very high rate so the test doesn't hang: 1000 tokens/sec
        bucket = TokenBucket(rate=1000.0, capacity=10.0)
        await bucket.acquire(10.0)  # drain fully
        # Now acquire 1 more — should complete very quickly (1ms at 1000/s)
        await asyncio.wait_for(bucket.acquire(1.0), timeout=1.0)


# ---------------------------------------------------------------------------
# BinanceRateLimiter
# ---------------------------------------------------------------------------

class TestBinanceRateLimiter:
    async def test_acquire_order_acquires_all_three_buckets(self) -> None:
        rl = BinanceRateLimiter()
        before_sec = rl.available_orders_sec()
        before_day = rl.available_orders_day()
        before_req = rl.available_request_weight()

        await rl.acquire_order()

        assert rl.available_orders_sec() < before_sec
        assert rl.available_orders_day() < before_day
        assert rl.available_request_weight() < before_req

    async def test_acquire_request_only_touches_request_bucket(self) -> None:
        rl = BinanceRateLimiter()
        before_sec = rl.available_orders_sec()
        before_day = rl.available_orders_day()

        await rl.acquire_request(weight=1)

        # Order buckets should be untouched
        assert rl.available_orders_sec() == pytest.approx(before_sec, abs=0.1)
        assert rl.available_orders_day() == pytest.approx(before_day, abs=1.0)

    async def test_acquire_request_with_weight_5(self) -> None:
        rl = BinanceRateLimiter()
        before = rl.available_request_weight()
        await rl.acquire_request(weight=5)
        assert rl.available_request_weight() < before

    async def test_concurrent_acquirers_are_serialized(self) -> None:
        """Concurrent callers share the same bucket; combined consumption is tracked."""
        # Use a very low refill rate so tokens don't replenish before we check.
        # capacity=10, rate=0.001/s: 3 acquires consume 3 tokens; refill in that
        # time is negligible (< 0.001 tokens).
        bucket = TokenBucket(rate=0.001, capacity=10.0)
        await asyncio.gather(
            bucket.acquire(1.0),
            bucket.acquire(1.0),
            bucket.acquire(1.0),
        )
        # 10 - 3 = 7 tokens left; refill is negligible at 0.001 tokens/s
        assert bucket.available() < 9.0

    async def test_available_orders_sec_decreases_after_acquire_order(self) -> None:
        rl = BinanceRateLimiter()
        before = rl.available_orders_sec()
        await rl.acquire_order()
        assert rl.available_orders_sec() < before

    async def test_available_orders_day_decreases_after_acquire_order(self) -> None:
        rl = BinanceRateLimiter()
        before = rl.available_orders_day()
        await rl.acquire_order()
        assert rl.available_orders_day() < before

    async def test_multiple_order_acquires_accumulate(self) -> None:
        rl = BinanceRateLimiter()
        before = rl.available_orders_sec()
        await rl.acquire_order()
        await rl.acquire_order()
        after = rl.available_orders_sec()
        # Two acquires should consume two tokens
        assert before - after >= 1.9  # slight tolerance for refill
