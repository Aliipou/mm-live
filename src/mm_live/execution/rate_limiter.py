"""Token bucket rate limiter for Binance API request and order weight limits."""

from __future__ import annotations

import asyncio
import time


class TokenBucket:
    """Generic token bucket for rate limiting.

    Tokens refill at *rate* tokens per second up to *capacity*.  Callers
    ``await acquire(n)`` and the coroutine suspends until enough tokens are
    available.
    """

    def __init__(self, rate: float, capacity: float) -> None:
        """Initialise the bucket.

        Args:
            rate: Refill speed in tokens per second.
            capacity: Maximum number of tokens the bucket can hold.
        """
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")

        self._rate: float = rate
        self._capacity: float = capacity
        self._tokens: float = capacity
        self._last_refill: float = time.monotonic()
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Add tokens that have accumulated since the last call."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acquire(self, tokens: float = 1.0) -> None:
        """Wait until *tokens* are available, then consume them.

        Args:
            tokens: Number of tokens to consume (default 1).

        Raises:
            ValueError: If *tokens* exceeds the bucket capacity.
        """
        if tokens > self._capacity:
            raise ValueError(
                f"requested {tokens} tokens exceeds capacity {self._capacity}"
            )

        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # Sleep until enough tokens will have accumulated.
                deficit = tokens - self._tokens
                wait_seconds = deficit / self._rate
                await asyncio.sleep(wait_seconds)

    def available(self) -> float:
        """Return the current number of available tokens (snapshot).

        The value is computed without acquiring the lock so it may be
        slightly stale by the time the caller acts on it.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill
        return min(self._capacity, self._tokens + elapsed * self._rate)


class BinanceRateLimiter:
    """Composite rate limiter that enforces Binance API limits.

    Binance enforces three independent limits:

    * **Request weight** – 1 200 weight units per minute (rolling).
    * **Order rate**     – 10 new orders per second (rolling).
    * **Daily orders**   – 100 000 orders per UTC day.

    Each limit is modelled as a :class:`TokenBucket`.  Callers use
    :meth:`acquire_order` before placing any order and
    :meth:`acquire_request` before any other signed/unsigned REST call.
    """

    # Binance documented limits
    _REQUEST_WEIGHT_PER_MIN: int = 1_200
    _ORDERS_PER_SEC: int = 10
    _ORDERS_PER_DAY: int = 100_000

    def __init__(self) -> None:
        # Request-weight bucket: 1 200 weight/min → 20 weight/sec
        self._request_bucket = TokenBucket(
            rate=self._REQUEST_WEIGHT_PER_MIN / 60.0,
            capacity=float(self._REQUEST_WEIGHT_PER_MIN),
        )

        # Order-rate bucket: 10 orders/sec
        self._order_sec_bucket = TokenBucket(
            rate=float(self._ORDERS_PER_SEC),
            capacity=float(self._ORDERS_PER_SEC),
        )

        # Daily order bucket: 100 000 orders/day → ~1.157 orders/sec
        self._order_day_bucket = TokenBucket(
            rate=self._ORDERS_PER_DAY / 86_400.0,
            capacity=float(self._ORDERS_PER_DAY),
        )

    async def acquire_order(self) -> None:
        """Consume one order slot from both the per-second and daily buckets.

        Also consumes the default request weight (1) because every order
        placement is also an API request.
        """
        # Acquire all three limits concurrently where possible, but the
        # per-second bucket is the tightest constraint so we await it first
        # to get the most accurate throttling feel.
        await self._order_sec_bucket.acquire(1.0)
        await asyncio.gather(
            self._order_day_bucket.acquire(1.0),
            self._request_bucket.acquire(1.0),
        )

    async def acquire_request(self, weight: int = 1) -> None:
        """Consume *weight* units from the request-weight bucket.

        Args:
            weight: The documented weight of the endpoint being called
                    (default 1 for lightweight endpoints).
        """
        await self._request_bucket.acquire(float(weight))

    # ------------------------------------------------------------------
    # Introspection helpers (useful for dashboards / tests)
    # ------------------------------------------------------------------

    def available_request_weight(self) -> float:
        """Snapshot of remaining request weight tokens."""
        return self._request_bucket.available()

    def available_orders_sec(self) -> float:
        """Snapshot of remaining per-second order tokens."""
        return self._order_sec_bucket.available()

    def available_orders_day(self) -> float:
        """Snapshot of remaining daily order tokens."""
        return self._order_day_bucket.available()
