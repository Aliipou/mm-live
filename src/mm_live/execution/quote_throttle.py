"""Quote throttle — suppresses redundant re-posts when price has not moved enough."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class QuoteThrottle:
    """Decides whether a new bid/ask pair warrants sending updated quotes.

    Two independent conditions must *both* be satisfied before an update is
    considered necessary:

    1. **Price move** – at least one leg (bid or ask) has moved by at least
       ``min_price_move`` USD since the last posted quote.
    2. **Minimum interval** – at least ``min_interval_ms`` milliseconds have
       elapsed since the last posted quote.

    Typical usage::

        throttle = QuoteThrottle()
        if throttle.should_update(new_bid, new_ask):
            await post_quotes(new_bid, new_ask)
            throttle.record_update(new_bid, new_ask)

    The two methods are intentionally separate so the caller can avoid
    recording an update if the actual order placement fails.
    """

    min_price_move: float = 0.10
    """Minimum price movement in USD before a re-post is warranted."""

    min_interval_ms: float = 50.0
    """Minimum time between re-posts in milliseconds."""

    # State — excluded from the dataclass __repr__ for brevity
    _last_bid: float = field(default=float("nan"), init=False, repr=False)
    _last_ask: float = field(default=float("nan"), init=False, repr=False)
    _last_update_ns: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_update(self, new_bid: float, new_ask: float) -> bool:
        """Return ``True`` if the new quotes should be posted.

        Args:
            new_bid: The proposed new best bid price.
            new_ask: The proposed new best ask price.

        Returns:
            ``True`` when *both* the minimum interval has elapsed *and*
            either the bid or the ask has moved by at least
            ``min_price_move``.  Always returns ``True`` when no quote has
            been posted yet (first call after construction or a reset).
        """
        # First update — always allow.
        import math

        if math.isnan(self._last_bid) or math.isnan(self._last_ask):
            return True

        # --- Time gate ---------------------------------------------------
        elapsed_ms = (time.monotonic_ns() - self._last_update_ns) / 1_000_000.0
        if elapsed_ms < self.min_interval_ms:
            return False

        # --- Price move gate ---------------------------------------------
        bid_move = abs(new_bid - self._last_bid)
        ask_move = abs(new_ask - self._last_ask)
        if bid_move < self.min_price_move and ask_move < self.min_price_move:
            return False

        return True

    def record_update(self, bid: float, ask: float) -> None:
        """Record that quotes at *bid* / *ask* have been successfully posted.

        Call this **after** the order placement succeeds so that a failed
        placement does not suppress future updates.

        Args:
            bid: The bid price that was posted.
            ask: The ask price that was posted.
        """
        self._last_bid = bid
        self._last_ask = ask
        self._last_update_ns = time.monotonic_ns()

    def reset(self) -> None:
        """Clear all state so the next :meth:`should_update` call returns ``True``."""
        import math

        self._last_bid = math.nan
        self._last_ask = math.nan
        self._last_update_ns = 0

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def last_bid(self) -> float:
        """The bid price from the most recently recorded update."""
        return self._last_bid

    @property
    def last_ask(self) -> float:
        """The ask price from the most recently recorded update."""
        return self._last_ask

    @property
    def ms_since_last_update(self) -> float:
        """Milliseconds elapsed since the last recorded update.

        Returns ``inf`` if no update has been recorded yet.
        """
        if self._last_update_ns == 0:
            return float("inf")
        return (time.monotonic_ns() - self._last_update_ns) / 1_000_000.0
