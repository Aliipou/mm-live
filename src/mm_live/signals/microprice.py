"""
Microprice — queue-weighted fair value estimate.

Better than mid-price because it accounts for queue imbalance at the inside.
If bid has 10x more size than ask, price is more likely to move up.

    microprice = (ask · bid_size + bid · ask_size) / (bid_size + ask_size)

At balanced queues: microprice == mid.
When bid queue dominates: microprice > mid (bullish pressure).

Empirically outperforms mid as a short-horizon price predictor.
Reference: Stoikov (2018), "The micro-price: a high frequency estimator of future prices"
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mm_live.feed.orderbook import OrderBook


@dataclass
class MicropriceSignal:
    """
    Queue-weighted microprice estimator using best bid/ask sizes.

    Computes the Stoikov microprice:

        microprice = (ask · bid_size + bid · ask_size) / (bid_size + ask_size)

    The weighting cross-multiplies prices with the *opposite* side's size, so
    a thick bid queue (bid_size >> ask_size) pulls microprice above mid.
    """

    _last: float = field(init=False, default=0.0)

    def update(self, book: OrderBook) -> float:
        """
        Compute and store microprice from the current order book state.

        Returns microprice when the book is ready, otherwise falls back to
        ``book.mid`` (and further to 0.0 if mid is also unavailable).
        """
        if not book.is_ready:
            mid = book.mid
            self._last = mid if mid is not None else 0.0
            return self._last

        bid = book.best_bid
        ask = book.best_ask

        # best_bid / best_ask are guaranteed non-None when is_ready is True,
        # but we guard anyway to satisfy the type checker.
        if bid is None or ask is None:
            mid = book.mid
            self._last = mid if mid is not None else 0.0
            return self._last

        bid_size = book.bids[bid]
        ask_size = book.asks[ask]

        total = bid_size + ask_size
        if total == 0.0:
            mid = book.mid
            self._last = mid if mid is not None else 0.0
            return self._last

        self._last = (ask * bid_size + bid * ask_size) / total
        return self._last

    @property
    def last(self) -> float:
        """Last computed microprice (0.0 before first update)."""
        return self._last

    def deviation_from_mid(self, book: OrderBook) -> float:
        """
        Return ``microprice - mid``.

        Positive value → bid queue dominates → bullish queue pressure.
        Negative value → ask queue dominates → bearish queue pressure.
        Zero when queues are balanced or the book is not ready.
        """
        mid = book.mid
        if mid is None:
            return 0.0
        return self._last - mid
