"""
Unified multi-venue order book aggregator.

Maintains the latest ``VenueQuote`` (best bid/ask snapshot) for each connected
venue and exposes helpers for cross-venue spread analysis used by the
cross-venue arbitrage strategy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from mm_live.feed.orderbook import OrderBook


@dataclass
class VenueQuote:
    """Best bid/ask snapshot for a single venue at a point in time."""

    venue: str          # e.g. "binance" | "okx"
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp_ms: int


class UnifiedBook:
    """
    Aggregates real-time ``OrderBook`` updates from multiple venues and exposes
    cross-venue best bid/ask and spread queries.

    Typical usage::

        ub = UnifiedBook()

        # Inside each venue's streaming loop:
        ub.update("binance", binance_book)
        ub.update("okx", okx_book)

        if ub.spread() is not None:
            print(f"Global spread: {ub.spread():.2f}")
        if ub.cross_spread() is not None and ub.cross_spread() < 0:
            print("Arbitrage opportunity detected!")
    """

    def __init__(self) -> None:
        self._quotes: dict[str, VenueQuote] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(self, venue: str, book: OrderBook) -> None:
        """
        Ingest the latest ``OrderBook`` snapshot from *venue*.

        Silently ignores books that are not yet ready (missing bids or asks).
        """
        if not book.is_ready:
            return

        best_bid_price = book.best_bid
        best_ask_price = book.best_ask

        # Both must be present (is_ready guarantees this, but be explicit).
        if best_bid_price is None or best_ask_price is None:
            return

        bid_size = book.bids.get(best_bid_price, 0.0)
        ask_size = book.asks.get(best_ask_price, 0.0)

        self._quotes[venue] = VenueQuote(
            venue=venue,
            bid=best_bid_price,
            ask=best_ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            timestamp_ms=int(time.time() * 1000),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def venues(self) -> list[str]:
        """Return the list of venues that have at least one valid quote."""
        return list(self._quotes.keys())

    def get(self, venue: str) -> VenueQuote | None:
        """Return the latest ``VenueQuote`` for *venue*, or ``None``."""
        return self._quotes.get(venue)

    def best_bid(self) -> VenueQuote | None:
        """
        Return the ``VenueQuote`` that offers the highest bid price across all
        venues.  Returns ``None`` if no venue has a valid quote yet.
        """
        if not self._quotes:
            return None
        return max(self._quotes.values(), key=lambda q: q.bid)

    def best_ask(self) -> VenueQuote | None:
        """
        Return the ``VenueQuote`` that offers the lowest ask price across all
        venues.  Returns ``None`` if no venue has a valid quote yet.
        """
        if not self._quotes:
            return None
        return min(self._quotes.values(), key=lambda q: q.ask)

    def spread(self) -> float | None:
        """
        Global best-ask minus global best-bid.

        A positive value is the normal (no-arb) state.
        Returns ``None`` when fewer than one venue has a valid quote.
        """
        bid_quote = self.best_bid()
        ask_quote = self.best_ask()
        if bid_quote is None or ask_quote is None:
            return None
        return ask_quote.ask - bid_quote.bid

    def cross_spread(self) -> float | None:
        """
        Cross-venue spread: ``best_bid_across_venues - best_ask_across_venues``.

        Interpretation
        --------------
        * Negative  – normal; best bid is below best ask (no arb).
        * Zero       – breakeven (before fees).
        * Positive   – arbitrage: you can simultaneously lift the cheapest ask
                       and hit the highest bid for a gross profit equal to this
                       value.

        Returns ``None`` when fewer than two venues have valid quotes, or when
        the best bid and best ask come from the same venue (no cross-venue arb
        is possible in that degenerate case).
        """
        if len(self._quotes) < 2:
            return None

        bid_quote = self.best_bid()
        ask_quote = self.best_ask()

        if bid_quote is None or ask_quote is None:
            return None

        # Cross-venue only: the two legs must be on different venues.
        if bid_quote.venue == ask_quote.venue:
            return None

        return bid_quote.bid - ask_quote.ask
