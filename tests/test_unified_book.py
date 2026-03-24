"""
Tests for mm_live.feed.unified_book.UnifiedBook and VenueQuote.

All tests operate on in-memory OrderBook instances; no network I/O.
"""

from __future__ import annotations

import pytest

from mm_live.feed.orderbook import OrderBook
from mm_live.feed.unified_book import UnifiedBook, VenueQuote

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ready_book(symbol: str, bid: float, bid_size: float, ask: float, ask_size: float) -> OrderBook:
    """Return a ready OrderBook with a single bid/ask level."""
    book = OrderBook(symbol=symbol)
    book.bids = {bid: bid_size}
    book.asks = {ask: ask_size}
    return book


def _empty_book(symbol: str) -> OrderBook:
    """Return a not-ready OrderBook with no levels."""
    return OrderBook(symbol=symbol)


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_stores_venue_quote_from_ready_book(self) -> None:
        ub = UnifiedBook()
        book = _ready_book("BTCUSDT", 50000.0, 1.0, 50100.0, 0.5)
        ub.update("binance", book)
        assert "binance" in ub.venues()

    def test_update_ignores_non_ready_book(self) -> None:
        ub = UnifiedBook()
        book = _empty_book("BTCUSDT")
        ub.update("binance", book)
        assert "binance" not in ub.venues()

    def test_update_stores_correct_bid(self) -> None:
        ub = UnifiedBook()
        ub.update("binance", _ready_book("BTCUSDT", 50000.0, 1.0, 50200.0, 0.5))
        q = ub.get("binance")
        assert q is not None
        assert q.bid == pytest.approx(50000.0)

    def test_update_stores_correct_ask(self) -> None:
        ub = UnifiedBook()
        ub.update("binance", _ready_book("BTCUSDT", 50000.0, 1.0, 50200.0, 0.5))
        q = ub.get("binance")
        assert q is not None
        assert q.ask == pytest.approx(50200.0)

    def test_update_stores_correct_bid_size(self) -> None:
        ub = UnifiedBook()
        ub.update("okx", _ready_book("BTC-USDT", 50000.0, 2.5, 50200.0, 1.0))
        q = ub.get("okx")
        assert q is not None
        assert q.bid_size == pytest.approx(2.5)

    def test_update_stores_correct_ask_size(self) -> None:
        ub = UnifiedBook()
        ub.update("okx", _ready_book("BTC-USDT", 50000.0, 2.5, 50200.0, 1.0))
        q = ub.get("okx")
        assert q is not None
        assert q.ask_size == pytest.approx(1.0)

    def test_update_overwrites_previous_quote(self) -> None:
        ub = UnifiedBook()
        ub.update("binance", _ready_book("BTCUSDT", 50000.0, 1.0, 50200.0, 0.5))
        ub.update("binance", _ready_book("BTCUSDT", 51000.0, 0.8, 51200.0, 0.3))
        q = ub.get("binance")
        assert q is not None
        assert q.bid == pytest.approx(51000.0)


# ---------------------------------------------------------------------------
# best_bid
# ---------------------------------------------------------------------------

class TestBestBid:
    def test_best_bid_returns_highest_bid_across_venues(self) -> None:
        ub = UnifiedBook()
        ub.update("binance", _ready_book("BTCUSDT", 50000.0, 1.0, 50200.0, 0.5))
        ub.update("okx",     _ready_book("BTC-USDT", 50050.0, 0.5, 50250.0, 0.5))
        best = ub.best_bid()
        assert best is not None
        assert best.bid == pytest.approx(50050.0)
        assert best.venue == "okx"

    def test_best_bid_returns_none_when_no_venues(self) -> None:
        ub = UnifiedBook()
        assert ub.best_bid() is None

    def test_best_bid_single_venue(self) -> None:
        ub = UnifiedBook()
        ub.update("binance", _ready_book("BTCUSDT", 49900.0, 1.0, 50100.0, 1.0))
        best = ub.best_bid()
        assert best is not None
        assert best.venue == "binance"


# ---------------------------------------------------------------------------
# best_ask
# ---------------------------------------------------------------------------

class TestBestAsk:
    def test_best_ask_returns_lowest_ask_across_venues(self) -> None:
        ub = UnifiedBook()
        ub.update("binance", _ready_book("BTCUSDT", 50000.0, 1.0, 50200.0, 0.5))
        ub.update("okx",     _ready_book("BTC-USDT", 50050.0, 0.5, 50150.0, 0.5))
        best = ub.best_ask()
        assert best is not None
        assert best.ask == pytest.approx(50150.0)
        assert best.venue == "okx"

    def test_best_ask_returns_none_when_no_venues(self) -> None:
        ub = UnifiedBook()
        assert ub.best_ask() is None

    def test_best_ask_single_venue(self) -> None:
        ub = UnifiedBook()
        ub.update("binance", _ready_book("BTCUSDT", 49900.0, 1.0, 50100.0, 1.0))
        best = ub.best_ask()
        assert best is not None
        assert best.venue == "binance"


# ---------------------------------------------------------------------------
# spread
# ---------------------------------------------------------------------------

class TestSpread:
    def test_spread_equals_best_ask_minus_best_bid(self) -> None:
        ub = UnifiedBook()
        ub.update("binance", _ready_book("BTCUSDT", 50000.0, 1.0, 50200.0, 0.5))
        spread = ub.spread()
        assert spread is not None
        assert spread == pytest.approx(200.0)

    def test_spread_across_venues(self) -> None:
        ub = UnifiedBook()
        ub.update("binance", _ready_book("BTCUSDT", 50000.0, 1.0, 50300.0, 0.5))
        ub.update("okx",     _ready_book("BTC-USDT", 50050.0, 0.5, 50150.0, 0.5))
        # best_bid = 50050 (okx), best_ask = 50150 (okx)
        spread = ub.spread()
        assert spread is not None
        assert spread == pytest.approx(100.0)

    def test_spread_returns_none_when_no_venues(self) -> None:
        ub = UnifiedBook()
        assert ub.spread() is None


# ---------------------------------------------------------------------------
# cross_spread
# ---------------------------------------------------------------------------

class TestCrossSpread:
    def test_cross_spread_none_with_single_venue(self) -> None:
        ub = UnifiedBook()
        ub.update("binance", _ready_book("BTCUSDT", 50000.0, 1.0, 50200.0, 0.5))
        assert ub.cross_spread() is None

    def test_cross_spread_none_when_no_venues(self) -> None:
        ub = UnifiedBook()
        assert ub.cross_spread() is None

    def test_cross_spread_positive_means_arb(self) -> None:
        ub = UnifiedBook()
        # Binance bid=50200, OKX ask=50100 → arb: buy OKX, sell Binance
        ub.update("binance", _ready_book("BTCUSDT", 50200.0, 1.0, 50500.0, 0.5))
        ub.update("okx",     _ready_book("BTC-USDT", 49900.0, 0.5, 50100.0, 1.0))
        cs = ub.cross_spread()
        assert cs is not None
        assert cs == pytest.approx(50200.0 - 50100.0)  # bid_binance - ask_okx = 100

    def test_cross_spread_negative_means_no_arb(self) -> None:
        ub = UnifiedBook()
        ub.update("binance", _ready_book("BTCUSDT", 50000.0, 1.0, 50200.0, 0.5))
        ub.update("okx",     _ready_book("BTC-USDT", 49800.0, 0.5, 50050.0, 1.0))
        cs = ub.cross_spread()
        assert cs is not None
        assert cs < 0  # best_bid(50000) - best_ask(50050) = -50

    def test_cross_spread_none_when_same_venue_is_best_on_both_sides(self) -> None:
        ub = UnifiedBook()
        # Binance has best bid AND best ask
        ub.update("binance", _ready_book("BTCUSDT", 50100.0, 1.0, 50200.0, 0.5))
        ub.update("okx",     _ready_book("BTC-USDT", 49900.0, 0.5, 50300.0, 1.0))
        # best_bid = binance(50100), best_ask = binance(50200) → same venue
        cs = ub.cross_spread()
        assert cs is None


# ---------------------------------------------------------------------------
# get / venues
# ---------------------------------------------------------------------------

class TestGetVenues:
    def test_get_returns_correct_venue_quote(self) -> None:
        ub = UnifiedBook()
        ub.update("binance", _ready_book("BTCUSDT", 50000.0, 1.0, 50200.0, 0.5))
        q = ub.get("binance")
        assert q is not None
        assert isinstance(q, VenueQuote)
        assert q.venue == "binance"

    def test_get_returns_none_for_unknown_venue(self) -> None:
        ub = UnifiedBook()
        assert ub.get("nonexistent") is None

    def test_venues_returns_list(self) -> None:
        ub = UnifiedBook()
        ub.update("binance", _ready_book("BTCUSDT", 50000.0, 1.0, 50200.0, 0.5))
        ub.update("okx",     _ready_book("BTC-USDT", 49900.0, 0.5, 50250.0, 1.0))
        venues = ub.venues()
        assert "binance" in venues
        assert "okx" in venues

    def test_venues_empty_initially(self) -> None:
        ub = UnifiedBook()
        assert ub.venues() == []

    def test_venues_does_not_include_non_ready(self) -> None:
        ub = UnifiedBook()
        ub.update("binance", _empty_book("BTCUSDT"))
        assert ub.venues() == []
