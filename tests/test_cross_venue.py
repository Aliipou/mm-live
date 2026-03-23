"""
Tests for mm_live.strategy.cross_venue.CrossVenueStrategy and ArbSignal.

All tests use in-memory UnifiedBook instances built from synthetic OrderBooks.
No network I/O is involved.
"""

from __future__ import annotations

import pytest

from mm_live.feed.orderbook import OrderBook
from mm_live.feed.unified_book import UnifiedBook
from mm_live.strategy.cross_venue import ArbSignal, CrossVenueStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ready_book(
    symbol: str, bid: float, bid_size: float, ask: float, ask_size: float
) -> OrderBook:
    book = OrderBook(symbol=symbol)
    book.bids = {bid: bid_size}
    book.asks = {ask: ask_size}
    return book


def _two_venue_book(
    venue_a_bid: float,
    venue_a_bid_size: float,
    venue_a_ask: float,
    venue_a_ask_size: float,
    venue_b_bid: float,
    venue_b_bid_size: float,
    venue_b_ask: float,
    venue_b_ask_size: float,
) -> UnifiedBook:
    ub = UnifiedBook()
    ub.update("binance", _ready_book("BTCUSDT", venue_a_bid, venue_a_bid_size, venue_a_ask, venue_a_ask_size))
    ub.update("okx",     _ready_book("BTC-USDT", venue_b_bid, venue_b_bid_size, venue_b_ask, venue_b_ask_size))
    return ub


# ---------------------------------------------------------------------------
# check_arb — insufficient venues
# ---------------------------------------------------------------------------

class TestCheckArbInsufficientVenues:
    def test_no_arb_with_empty_book(self) -> None:
        strategy = CrossVenueStrategy()
        ub = UnifiedBook()
        signal = strategy.check_arb(ub)
        assert signal.exists is False

    def test_no_arb_with_single_venue(self) -> None:
        strategy = CrossVenueStrategy()
        ub = UnifiedBook()
        ub.update("binance", _ready_book("BTCUSDT", 50000.0, 1.0, 50200.0, 0.5))
        signal = strategy.check_arb(ub)
        assert signal.exists is False

    def test_no_arb_when_same_venue_best_on_both_sides(self) -> None:
        strategy = CrossVenueStrategy(min_net_spread=0.0)
        ub = UnifiedBook()
        # Binance has higher bid AND lower ask than OKX
        ub.update("binance", _ready_book("BTCUSDT", 50100.0, 1.0, 50150.0, 1.0))
        ub.update("okx",     _ready_book("BTC-USDT", 49900.0, 1.0, 50300.0, 1.0))
        signal = strategy.check_arb(ub)
        # best_bid = binance(50100), best_ask = binance(50150) → same venue
        assert signal.exists is False


# ---------------------------------------------------------------------------
# check_arb — net_spread below threshold
# ---------------------------------------------------------------------------

class TestCheckArbBelowThreshold:
    def test_no_arb_when_net_spread_below_min(self) -> None:
        strategy = CrossVenueStrategy(fee_bps=7.0, min_net_spread=100.0)
        # small gross spread = 10, fee_drag >> 10, so net < 100
        ub = _two_venue_book(
            50010.0, 1.0, 50300.0, 1.0,
            49900.0, 1.0, 50000.0, 1.0,
        )
        # best_bid = binance(50010), best_ask = okx(50000) → gross = 10
        signal = strategy.check_arb(ub)
        assert signal.exists is False

    def test_no_arb_when_gross_spread_is_zero(self) -> None:
        strategy = CrossVenueStrategy(min_net_spread=0.01)
        ub = _two_venue_book(
            50000.0, 1.0, 50200.0, 1.0,
            49900.0, 1.0, 50000.0, 1.0,
        )
        # best_bid = binance(50000), best_ask = okx(50000) → gross = 0
        signal = strategy.check_arb(ub)
        assert signal.exists is False


# ---------------------------------------------------------------------------
# check_arb — arb exists
# ---------------------------------------------------------------------------

class TestCheckArbExists:
    def _arb_setup(self) -> tuple[CrossVenueStrategy, UnifiedBook]:
        strategy = CrossVenueStrategy(fee_bps=1.0, min_net_spread=5.0, max_qty=0.01)
        # Binance bid=50200, OKX ask=50100 → gross=100, fee tiny
        ub = _two_venue_book(
            50200.0, 0.5, 50500.0, 0.5,   # binance: bid=50200, ask=50500
            49800.0, 0.5, 50100.0, 0.3,   # okx: bid=49800, ask=50100
        )
        return strategy, ub

    def test_arb_exists_true(self) -> None:
        strategy, ub = self._arb_setup()
        signal = strategy.check_arb(ub)
        assert signal.exists is True

    def test_arb_buy_venue_is_cheapest_ask(self) -> None:
        strategy, ub = self._arb_setup()
        signal = strategy.check_arb(ub)
        assert signal.buy_venue == "okx"

    def test_arb_sell_venue_is_highest_bid(self) -> None:
        strategy, ub = self._arb_setup()
        signal = strategy.check_arb(ub)
        assert signal.sell_venue == "binance"

    def test_arb_buy_price_is_ask_price(self) -> None:
        strategy, ub = self._arb_setup()
        signal = strategy.check_arb(ub)
        assert signal.buy_price == pytest.approx(50100.0)

    def test_arb_sell_price_is_bid_price(self) -> None:
        strategy, ub = self._arb_setup()
        signal = strategy.check_arb(ub)
        assert signal.sell_price == pytest.approx(50200.0)


# ---------------------------------------------------------------------------
# gross_spread and net_spread
# ---------------------------------------------------------------------------

class TestSpreadCalculations:
    def test_gross_spread_equals_sell_minus_buy(self) -> None:
        strategy = CrossVenueStrategy(fee_bps=1.0, min_net_spread=5.0, max_qty=0.01)
        ub = _two_venue_book(
            50200.0, 0.5, 50500.0, 0.5,
            49800.0, 0.5, 50100.0, 0.3,
        )
        signal = strategy.check_arb(ub)
        expected_gross = signal.sell_price - signal.buy_price
        assert signal.gross_spread == pytest.approx(expected_gross)

    def test_net_spread_equals_gross_minus_fee_drag(self) -> None:
        fee_bps = 2.0
        strategy = CrossVenueStrategy(fee_bps=fee_bps, min_net_spread=5.0, max_qty=0.01)
        ub = _two_venue_book(
            50300.0, 0.5, 50600.0, 0.5,
            49800.0, 0.5, 50100.0, 0.3,
        )
        signal = strategy.check_arb(ub)
        mid = (signal.sell_price + signal.buy_price) / 2.0
        expected_fee_drag = 2.0 * (fee_bps / 10_000.0) * mid
        expected_net = signal.gross_spread - expected_fee_drag
        assert signal.net_spread == pytest.approx(expected_net)

    def test_gross_spread_is_positive_when_arb_exists(self) -> None:
        strategy = CrossVenueStrategy(fee_bps=1.0, min_net_spread=5.0, max_qty=0.01)
        ub = _two_venue_book(
            50200.0, 0.5, 50500.0, 0.5,
            49800.0, 0.5, 50100.0, 0.3,
        )
        signal = strategy.check_arb(ub)
        assert signal.gross_spread > 0


# ---------------------------------------------------------------------------
# qty sizing
# ---------------------------------------------------------------------------

class TestQtySizing:
    def test_qty_is_min_of_bid_size_ask_size_max_qty(self) -> None:
        strategy = CrossVenueStrategy(fee_bps=1.0, min_net_spread=5.0, max_qty=0.01)
        # bid_size=0.5, ask_size=0.3, max_qty=0.01 → min = 0.01
        ub = _two_venue_book(
            50200.0, 0.5, 50500.0, 0.5,
            49800.0, 0.5, 50100.0, 0.3,
        )
        signal = strategy.check_arb(ub)
        assert signal.qty == pytest.approx(min(0.5, 0.3, 0.01))

    def test_qty_limited_by_ask_size(self) -> None:
        strategy = CrossVenueStrategy(fee_bps=1.0, min_net_spread=5.0, max_qty=1.0)
        ub = _two_venue_book(
            50200.0, 5.0, 50500.0, 5.0,
            49800.0, 5.0, 50100.0, 0.1,  # ask_size is the limiting factor
        )
        signal = strategy.check_arb(ub)
        assert signal.qty == pytest.approx(0.1)

    def test_zero_qty_returns_no_arb(self) -> None:
        strategy = CrossVenueStrategy(fee_bps=1.0, min_net_spread=5.0, max_qty=0.01)
        ub = _two_venue_book(
            50200.0, 0.0, 50500.0, 0.0,
            49800.0, 0.0, 50100.0, 0.0,  # all sizes zero
        )
        signal = strategy.check_arb(ub)
        assert signal.exists is False


# ---------------------------------------------------------------------------
# should_hedge
# ---------------------------------------------------------------------------

class TestShouldHedge:
    def test_should_hedge_true_when_inventory_equals_max(self) -> None:
        strategy = CrossVenueStrategy()
        assert strategy.should_hedge(0.1, max_inventory=0.1) is True

    def test_should_hedge_true_when_inventory_exceeds_max(self) -> None:
        strategy = CrossVenueStrategy()
        assert strategy.should_hedge(0.5, max_inventory=0.1) is True

    def test_should_hedge_true_for_short_position(self) -> None:
        strategy = CrossVenueStrategy()
        assert strategy.should_hedge(-0.2, max_inventory=0.1) is True

    def test_should_hedge_false_when_within_limits(self) -> None:
        strategy = CrossVenueStrategy()
        assert strategy.should_hedge(0.05, max_inventory=0.1) is False

    def test_should_hedge_false_at_zero_inventory(self) -> None:
        strategy = CrossVenueStrategy()
        assert strategy.should_hedge(0.0, max_inventory=0.1) is False

    def test_should_hedge_false_when_max_inventory_zero(self) -> None:
        strategy = CrossVenueStrategy()
        assert strategy.should_hedge(999.0, max_inventory=0.0) is False

    def test_should_hedge_false_when_max_inventory_negative(self) -> None:
        strategy = CrossVenueStrategy()
        assert strategy.should_hedge(999.0, max_inventory=-1.0) is False
