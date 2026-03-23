"""
Tests for mm_live.feed.okx_ws.OKXOrderBookFeed.

All tests operate on the message-parsing layer (_handle_message and
_apply_okx_book). No real WebSocket connections are made.
"""

from __future__ import annotations

import orjson
import pytest

from mm_live.feed.binance_ws import TradeEvent
from mm_live.feed.okx_ws import OKXOrderBookFeed
from mm_live.feed.orderbook import OrderBook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_books5_msg(
    bids: list[list[str]],
    asks: list[list[str]],
    ts: str = "1700000000000",
    inst_id: str = "BTC-USDT",
) -> bytes:
    """Build a raw OKX books5 snapshot WebSocket frame."""
    return orjson.dumps({
        "arg": {"channel": "books5", "instId": inst_id},
        "data": [
            {
                "bids": bids,
                "asks": asks,
                "ts": ts,
                "instId": inst_id,
            }
        ],
    })


def _make_trade_msg(
    price: str,
    size: str,
    side: str,
    ts: str = "1700000000123",
    inst_id: str = "BTC-USDT",
) -> bytes:
    return orjson.dumps({
        "arg": {"channel": "trades", "instId": inst_id},
        "data": [
            {
                "instId": inst_id,
                "tradeId": "12345",
                "px": price,
                "sz": size,
                "side": side,
                "ts": ts,
            }
        ],
    })


# ---------------------------------------------------------------------------
# books5 snapshot parsing
# ---------------------------------------------------------------------------

class TestBooks5Parsing:
    def test_books5_snapshot_sets_bids(self) -> None:
        feed = OKXOrderBookFeed("BTC-USDT")
        raw = _make_books5_msg(
            bids=[["50000.0", "1.5", "0", "1"]],
            asks=[["50100.0", "0.8", "0", "1"]],
        )
        result = feed._handle_message(raw)
        assert result is not None
        book, _ = result
        assert 50000.0 in book.bids
        assert book.bids[50000.0] == pytest.approx(1.5)

    def test_books5_snapshot_sets_asks(self) -> None:
        feed = OKXOrderBookFeed("BTC-USDT")
        raw = _make_books5_msg(
            bids=[["50000.0", "1.5", "0", "1"]],
            asks=[["50100.0", "0.8", "0", "1"]],
        )
        result = feed._handle_message(raw)
        assert result is not None
        book, _ = result
        assert 50100.0 in book.asks
        assert book.asks[50100.0] == pytest.approx(0.8)

    def test_books5_multiple_bid_levels(self) -> None:
        feed = OKXOrderBookFeed("BTC-USDT")
        raw = _make_books5_msg(
            bids=[
                ["50000.0", "1.0", "0", "1"],
                ["49900.0", "2.0", "0", "1"],
                ["49800.0", "3.0", "0", "1"],
            ],
            asks=[["50100.0", "0.5", "0", "1"]],
        )
        result = feed._handle_message(raw)
        assert result is not None
        book, _ = result
        assert len(book.bids) == 3

    def test_books5_multiple_ask_levels(self) -> None:
        feed = OKXOrderBookFeed("BTC-USDT")
        raw = _make_books5_msg(
            bids=[["50000.0", "1.0", "0", "1"]],
            asks=[
                ["50100.0", "0.5", "0", "1"],
                ["50200.0", "0.3", "0", "1"],
            ],
        )
        result = feed._handle_message(raw)
        assert result is not None
        book, _ = result
        assert len(book.asks) == 2

    def test_books5_zero_qty_levels_excluded(self) -> None:
        """Levels with qty=0 should not be stored."""
        feed = OKXOrderBookFeed("BTC-USDT")
        raw = _make_books5_msg(
            bids=[
                ["50000.0", "0.0", "0", "1"],   # should be excluded
                ["49900.0", "1.5", "0", "1"],
            ],
            asks=[["50100.0", "0.5", "0", "1"]],
        )
        result = feed._handle_message(raw)
        assert result is not None
        book, _ = result
        assert 50000.0 not in book.bids
        assert 49900.0 in book.bids

    def test_books5_returns_book_and_none_trade(self) -> None:
        feed = OKXOrderBookFeed("BTC-USDT")
        raw = _make_books5_msg(
            bids=[["50000.0", "1.0", "0", "1"]],
            asks=[["50100.0", "1.0", "0", "1"]],
        )
        result = feed._handle_message(raw)
        assert result is not None
        book, trade = result
        assert isinstance(book, OrderBook)
        assert trade is None

    def test_books5_sets_last_update_id(self) -> None:
        feed = OKXOrderBookFeed("BTC-USDT")
        raw = _make_books5_msg(
            bids=[["50000.0", "1.0", "0", "1"]],
            asks=[["50100.0", "1.0", "0", "1"]],
            ts="1700000001234",
        )
        feed._handle_message(raw)
        assert feed.book.last_update_id == 1700000001234

    def test_books5_book_is_ready_after_snapshot(self) -> None:
        feed = OKXOrderBookFeed("BTC-USDT")
        raw = _make_books5_msg(
            bids=[["50000.0", "1.0", "0", "1"]],
            asks=[["50100.0", "1.0", "0", "1"]],
        )
        feed._handle_message(raw)
        assert feed.book.is_ready is True


# ---------------------------------------------------------------------------
# Trade parsing — side mapping
# ---------------------------------------------------------------------------

class TestTradeParsing:
    def test_trade_sell_side_maps_to_is_buyer_maker_true(self) -> None:
        """OKX side='sell' (taker sold) → buyer was the maker."""
        feed = OKXOrderBookFeed("BTC-USDT")
        # Prime the book first so handle_message can return a result
        feed._handle_message(_make_books5_msg(
            [["50000.0", "1.0", "0", "1"]],
            [["50100.0", "1.0", "0", "1"]],
        ))
        raw = _make_trade_msg(price="50050.0", size="0.5", side="sell")
        result = feed._handle_message(raw)
        assert result is not None
        _, trade = result
        assert trade is not None
        assert trade.is_buyer_maker is True

    def test_trade_buy_side_maps_to_is_buyer_maker_false(self) -> None:
        """OKX side='buy' (taker bought) → seller was the maker."""
        feed = OKXOrderBookFeed("BTC-USDT")
        feed._handle_message(_make_books5_msg(
            [["50000.0", "1.0", "0", "1"]],
            [["50100.0", "1.0", "0", "1"]],
        ))
        raw = _make_trade_msg(price="50050.0", size="0.5", side="buy")
        result = feed._handle_message(raw)
        assert result is not None
        _, trade = result
        assert trade is not None
        assert trade.is_buyer_maker is False

    def test_trade_price_parsed_correctly(self) -> None:
        feed = OKXOrderBookFeed("BTC-USDT")
        feed._handle_message(_make_books5_msg(
            [["50000.0", "1.0", "0", "1"]],
            [["50100.0", "1.0", "0", "1"]],
        ))
        raw = _make_trade_msg(price="51234.56", size="0.25", side="buy")
        result = feed._handle_message(raw)
        assert result is not None
        _, trade = result
        assert trade is not None
        assert trade.price == pytest.approx(51234.56)

    def test_trade_qty_parsed_correctly(self) -> None:
        feed = OKXOrderBookFeed("BTC-USDT")
        feed._handle_message(_make_books5_msg(
            [["50000.0", "1.0", "0", "1"]],
            [["50100.0", "1.0", "0", "1"]],
        ))
        raw = _make_trade_msg(price="50000.0", size="1.23456", side="sell")
        result = feed._handle_message(raw)
        assert result is not None
        _, trade = result
        assert trade is not None
        assert trade.qty == pytest.approx(1.23456)

    def test_trade_timestamp_parsed(self) -> None:
        feed = OKXOrderBookFeed("BTC-USDT")
        feed._handle_message(_make_books5_msg(
            [["50000.0", "1.0", "0", "1"]],
            [["50100.0", "1.0", "0", "1"]],
        ))
        raw = _make_trade_msg(price="50000.0", size="1.0", side="buy", ts="1700000099999")
        result = feed._handle_message(raw)
        assert result is not None
        _, trade = result
        assert trade is not None
        assert trade.timestamp_ms == 1700000099999

    def test_trade_returns_trade_event_instance(self) -> None:
        feed = OKXOrderBookFeed("BTC-USDT")
        feed._handle_message(_make_books5_msg(
            [["50000.0", "1.0", "0", "1"]],
            [["50100.0", "1.0", "0", "1"]],
        ))
        raw = _make_trade_msg(price="50000.0", size="1.0", side="sell")
        result = feed._handle_message(raw)
        assert result is not None
        _, trade = result
        assert isinstance(trade, TradeEvent)


# ---------------------------------------------------------------------------
# Malformed / missing messages
# ---------------------------------------------------------------------------

class TestMalformedMessages:
    def test_invalid_json_returns_none(self) -> None:
        feed = OKXOrderBookFeed("BTC-USDT")
        assert feed._handle_message(b"not json {{{") is None

    def test_empty_bytes_returns_none(self) -> None:
        feed = OKXOrderBookFeed("BTC-USDT")
        assert feed._handle_message(b"") is None

    def test_subscription_confirm_returns_none(self) -> None:
        feed = OKXOrderBookFeed("BTC-USDT")
        msg = orjson.dumps({"event": "subscribe", "arg": {"channel": "books5", "instId": "BTC-USDT"}})
        assert feed._handle_message(msg) is None

    def test_error_event_returns_none(self) -> None:
        feed = OKXOrderBookFeed("BTC-USDT")
        msg = orjson.dumps({"event": "error", "code": "60018", "msg": "Invalid OK_ACCESS_KEY"})
        assert feed._handle_message(msg) is None

    def test_empty_data_list_returns_none(self) -> None:
        feed = OKXOrderBookFeed("BTC-USDT")
        msg = orjson.dumps({"arg": {"channel": "books5", "instId": "BTC-USDT"}, "data": []})
        assert feed._handle_message(msg) is None

    def test_unknown_channel_returns_none(self) -> None:
        feed = OKXOrderBookFeed("BTC-USDT")
        msg = orjson.dumps({
            "arg": {"channel": "tickers", "instId": "BTC-USDT"},
            "data": [{"last": "50000"}],
        })
        assert feed._handle_message(msg) is None

    def test_pong_string_is_not_parsed_as_json(self) -> None:
        """The 'pong' string is handled by _iter_with_ping, not _handle_message;
        passing it to _handle_message should not crash (returns None)."""
        feed = OKXOrderBookFeed("BTC-USDT")
        result = feed._handle_message("pong")
        assert result is None

    def test_missing_bids_key_does_not_crash(self) -> None:
        """books5 message missing 'bids' key should not raise."""
        feed = OKXOrderBookFeed("BTC-USDT")
        msg = orjson.dumps({
            "arg": {"channel": "books5", "instId": "BTC-USDT"},
            "data": [{"asks": [["50100.0", "1.0", "0", "1"]], "ts": "1700000000000"}],
        })
        result = feed._handle_message(msg)
        # Should not raise; book may have asks but no bids
        # result should be the (book, None) tuple
        assert result is not None

    def test_missing_asks_key_does_not_crash(self) -> None:
        """books5 message missing 'asks' key should not raise."""
        feed = OKXOrderBookFeed("BTC-USDT")
        msg = orjson.dumps({
            "arg": {"channel": "books5", "instId": "BTC-USDT"},
            "data": [{"bids": [["50000.0", "1.0", "0", "1"]], "ts": "1700000000000"}],
        })
        result = feed._handle_message(msg)
        assert result is not None
