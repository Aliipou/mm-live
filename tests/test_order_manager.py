"""
Tests for mm_live.execution.order_manager.OrderManager.

Covers both paper and live trading modes, quote update logic, cancellation,
and fill tracking.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mm_live.execution.order_manager import Order, OrderManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_live_client() -> MagicMock:
    client = MagicMock()
    client.place_order = AsyncMock(return_value={"orderId": 42})
    client.cancel_order = AsyncMock(return_value={"orderId": 42, "status": "CANCELED"})
    client.cancel_all_orders = AsyncMock(return_value=[{"orderId": 42}])
    return client


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_paper_mode_default(self) -> None:
        om = OrderManager()
        assert om.paper_mode is True

    def test_live_mode_requires_client(self) -> None:
        with pytest.raises(ValueError, match="client must be provided"):
            OrderManager(paper_mode=False, client=None)

    def test_live_mode_with_client_succeeds(self) -> None:
        client = _make_live_client()
        om = OrderManager(paper_mode=False, client=client)
        assert om.paper_mode is False

    def test_paper_mode_with_client_none_succeeds(self) -> None:
        om = OrderManager(paper_mode=True, client=None)
        assert om.paper_mode is True


# ---------------------------------------------------------------------------
# Paper mode — update_quotes
# ---------------------------------------------------------------------------

class TestPaperModeUpdateQuotes:
    async def test_paper_update_quotes_returns_bid_ask_updated(self) -> None:
        om = OrderManager(paper_mode=True)
        bid_updated, ask_updated = await om.update_quotes(49900.0, 50100.0, "BTCUSDT")
        assert bid_updated is True
        assert ask_updated is True

    async def test_paper_update_quotes_sets_bid_price(self) -> None:
        om = OrderManager(paper_mode=True)
        await om.update_quotes(49900.0, 50100.0, "BTCUSDT")
        assert om.current_bid == pytest.approx(49900.0)

    async def test_paper_update_quotes_sets_ask_price(self) -> None:
        om = OrderManager(paper_mode=True)
        await om.update_quotes(49900.0, 50100.0, "BTCUSDT")
        assert om.current_ask == pytest.approx(50100.0)

    async def test_paper_update_quotes_no_rest_calls(self) -> None:
        """Paper mode must not call any REST client methods."""
        client = _make_live_client()
        om = OrderManager(paper_mode=True, client=client)
        await om.update_quotes(49900.0, 50100.0, "BTCUSDT")
        client.place_order.assert_not_called()
        client.cancel_order.assert_not_called()

    async def test_paper_update_quotes_no_requote_when_price_stable(self) -> None:
        """If price moved less than min_price_move, do not replace."""
        om = OrderManager(paper_mode=True, min_price_move=1.0)
        await om.update_quotes(50000.0, 50100.0, "BTCUSDT")
        # Move price by less than 1.0
        bid_updated, ask_updated = await om.update_quotes(50000.5, 50100.4, "BTCUSDT")
        assert bid_updated is False
        assert ask_updated is False

    async def test_paper_update_quotes_requotes_when_price_moves(self) -> None:
        om = OrderManager(paper_mode=True, min_price_move=1.0)
        await om.update_quotes(50000.0, 50100.0, "BTCUSDT")
        # Move price by more than min_price_move
        bid_updated, ask_updated = await om.update_quotes(50002.0, 50102.0, "BTCUSDT")
        assert bid_updated is True
        assert ask_updated is True


# ---------------------------------------------------------------------------
# Paper mode — cancel_all
# ---------------------------------------------------------------------------

class TestPaperModeCancelAll:
    async def test_paper_cancel_all_no_rest_calls(self) -> None:
        client = _make_live_client()
        om = OrderManager(paper_mode=True, client=client)
        await om.update_quotes(49900.0, 50100.0, "BTCUSDT")
        await om.cancel_all("BTCUSDT")
        client.cancel_all_orders.assert_not_called()

    async def test_paper_cancel_all_clears_bid(self) -> None:
        om = OrderManager(paper_mode=True)
        await om.update_quotes(49900.0, 50100.0, "BTCUSDT")
        await om.cancel_all("BTCUSDT")
        assert om.current_bid is None

    async def test_paper_cancel_all_clears_ask(self) -> None:
        om = OrderManager(paper_mode=True)
        await om.update_quotes(49900.0, 50100.0, "BTCUSDT")
        await om.cancel_all("BTCUSDT")
        assert om.current_ask is None

    async def test_paper_cancel_all_when_no_orders_does_not_raise(self) -> None:
        om = OrderManager(paper_mode=True)
        await om.cancel_all("BTCUSDT")  # no orders — should not raise


# ---------------------------------------------------------------------------
# Live mode — update_quotes
# ---------------------------------------------------------------------------

class TestLiveModeUpdateQuotes:
    async def test_live_update_quotes_calls_place_order_bid(self) -> None:
        client = _make_live_client()
        om = OrderManager(paper_mode=False, client=client)
        await om.update_quotes(49900.0, 50100.0, "BTCUSDT")
        # Expect one BUY and one SELL call
        calls = [c.kwargs["side"] for c in client.place_order.call_args_list]
        assert "BUY" in calls

    async def test_live_update_quotes_calls_place_order_ask(self) -> None:
        client = _make_live_client()
        om = OrderManager(paper_mode=False, client=client)
        await om.update_quotes(49900.0, 50100.0, "BTCUSDT")
        calls = [c.kwargs["side"] for c in client.place_order.call_args_list]
        assert "SELL" in calls

    async def test_live_update_quotes_no_replace_when_price_stable(self) -> None:
        client = _make_live_client()
        client.place_order = AsyncMock(side_effect=[
            {"orderId": 1},
            {"orderId": 2},
            {"orderId": 3},
            {"orderId": 4},
        ])
        om = OrderManager(paper_mode=False, client=client, min_price_move=1.0)
        await om.update_quotes(50000.0, 50100.0, "BTCUSDT")
        first_count = client.place_order.call_count  # should be 2

        # Move less than min_price_move
        await om.update_quotes(50000.4, 50100.3, "BTCUSDT")
        # No extra calls expected
        assert client.place_order.call_count == first_count

    async def test_live_update_quotes_does_replace_when_price_moves(self) -> None:
        client = _make_live_client()
        client.place_order = AsyncMock(side_effect=[
            {"orderId": 1},
            {"orderId": 2},
            {"orderId": 3},
            {"orderId": 4},
        ])
        om = OrderManager(paper_mode=False, client=client, min_price_move=1.0)
        await om.update_quotes(50000.0, 50100.0, "BTCUSDT")
        await om.update_quotes(50002.0, 50102.0, "BTCUSDT")
        # 2 initial + 2 replacements = 4
        assert client.place_order.call_count == 4

    async def test_live_update_quotes_tracks_order_id(self) -> None:
        client = _make_live_client()
        client.place_order = AsyncMock(side_effect=[{"orderId": 77}, {"orderId": 88}])
        om = OrderManager(paper_mode=False, client=client)
        await om.update_quotes(49900.0, 50100.0, "BTCUSDT")
        assert "77" in om._orders
        assert "88" in om._orders


# ---------------------------------------------------------------------------
# Live mode — cancel_all
# ---------------------------------------------------------------------------

class TestLiveModeCancelAll:
    async def test_live_cancel_all_calls_cancel_all_orders(self) -> None:
        client = _make_live_client()
        om = OrderManager(paper_mode=False, client=client)
        await om.cancel_all("BTCUSDT")
        client.cancel_all_orders.assert_called_once_with("BTCUSDT")

    async def test_live_cancel_all_clears_bid(self) -> None:
        client = _make_live_client()
        om = OrderManager(paper_mode=False, client=client)
        await om.update_quotes(49900.0, 50100.0, "BTCUSDT")
        await om.cancel_all("BTCUSDT")
        assert om.current_bid is None

    async def test_live_cancel_all_clears_ask(self) -> None:
        client = _make_live_client()
        om = OrderManager(paper_mode=False, client=client)
        await om.update_quotes(49900.0, 50100.0, "BTCUSDT")
        await om.cancel_all("BTCUSDT")
        assert om.current_ask is None

    async def test_live_cancel_all_clears_order_cache(self) -> None:
        client = _make_live_client()
        client.place_order = AsyncMock(side_effect=[{"orderId": 1}, {"orderId": 2}])
        om = OrderManager(paper_mode=False, client=client)
        await om.update_quotes(49900.0, 50100.0, "BTCUSDT")
        assert len(om._orders) == 2
        await om.cancel_all("BTCUSDT")
        assert len(om._orders) == 0


# ---------------------------------------------------------------------------
# mark_filled
# ---------------------------------------------------------------------------

class TestMarkFilled:
    async def test_mark_filled_removes_order_from_tracking(self) -> None:
        client = _make_live_client()
        client.place_order = AsyncMock(side_effect=[{"orderId": 100}, {"orderId": 101}])
        om = OrderManager(paper_mode=False, client=client)
        await om.update_quotes(49900.0, 50100.0, "BTCUSDT")
        assert "100" in om._orders
        om.mark_filled("100")
        assert "100" not in om._orders

    async def test_mark_filled_clears_bid_pointer(self) -> None:
        client = _make_live_client()
        client.place_order = AsyncMock(side_effect=[{"orderId": 200}, {"orderId": 201}])
        om = OrderManager(paper_mode=False, client=client)
        await om.update_quotes(49900.0, 50100.0, "BTCUSDT")
        bid_id = om._bid_order.order_id if om._bid_order else None
        assert bid_id is not None
        om.mark_filled(bid_id)
        assert om.current_bid is None

    async def test_mark_filled_clears_ask_pointer(self) -> None:
        client = _make_live_client()
        client.place_order = AsyncMock(side_effect=[{"orderId": 300}, {"orderId": 301}])
        om = OrderManager(paper_mode=False, client=client)
        await om.update_quotes(49900.0, 50100.0, "BTCUSDT")
        ask_id = om._ask_order.order_id if om._ask_order else None
        assert ask_id is not None
        om.mark_filled(ask_id)
        assert om.current_ask is None

    def test_mark_filled_unknown_id_does_not_raise(self) -> None:
        om = OrderManager(paper_mode=True)
        om.mark_filled("nonexistent-order-id")  # should not raise

    async def test_mark_filled_sets_status_filled(self) -> None:
        client = _make_live_client()
        client.place_order = AsyncMock(return_value={"orderId": 400})
        om = OrderManager(paper_mode=False, client=client)
        # Only place one order (we'll manually inspect)
        await om._place_order_async("buy", 49900.0, 0.001, "BTCUSDT")
        order = om._orders.get("400")
        assert order is not None
        om.mark_filled("400")
        assert order.status == "filled"
