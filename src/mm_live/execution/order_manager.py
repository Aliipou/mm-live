"""
Order manager for live trading (paper → live transition).

In paper trading: tracks desired quotes, logs what would be sent.
In live trading: sends REST/WebSocket orders to exchange via BinanceClient.

The interface is the same in both modes. Switch with MM_LIVE=1 env var.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """An open order on the exchange (or simulated)."""

    order_id: str
    side: str       # "buy" or "sell"
    price: float
    qty: float
    status: str = "open"   # open | filled | cancelled


@dataclass
class OrderManager:
    """
    Manages the market maker's outstanding quotes.

    In paper mode: just tracks state and logs.
    In live mode: calls Binance REST API via :class:`~mm_live.execution.binance_client.BinanceClient`.

    Maintains exactly 2 orders at a time: one bid, one ask.
    On each timer event, cancel-and-replace if quotes have moved.

    Parameters
    ----------
    paper_mode:
        When ``True`` no real orders are sent.  All actions are logged at
        DEBUG level with a ``[PAPER]`` prefix.
    min_price_move:
        Minimum absolute price change that triggers a cancel-and-replace.
        Prevents excessive churn when quotes drift by sub-tick amounts.
    client:
        A :class:`~mm_live.execution.binance_client.BinanceClient` instance.
        Required when ``paper_mode=False``; ignored otherwise.
    """

    paper_mode: bool = True
    min_price_move: float = 0.01   # don't requote unless price moved > this
    client: Any | None = field(default=None, repr=False)   # BinanceClient | None

    # Internal state -------------------------------------------------------
    _bid_order: Order | None = field(init=False, default=None)
    _ask_order: Order | None = field(init=False, default=None)
    _order_counter: int = field(init=False, default=0)
    # Live-mode: order_id (str) → Order, kept in sync with the exchange.
    _orders: dict[str, Order] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not self.paper_mode and self.client is None:
            raise ValueError("OrderManager: client must be provided when paper_mode=False")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update_quotes(
        self,
        bid_price: float,
        ask_price: float,
        symbol: str,
        qty: float = 0.001,
    ) -> tuple[bool, bool]:
        """
        Update bid and ask quotes. Cancel-and-replace if moved significantly.

        Compatible with the previous synchronous ``update_quotes(bid, ask, qty)``
        call signature for paper mode; callers should ``await`` this in all modes.

        Parameters
        ----------
        bid_price:
            New desired bid price.
        ask_price:
            New desired ask price.
        symbol:
            Instrument symbol (e.g. ``"BTCUSDT"``).  Ignored in paper mode.
        qty:
            Order quantity.

        Returns
        -------
        tuple[bool, bool]
            ``(bid_updated, ask_updated)`` — ``True`` for each side that was
            actually cancelled and replaced.
        """
        bid_updated = False
        ask_updated = False

        # --- bid side ---
        if (
            self._bid_order is None
            or abs(self._bid_order.price - bid_price) >= self.min_price_move
        ):
            await self._cancel_order_async(self._bid_order, symbol)
            self._bid_order = await self._place_order_async("buy", bid_price, qty, symbol)
            bid_updated = True

        # --- ask side ---
        if (
            self._ask_order is None
            or abs(self._ask_order.price - ask_price) >= self.min_price_move
        ):
            await self._cancel_order_async(self._ask_order, symbol)
            self._ask_order = await self._place_order_async("sell", ask_price, qty, symbol)
            ask_updated = True

        return bid_updated, ask_updated

    async def cancel_all(self, symbol: str = "") -> None:
        """
        Cancel all outstanding quotes (called by the risk circuit breaker).

        In live mode this issues DELETE /api/v3/openOrders for *symbol*
        (which atomically cancels every open order on that symbol) and
        then clears the local order cache.

        Parameters
        ----------
        symbol:
            Instrument symbol.  Required in live mode; optional in paper mode.
        """
        if self.paper_mode:
            self._cancel_order(self._bid_order)
            self._cancel_order(self._ask_order)
            self._bid_order = None
            self._ask_order = None
            logger.warning("[PAPER] All quotes cancelled by risk manager")
            return

        # Live mode: single REST call cancels everything atomically.
        if symbol:
            try:
                cancelled = await self.client.cancel_all_orders(symbol)
                logger.warning(
                    "[LIVE] cancel_all → %d order(s) cancelled for %s",
                    len(cancelled),
                    symbol,
                )
            except Exception:
                logger.exception("[LIVE] cancel_all failed for %s", symbol)
        else:
            # Fallback: cancel individually from local cache.
            tasks = [
                self.client.cancel_order(order.order_id.split(":")[0], order.order_id)
                for order in list(self._orders.values())
            ]
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        logger.error("[LIVE] cancel individual order failed: %s", r)

        self._bid_order = None
        self._ask_order = None
        self._orders.clear()

    def mark_filled(self, order_id: str) -> None:
        """
        Mark a tracked live order as filled (called from fill-event handler).

        This removes the order from the internal tracking dict and clears
        the relevant side pointer so the next ``update_quotes`` call will
        place a fresh order.

        Parameters
        ----------
        order_id:
            Exchange order ID (str representation of Binance's numeric ID).
        """
        order = self._orders.pop(str(order_id), None)
        if order is None:
            return
        order.status = "filled"
        logger.info("[LIVE] Order %s (%s) marked as filled", order_id, order.side)
        if self._bid_order and self._bid_order.order_id == str(order_id):
            self._bid_order = None
        if self._ask_order and self._ask_order.order_id == str(order_id):
            self._ask_order = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_bid(self) -> float | None:
        """Current bid price, or ``None`` if no bid order exists."""
        return self._bid_order.price if self._bid_order else None

    @property
    def current_ask(self) -> float | None:
        """Current ask price, or ``None`` if no ask order exists."""
        return self._ask_order.price if self._ask_order else None

    # ------------------------------------------------------------------
    # Internal helpers — async versions (used by update_quotes / cancel_all)
    # ------------------------------------------------------------------

    async def _place_order_async(
        self, side: str, price: float, qty: float, symbol: str
    ) -> Order:
        """Place an order via REST (live) or simulate (paper)."""
        if self.paper_mode:
            return self._place_order(side, price, qty)

        # Live mode: call Binance REST.
        binance_side = "BUY" if side == "buy" else "SELL"
        try:
            resp = await self.client.place_order(
                symbol=symbol,
                side=binance_side,
                price=price,
                qty=qty,
            )
            order_id = str(resp["orderId"])
            order = Order(order_id=order_id, side=side, price=price, qty=qty)
            self._orders[order_id] = order
            logger.info(
                "[LIVE] Placed %s @ %.2f qty=%.6f id=%s",
                side,
                price,
                qty,
                order_id,
            )
            return order
        except Exception:
            logger.exception(
                "[LIVE] Failed to place %s @ %.2f qty=%.6f",
                side,
                price,
                qty,
            )
            # Return a placeholder so the internal state doesn't go None;
            # the next requote cycle will retry.
            self._order_counter += 1
            placeholder_id = f"err-{self._order_counter:06d}"
            return Order(
                order_id=placeholder_id,
                side=side,
                price=price,
                qty=qty,
                status="error",
            )

    async def _cancel_order_async(
        self, order: Order | None, symbol: str
    ) -> None:
        """Cancel an order via REST (live) or simulate (paper)."""
        if order is None:
            return

        if self.paper_mode:
            self._cancel_order(order)
            return

        # Live mode: skip placeholder / error orders.
        if order.status in ("cancelled", "filled", "error"):
            return

        try:
            await self.client.cancel_order(symbol, order.order_id)
            order.status = "cancelled"
            self._orders.pop(order.order_id, None)
            logger.info(
                "[LIVE] Cancelled %s @ %.2f id=%s",
                order.side,
                order.price,
                order.order_id,
            )
        except Exception:
            logger.exception(
                "[LIVE] Failed to cancel %s @ %.2f id=%s",
                order.side,
                order.price,
                order.order_id,
            )

    # ------------------------------------------------------------------
    # Internal helpers — synchronous (paper mode only)
    # ------------------------------------------------------------------

    def _place_order(self, side: str, price: float, qty: float) -> Order:
        """Create a simulated paper order and log it."""
        self._order_counter += 1
        order_id = f"paper-{self._order_counter:06d}"
        order = Order(order_id=order_id, side=side, price=price, qty=qty)
        logger.debug("[PAPER] Place %s @ %.2f qty=%.4f id=%s", side, price, qty, order_id)
        return order

    def _cancel_order(self, order: Order | None) -> None:
        """Mark a simulated paper order as cancelled and log it."""
        if order is None:
            return
        order.status = "cancelled"
        logger.debug(
            "[PAPER] Cancel %s @ %.2f id=%s", order.side, order.price, order.order_id
        )
