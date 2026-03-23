"""
Order manager for live trading (stub for paper → live transition).

In paper trading: tracks desired quotes, logs what would be sent.
In live trading: sends REST/WebSocket orders to exchange.

The interface is the same in both modes. Switch with MM_LIVE=1 env var.
"""

from __future__ import annotations

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
    In live mode (future): would call exchange API.

    Maintains exactly 2 orders at a time: one bid, one ask.
    On each timer event, cancel-and-replace if quotes have moved.
    """

    paper_mode: bool = True
    min_price_move: float = 0.01   # don't requote unless price moved > this

    _bid_order: Order | None = field(init=False, default=None)
    _ask_order: Order | None = field(init=False, default=None)
    _order_counter: int = field(init=False, default=0)

    def update_quotes(self, bid: float, ask: float, qty: float = 0.001) -> tuple[bool, bool]:
        """
        Update bid and ask quotes. Cancel-and-replace if moved significantly.

        Returns (bid_updated, ask_updated).
        """
        bid_updated = False
        ask_updated = False

        # Check if bid needs updating
        if self._bid_order is None or abs(self._bid_order.price - bid) >= self.min_price_move:
            self._cancel_order(self._bid_order)
            self._bid_order = self._place_order("buy", bid, qty)
            bid_updated = True

        # Check if ask needs updating
        if self._ask_order is None or abs(self._ask_order.price - ask) >= self.min_price_move:
            self._cancel_order(self._ask_order)
            self._ask_order = self._place_order("sell", ask, qty)
            ask_updated = True

        return bid_updated, ask_updated

    def cancel_all(self) -> None:
        """Cancel all outstanding quotes (called by risk circuit breaker)."""
        self._cancel_order(self._bid_order)
        self._cancel_order(self._ask_order)
        self._bid_order = None
        self._ask_order = None
        logger.warning("All quotes cancelled by risk manager")

    def _place_order(self, side: str, price: float, qty: float) -> Order:
        self._order_counter += 1
        order_id = f"paper-{self._order_counter:06d}"
        order = Order(order_id=order_id, side=side, price=price, qty=qty)
        if self.paper_mode:
            logger.debug("[PAPER] Place %s @ %.2f qty=%.4f id=%s", side, price, qty, order_id)
        return order

    def _cancel_order(self, order: Order | None) -> None:
        if order is None:
            return
        order.status = "cancelled"
        if self.paper_mode:
            logger.debug("[PAPER] Cancel %s @ %.2f id=%s", order.side, order.price, order.order_id)

    @property
    def current_bid(self) -> float | None:
        return self._bid_order.price if self._bid_order else None

    @property
    def current_ask(self) -> float | None:
        return self._ask_order.price if self._ask_order else None
