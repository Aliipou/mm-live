"""
L2 order book state machine.

Maintains a real-time view of the best bid/ask and mid-price
from Binance depth update streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OrderBook:
    """
    Level-2 order book for a single symbol.

    Binance sends full snapshots (depthUpdate) with bids and asks as
    [price, quantity] pairs. Quantity == "0" means remove that level.
    """

    symbol: str
    bids: dict[float, float] = field(default_factory=dict)  # price → qty
    asks: dict[float, float] = field(default_factory=dict)
    last_update_id: int = 0

    def apply_update(self, update: dict) -> None:
        """
        Apply a Binance depthUpdate message.

        Parameters
        ----------
        update:
            Parsed JSON from Binance @depth stream:
            {"b": [["price", "qty"], ...], "a": [...], "u": last_update_id}
        """
        for price_str, qty_str in update.get("b", []):
            price = float(price_str)
            qty = float(qty_str)
            if qty == 0.0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        for price_str, qty_str in update.get("a", []):
            price = float(price_str)
            qty = float(qty_str)
            if qty == 0.0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

        self.last_update_id = update.get("u", self.last_update_id)

    @property
    def best_bid(self) -> float | None:
        """Highest bid price."""
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> float | None:
        """Lowest ask price."""
        return min(self.asks) if self.asks else None

    @property
    def mid(self) -> float | None:
        """Mid-price = (best_bid + best_ask) / 2."""
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return (b + a) / 2

    @property
    def spread(self) -> float | None:
        """Bid-ask spread."""
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return a - b

    @property
    def is_ready(self) -> bool:
        """True once we have at least one bid and one ask."""
        return bool(self.bids and self.asks)
