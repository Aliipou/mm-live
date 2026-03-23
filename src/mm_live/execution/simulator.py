"""
Fill simulator for paper trading.

In paper trading mode, we simulate fills based on real trade data.
A fill occurs when:
  - A sell-initiated trade (buyer_maker=True) crosses our bid price
  - A buy-initiated trade (buyer_maker=False) crosses our ask price

This is more realistic than simple "did any trade happen?" because
it accounts for price priority: our quote only gets filled if the
trade price is at or through our level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mm_live.feed.binance_ws import TradeEvent


@dataclass
class FillSimulator:
    """
    Simulates order fills from real trade data.

    Parameters
    ----------
    fill_qty:
        Fixed quantity (BTC) per fill. Real systems use partial fills,
        but fixed qty simplifies P&L tracking in paper mode.
    """

    fill_qty: float = 0.001  # 0.001 BTC ≈ $60-100 per fill

    def simulate_fill(
        self, trade: TradeEvent, quotes: Any
    ) -> dict[str, Any] | None:
        """
        Check if a trade would fill our bid or ask.

        Returns fill dict or None if no fill.

        Fill dict keys:
            side: "buy" (we bought at bid) or "sell" (we sold at ask)
            price: fill price
            qty: fill quantity
            ts: trade timestamp
        """
        if trade.is_buyer_maker:
            # Sell-initiated: someone sold aggressively → hits bids
            # Our bid gets filled if trade.price <= our bid
            if trade.price <= quotes.bid:
                return {
                    "side": "buy",
                    "price": quotes.bid,
                    "qty": self.fill_qty,
                    "ts": trade.timestamp_ms,
                    "market_price": trade.price,
                }
        else:
            # Buy-initiated: someone bought aggressively → hits asks
            # Our ask gets filled if trade.price >= our ask
            if trade.price >= quotes.ask:
                return {
                    "side": "sell",
                    "price": quotes.ask,
                    "qty": self.fill_qty,
                    "ts": trade.timestamp_ms,
                    "market_price": trade.price,
                }
        return None
