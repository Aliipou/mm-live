"""
Order flow imbalance signal.

Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
Range: [-1, +1]. Positive = more buy pressure, negative = more sell pressure.

Why this matters:
    Kyle's model shows informed traders hide in order flow.
    Order book imbalance is a predictive signal for short-term price direction.
    Adding imbalance to fair value estimation reduces adverse selection.

Reference:
    Cartea, Jaimungal, Ricci (2018). Buy Low, Sell High: A High Frequency
    Trading Perspective. SIAM Journal on Financial Mathematics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from mm_live.feed.orderbook import OrderBook


@dataclass
class OrderFlowImbalance:
    """
    Real-time order flow imbalance from the L2 order book.

    Parameters
    ----------
    depth_levels:
        Number of price levels to include in the imbalance calculation.
        More levels = less noisy but slower signal.
    ema_alpha:
        Smoothing factor for the EMA of raw imbalance. Higher = faster.
    alpha_impact:
        How much imbalance shifts the fair value estimate (USD per unit imbalance).
        E.g., alpha=5.0 means full imbalance (+1) shifts FV by $5.
    """

    depth_levels: int = 5
    ema_alpha: float = 0.2
    alpha_impact: float = 2.0

    _smoothed: float = field(init=False, default=0.0)
    _n: int = field(init=False, default=0)

    def update(self, book: OrderBook) -> float:
        """
        Compute imbalance from current order book state.

        Returns smoothed imbalance in [-1, +1].
        """
        bid_vol = 0.0
        ask_vol = 0.0

        # Sum top N bid levels
        sorted_bids = sorted(book.bids.items(), reverse=True)
        for _, qty in sorted_bids[: self.depth_levels]:
            bid_vol += qty

        # Sum top N ask levels
        sorted_asks = sorted(book.asks.items())
        for _, qty in sorted_asks[: self.depth_levels]:
            ask_vol += qty

        total = bid_vol + ask_vol
        if total == 0:
            return self._smoothed

        raw = (bid_vol - ask_vol) / total

        # EMA smoothing
        if self._n == 0:
            self._smoothed = raw
        else:
            self._smoothed = (1 - self.ema_alpha) * self._smoothed + self.ema_alpha * raw

        self._n += 1
        return self._smoothed

    def fair_value_adjustment(self, imbalance: float) -> float:
        """
        Convert imbalance to a fair value adjustment (USD).

        fair_value += alpha_impact * imbalance
        """
        return self.alpha_impact * imbalance

    @property
    def current(self) -> float:
        return self._smoothed
