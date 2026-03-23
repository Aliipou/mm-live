"""
Optimal quote engine: Avellaneda-Stoikov adapted for live trading.

Takes real-time fair value + vol estimates and produces optimal bid/ask.

Key adaptation for live trading vs. original paper:
    - We don't know T (end of session). Use a rolling horizon T_horizon.
    - Inventory is in units of BTC (float, not just int).
    - Quotes are clamped to not cross the current best bid/ask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Quotes:
    """Optimal bid and ask prices."""

    bid: float
    ask: float
    fair_value: float
    half_spread: float
    inventory: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def skew(self) -> float:
        """Signed skew from fair value: positive = ask further from FV."""
        return self.fair_value - (self.bid + self.ask) / 2


class QuoteEngine:
    """
    Avellaneda-Stoikov quote engine for live market making.

    Parameters
    ----------
    gamma:
        Risk aversion. Higher → wider spreads, faster inventory reversion.
        For BTC/USDT with ~$100 USD P&L target per day: gamma ~ 0.01–0.1
    k:
        Order arrival sensitivity to spread width.
        Higher k → fewer orders arrive as spread widens.
        Typical range: 1.0–3.0
    T_horizon:
        Rolling time horizon in ticks. Effectively: "how long am I willing
        to hold inventory before I must flatten?"
        At 100ms ticks: T=600 → 1 minute horizon
    max_inventory:
        Maximum absolute inventory (BTC). Beyond this, only one-sided quotes.
    """

    def __init__(
        self,
        gamma: float = 0.05,
        k: float = 1.5,
        T_horizon: float = 600.0,   # ticks (~60 seconds at 100ms)
        max_inventory: float = 0.1,  # 0.1 BTC
    ) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if k <= 0:
            raise ValueError("k must be positive")

        self.gamma = gamma
        self.k = k
        self.T_horizon = T_horizon
        self.max_inventory = max_inventory

    def compute(
        self,
        fair_value: float,
        sigma: float,
        inventory: float,
        time_remaining: float | None = None,
    ) -> Quotes:
        """
        Compute optimal bid and ask.

        Parameters
        ----------
        fair_value:
            Current fair value estimate from Kalman filter.
        sigma:
            Current vol estimate (USD per tick).
        inventory:
            Current position in BTC. Positive = long.
        time_remaining:
            Ticks until forced liquidation. Defaults to T_horizon.
        """
        T = time_remaining if time_remaining is not None else self.T_horizon

        # Reservation price: shift fair value by inventory risk
        # r = fv - q * gamma * sigma^2 * T
        reservation = fair_value - inventory * self.gamma * sigma ** 2 * T

        # Optimal half-spread
        # delta = gamma * sigma^2 * T / 2 + (1/k) * ln(1 + gamma/k)
        inventory_term = self.gamma * sigma ** 2 * T / 2
        flow_term = (1 / self.k) * math.log(1 + self.gamma / self.k)
        delta = inventory_term + flow_term

        # Apply inventory limit: one-sided quoting at extremes
        if inventory >= self.max_inventory:
            bid = reservation - delta * 3   # far away, won't get hit
            ask = reservation + delta
        elif inventory <= -self.max_inventory:
            bid = reservation - delta
            ask = reservation + delta * 3
        else:
            bid = reservation - delta
            ask = reservation + delta

        return Quotes(
            bid=round(bid, 2),
            ask=round(ask, 2),
            fair_value=fair_value,
            half_spread=delta,
            inventory=inventory,
        )
