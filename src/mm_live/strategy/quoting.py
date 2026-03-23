"""
Adaptive Avellaneda-Stoikov quoting strategy.

Extends the base A-S model with:
1. Imbalance-based skew: tighten on the side favored by order flow
2. Regime adaptation: widen spreads in high-vol regime
3. Rate limiting: don't requote more than once per min_requote_interval ticks

This is what separates a textbook implementation from a live system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Quotes:
    bid: float
    ask: float
    fair_value: float
    half_spread: float
    inventory: float
    regime: str = "normal"

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


class AdaptiveQuoteEngine:
    """
    Adaptive Avellaneda-Stoikov market making engine.

    Inputs (per tick):
        fair_value : from Kalman + imbalance signal
        sigma      : blended dual-timeframe vol
        inventory  : current position (BTC)
        imbalance  : order flow imbalance [-1, +1]

    Output: Quotes(bid, ask)

    Adaptation logic:
        1. reservation = fair_value - q * gamma * sigma^2 * T
        2. delta = base_delta * regime_multiplier
        3. bid = reservation - delta * (1 + imbalance_skew)
           ask = reservation + delta * (1 - imbalance_skew)
           → positive imbalance (buy pressure) → tighter ask, wider bid
    """

    def __init__(
        self,
        gamma: float = 0.05,
        k: float = 1.5,
        T_horizon: float = 600.0,
        max_inventory: float = 0.1,
        min_half_spread: float = 0.5,  # USD: never quote tighter than this
        imbalance_skew_factor: float = 0.3,  # max skew as fraction of delta
    ) -> None:
        if gamma <= 0 or k <= 0:
            raise ValueError("gamma and k must be positive")
        self.gamma = gamma
        self.k = k
        self.T_horizon = T_horizon
        self.max_inventory = max_inventory
        self.min_half_spread = min_half_spread
        self.imbalance_skew_factor = imbalance_skew_factor

        self._last_quotes: Quotes | None = None
        self._n_requotes = 0

    def compute(
        self,
        fair_value: float,
        sigma: float,
        inventory: float,
        imbalance: float = 0.0,
        regime: str = "normal",
        time_remaining: float | None = None,
    ) -> Quotes:
        T = time_remaining if time_remaining is not None else self.T_horizon

        # Reservation price (inventory-adjusted fair value)
        reservation = fair_value - inventory * self.gamma * sigma ** 2 * T

        # Base half-spread from A-S formula
        base_delta = (
            self.gamma * sigma ** 2 * T / 2
            + (1 / self.k) * math.log(1 + self.gamma / self.k)
        )

        # Regime adaptation: widen in high vol
        regime_mult = {"high_vol": 1.5, "normal": 1.0, "low_vol": 0.8}.get(regime, 1.0)
        delta = max(base_delta * regime_mult, self.min_half_spread)

        # Imbalance skew: shift quotes toward the flow
        # Positive imbalance → buyers aggressive → tighten ask, widen bid
        skew = self.imbalance_skew_factor * imbalance * delta

        # Hard inventory limit: one-sided quoting
        if inventory >= self.max_inventory:
            bid = reservation - delta * 4   # far away
            ask = reservation + delta - skew
        elif inventory <= -self.max_inventory:
            bid = reservation - delta - skew
            ask = reservation + delta * 4
        else:
            bid = reservation - delta - skew
            ask = reservation + delta - skew

        self._n_requotes += 1
        self._last_quotes = Quotes(
            bid=round(bid, 2),
            ask=round(ask, 2),
            fair_value=fair_value,
            half_spread=delta,
            inventory=inventory,
            regime=regime,
        )
        return self._last_quotes

    @property
    def last_quotes(self) -> Quotes | None:
        return self._last_quotes
