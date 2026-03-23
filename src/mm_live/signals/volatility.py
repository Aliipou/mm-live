"""
Dual-timeframe volatility estimator.

Uses two EWMA estimators at different timescales:
  short: captures microstructure vol (fast regime changes)
  long:  captures macro vol (trend, intraday structure)

Blended sigma = w_short * sigma_short + w_long * sigma_long

Why dual-timeframe:
  - Single EWMA is either too fast (noisy) or too slow (stale)
  - During high vol events (news), short_vol spikes fast → wider spreads immediately
  - During calm periods, long_vol dominates → tighter spreads, more fills
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class _EWMA:
    halflife: float
    initial_vol: float
    _var: float = field(init=False)
    _last: float | None = field(init=False, default=None)
    _alpha: float = field(init=False)

    def __post_init__(self) -> None:
        self._var = self.initial_vol ** 2
        self._alpha = 1 - math.exp(-math.log(2) / self.halflife)

    def update(self, price: float) -> float:
        if self._last is not None and self._last > 0:
            log_ret = math.log(price / self._last)
            self._var = (1 - self._alpha) * self._var + self._alpha * log_ret ** 2
        self._last = price
        return math.sqrt(self._var)


@dataclass
class DualVolatility:
    """
    Blended short + long EWMA volatility.

    Parameters
    ----------
    short_halflife: Half-life of fast estimator in ticks.
    long_halflife:  Half-life of slow estimator in ticks.
    short_weight:   Weight on short-term vol (0 to 1). Long = 1 - short_weight.
    initial_vol:    Seed volatility (USD per tick).
    """

    short_halflife: float = 50.0
    long_halflife: float = 300.0
    short_weight: float = 0.7
    initial_vol: float = 1.0

    _short: _EWMA = field(init=False)
    _long: _EWMA = field(init=False)

    def __post_init__(self) -> None:
        self._short = _EWMA(halflife=self.short_halflife, initial_vol=self.initial_vol)
        self._long = _EWMA(halflife=self.long_halflife, initial_vol=self.initial_vol)

    def update(self, price: float) -> float:
        """Update both estimators. Returns blended sigma."""
        s = self._short.update(price)
        l = self._long.update(price)
        return self.short_weight * s + (1 - self.short_weight) * l

    @property
    def short_vol(self) -> float:
        return math.sqrt(self._short._var)

    @property
    def long_vol(self) -> float:
        return math.sqrt(self._long._var)

    @property
    def regime(self) -> str:
        """Simple regime detection: 'high' if short >> long."""
        ratio = self.short_vol / (self.long_vol + 1e-10)
        if ratio > 1.5:
            return "high_vol"
        elif ratio < 0.7:
            return "low_vol"
        return "normal"
