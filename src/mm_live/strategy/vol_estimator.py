"""
Exponentially-weighted realized volatility estimator.

Used by the Avellaneda-Stoikov model to compute sigma (per-tick vol).
The EWMA decays old observations so recent market conditions dominate.

At 100ms tick rate:
    halflife=50  → ~5 second memory
    halflife=300 → ~30 second memory

Convert tick vol to per-second vol:
    sigma_per_second = sigma_per_tick * sqrt(ticks_per_second)
    At 100ms ticks: ticks_per_second = 10
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RealizedVol:
    """
    EWMA realized volatility from log-returns.

    Parameters
    ----------
    halflife:
        Memory half-life in number of ticks.
    initial_vol:
        Seed volatility (USD per tick) before data arrives.
    """

    halflife: float = 100.0       # ticks (~10 seconds at 100ms)
    initial_vol: float = 1.0      # ~1 USD per tick initial guess for BTC

    _var: float = field(init=False)
    _last_price: float | None = field(init=False, default=None)
    _n: int = field(init=False, default=0)
    _alpha: float = field(init=False)

    def __post_init__(self) -> None:
        self._var = self.initial_vol ** 2
        self._alpha = 1 - math.exp(-math.log(2) / self.halflife)

    def update(self, price: float) -> float:
        """
        Update with new price. Returns current vol estimate (USD per tick).
        """
        if self._last_price is not None and self._last_price > 0:
            log_ret = math.log(price / self._last_price)
            self._var = (1 - self._alpha) * self._var + self._alpha * log_ret ** 2
            self._n += 1
        self._last_price = price
        return self.sigma

    @property
    def sigma(self) -> float:
        """Current volatility estimate (USD per tick)."""
        return math.sqrt(self._var)

    @property
    def sigma_annualized(self) -> float:
        """
        Annualized vol (for reference).
        100ms ticks → ~10 ticks/sec → ~3.15M ticks/year
        """
        ticks_per_year = 365 * 24 * 3600 * 10  # at 100ms
        return self.sigma * math.sqrt(ticks_per_year)

    @property
    def n_observations(self) -> int:
        return self._n
