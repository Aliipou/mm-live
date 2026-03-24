"""
Volatility Clustering Signal.

Volatility clusters: high vol now → likely high vol next period.
This is the core GARCH intuition, implemented without GARCH fitting.

We use a simple but robust approach:
    vol_forecast = α · |return_now| + (1-α) · vol_ema

If vol_forecast > vol_ema * threshold → expect continued elevated vol.
This tells the strategy: widen spreads NOW, don't wait for vol to update.

Also detects vol regime transitions:
    EXPANDING: vol_short rising faster than vol_long
    CONTRACTING: vol_short falling toward vol_long
    STABLE: ratio near 1.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class VolTransition(Enum):
    EXPANDING = "expanding"
    CONTRACTING = "contracting"
    STABLE = "stable"


@dataclass
class VolClusteringSignal:
    """
    GARCH-inspired volatility clustering detector.

    Parameters
    ----------
    alpha:
        EMA weight applied to ``|return_now|`` when updating the vol forecast.
        Higher alpha → faster reaction to fresh vol shocks.
    expand_threshold:
        Ratio ``vol_short / vol_long`` above which vol is classified as
        EXPANDING.  Mirror threshold ``1 / expand_threshold`` is used for
        CONTRACTING.
    """

    alpha: float = 0.1
    expand_threshold: float = 1.2

    # Slow EMA — baseline vol level (GARCH long-run component)
    _vol_long: float = field(init=False, default=0.0)
    # Fast EMA — reacts quickly to recent |returns|
    _vol_short: float = field(init=False, default=0.0)
    # Blended one-step-ahead forecast
    _vol_forecast: float = field(init=False, default=0.0)

    _last_price: float | None = field(init=False, default=None)
    _n: int = field(init=False, default=0)

    # Slow EMA alpha: half-life ≈ 100 ticks → alpha ≈ 1 - exp(-ln2/100) ≈ 0.007
    _ALPHA_LONG: float = field(init=False, default=0.007)

    def update(self, price: float) -> float:
        """
        Ingest a new price tick and return the vol forecast for the next period.

        On the first call (no previous price) returns 0.0.
        """
        if self._last_price is None or self._last_price <= 0.0:
            self._last_price = price
            return self._vol_forecast

        log_ret = math.log(price / self._last_price)
        abs_ret = abs(log_ret)
        self._last_price = price

        if self._n == 0:
            # Seed both EMAs with the first observation
            self._vol_short = abs_ret
            self._vol_long = abs_ret
            self._vol_forecast = abs_ret
        else:
            self._vol_short = (
                (1.0 - self.alpha) * self._vol_short + self.alpha * abs_ret
            )
            self._vol_long = (
                (1.0 - self._ALPHA_LONG) * self._vol_long
                + self._ALPHA_LONG * abs_ret
            )
            self._vol_forecast = (
                self.alpha * abs_ret + (1.0 - self.alpha) * self._vol_short
            )

        self._n += 1
        return self._vol_forecast

    @property
    def transition(self) -> VolTransition:
        """
        Current vol regime transition inferred from the short/long EMA ratio.

        EXPANDING  : short rising faster than long (vol_short / vol_long > threshold)
        CONTRACTING: vol_short well below vol_long (ratio < 1 / threshold)
        STABLE     : ratio near 1.0
        """
        if self._vol_long == 0.0:
            return VolTransition.STABLE

        ratio = self._vol_short / self._vol_long
        if ratio > self.expand_threshold:
            return VolTransition.EXPANDING
        if ratio < 1.0 / self.expand_threshold:
            return VolTransition.CONTRACTING
        return VolTransition.STABLE

    @property
    def urgency(self) -> float:
        """
        Spread-widening urgency in [0.0, 1.0].

        0.0 → stable or contracting vol; no rush to widen.
        1.0 → rapidly expanding vol; widen spreads immediately.

        Computed as a sigmoid-like scaling of how much the vol_short/vol_long
        ratio exceeds the expand_threshold.
        """
        if self._vol_long == 0.0:
            return 0.0

        ratio = self._vol_short / self._vol_long
        if ratio <= 1.0:
            return 0.0

        # Linearly scale: at ratio == expand_threshold → 0.5; saturates at 2x threshold
        excess = (ratio - 1.0) / (self.expand_threshold - 1.0 + 1e-10)
        return min(excess / 2.0, 1.0)
