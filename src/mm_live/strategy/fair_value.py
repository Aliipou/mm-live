"""
Kalman filter for real-time fair value estimation.

The observed mid-price contains microstructure noise (bid-ask bounce,
quote stuffing, rounding). The Kalman filter separates the latent
fair value from this noise.

State model:
    fair_value[t] = fair_value[t-1] + w_t    (random walk, w ~ N(0, Q))
    mid_price[t]  = fair_value[t]  + v_t    (observation noise, v ~ N(0, R))

Tuning:
    Q (process noise): how fast can fair value move between ticks?
        For BTC/USDT at 100ms updates: Q ~ (0.5 bps)^2
    R (measurement noise): how noisy is the mid relative to fair value?
        Typically R ~ (0.5 * spread)^2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class KalmanFairValue:
    """
    Online Kalman filter for mid-price → fair value.

    Parameters
    ----------
    process_noise_var:
        Q: variance of fair value's random walk per tick.
    measurement_noise_var:
        R: variance of mid-price observation noise.
    """

    process_noise_var: float = 0.01       # Q: ~0.1 USD per 100ms tick
    measurement_noise_var: float = 0.25   # R: ~0.5 USD mid noise

    _x: float = field(init=False, default=0.0)   # fair value estimate
    _p: float = field(init=False, default=1.0)   # estimate uncertainty
    _initialized: bool = field(init=False, default=False)

    def update(self, mid: float) -> float:
        """
        Update filter with new mid-price observation.

        Returns current fair value estimate.
        """
        if not self._initialized:
            self._x = mid
            self._p = self.measurement_noise_var
            self._initialized = True
            return self._x

        # Predict
        p_pred = self._p + self.process_noise_var

        # Kalman gain
        K = p_pred / (p_pred + self.measurement_noise_var)

        # Update
        self._x = self._x + K * (mid - self._x)
        self._p = (1 - K) * p_pred

        return self._x

    @property
    def fair_value(self) -> float:
        return self._x

    @property
    def uncertainty(self) -> float:
        """Current estimate standard deviation."""
        return math.sqrt(self._p)

    @property
    def is_initialized(self) -> bool:
        return self._initialized
