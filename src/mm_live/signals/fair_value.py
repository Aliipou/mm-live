"""
Fair value signal combining Kalman filter + order flow imbalance.

Fair value = Kalman(mid_price) + alpha * imbalance

This is better than a plain Kalman filter because:
- Kalman removes microstructure noise from mid price
- Imbalance adds short-term directional information
- Combined estimate has lower adverse selection vs. either alone
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class FairValueSignal:
    """
    Kalman fair value enhanced with order flow imbalance.

    Parameters
    ----------
    process_noise_var: Q — how fast fair value can move per tick.
    measurement_noise_var: R — mid-price noise variance.
    imbalance_alpha: weight on imbalance adjustment (USD per unit).
    """

    process_noise_var: float = 0.01
    measurement_noise_var: float = 0.25
    imbalance_alpha: float = 2.0

    _x: float = field(init=False, default=0.0)
    _p: float = field(init=False, default=1.0)
    _initialized: bool = field(init=False, default=False)

    def update(self, mid: float, imbalance: float = 0.0) -> float:
        """
        Update Kalman filter and apply imbalance adjustment.

        Parameters
        ----------
        mid: Current mid-price from order book.
        imbalance: Order flow imbalance in [-1, +1].

        Returns
        -------
        Fair value estimate (Kalman output + imbalance shift).
        """
        if not self._initialized:
            self._x = mid
            self._p = self.measurement_noise_var
            self._initialized = True
            return self._x + self.imbalance_alpha * imbalance

        # Kalman predict
        p_pred = self._p + self.process_noise_var

        # Kalman gain
        K = p_pred / (p_pred + self.measurement_noise_var)

        # Kalman update
        self._x = self._x + K * (mid - self._x)
        self._p = (1 - K) * p_pred

        # Add imbalance adjustment
        return self._x + self.imbalance_alpha * imbalance

    @property
    def kalman_estimate(self) -> float:
        """Raw Kalman estimate without imbalance."""
        return self._x

    @property
    def uncertainty(self) -> float:
        return math.sqrt(self._p)

    @property
    def is_initialized(self) -> bool:
        return self._initialized
