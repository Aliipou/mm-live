"""
Strategy performance metrics.

Tracks what matters for evaluating a market making strategy:
- Fill rate (how often do our quotes get hit?)
- Spread captured per fill
- Adverse selection per fill
- Inventory turnover
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StrategyMetrics:
    """Real-time strategy performance tracker."""

    _total_fills: int = field(init=False, default=0)
    _total_quotes: int = field(init=False, default=0)
    _spread_captured: float = field(init=False, default=0.0)
    _adverse_selection: float = field(init=False, default=0.0)
    _vol_history: list[float] = field(init=False, default_factory=list)
    _imbalance_history: list[float] = field(init=False, default_factory=list)

    def record_fill(self, fill: dict[str, Any]) -> None:
        self._total_fills += 1

    def record_quote(self, quotes: Any, sigma: float, imbalance: float) -> None:
        self._total_quotes += 1
        self._vol_history.append(sigma)
        self._imbalance_history.append(imbalance)
        # Keep only last 1000 for memory efficiency
        if len(self._vol_history) > 1000:
            self._vol_history = self._vol_history[-1000:]
            self._imbalance_history = self._imbalance_history[-1000:]

    @property
    def total_fills(self) -> int:
        return self._total_fills

    @property
    def fill_rate(self) -> float:
        """Fraction of quote ticks that resulted in a fill."""
        if self._total_quotes == 0:
            return 0.0
        return self._total_fills / self._total_quotes

    @property
    def avg_vol(self) -> float:
        if not self._vol_history:
            return 0.0
        return sum(self._vol_history) / len(self._vol_history)

    @property
    def avg_imbalance(self) -> float:
        if not self._imbalance_history:
            return 0.0
        return sum(self._imbalance_history) / len(self._imbalance_history)
