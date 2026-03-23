"""
P&L tracker for paper trading.

Tracks:
- Realized P&L: from completed fills (cash in - cash out)
- Unrealized P&L: current inventory * fair_value
- Total P&L: realized + unrealized

P&L attribution:
- Spread capture: how much came from bid-ask spread
- Adverse selection: how much we lost to informed flow
  (measured as: fill_price - fair_value at fill time)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PnLTracker:
    """
    Tracks mark-to-market P&L for a market maker.

    Cash accounting:
        Buy  → cash decreases by price*qty
        Sell → cash increases by price*qty
        Inventory = cumulative net qty bought
        Unrealized = inventory * current_fair_value
    """

    _cash: float = field(init=False, default=0.0)
    _inventory: float = field(init=False, default=0.0)
    _n_fills: int = field(init=False, default=0)
    _spread_capture: float = field(init=False, default=0.0)
    _adverse_selection: float = field(init=False, default=0.0)

    def record_fill(
        self,
        side: str,
        price: float,
        qty: float,
        fair_value: float,
    ) -> None:
        """
        Record a fill and update cash, inventory, and attribution.

        Parameters
        ----------
        side: "buy" (we bought) or "sell" (we sold)
        price: fill price
        qty: fill quantity (positive)
        fair_value: fair value at time of fill (for attribution)
        """
        if side == "buy":
            self._cash -= price * qty
            self._inventory += qty
            # Attribution: if we bought below fair value, that's good
            edge = fair_value - price
        else:
            self._cash += price * qty
            self._inventory -= qty
            edge = price - fair_value

        # Positive edge = spread capture; negative = adverse selection
        if edge >= 0:
            self._spread_capture += edge * qty
        else:
            self._adverse_selection += edge * qty  # negative

        self._n_fills += 1

    def unrealized_pnl(self, current_fair_value: float) -> float:
        """Mark inventory to fair value."""
        return self._inventory * current_fair_value

    @property
    def realized_pnl(self) -> float:
        return self._cash

    @property
    def inventory(self) -> float:
        return self._inventory

    @property
    def n_fills(self) -> int:
        return self._n_fills

    @property
    def spread_capture(self) -> float:
        return self._spread_capture

    @property
    def adverse_selection_cost(self) -> float:
        return self._adverse_selection

    def total(self, current_fair_value: float = 0.0) -> float:
        return self._cash + self.unrealized_pnl(current_fair_value)
