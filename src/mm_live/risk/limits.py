"""
Risk limits and circuit breakers.

Before sending any quote to the exchange, every order passes through
RiskLimits. If any limit is breached, quoting stops immediately.

Limits enforced:
1. Max inventory (position size)
2. Max drawdown from peak P&L
3. Max spread (sanity check — don't post absurd quotes)
4. Min spread (don't post inside the exchange's minimum tick)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RiskStatus(Enum):
    OK = "ok"
    INVENTORY_LIMIT = "inventory_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    SPREAD_TOO_WIDE = "spread_too_wide"
    SPREAD_TOO_TIGHT = "spread_too_tight"
    NOT_INITIALIZED = "not_initialized"


@dataclass
class RiskLimits:
    """
    Pre-trade risk checks for live market making.

    Parameters
    ----------
    max_inventory_btc:
        Maximum absolute BTC position. Quoting halts if breached.
    max_drawdown_usd:
        Maximum loss from peak P&L (USD). Circuit breaker.
    max_spread_usd:
        Sanity check: don't post spreads wider than this.
    min_spread_usd:
        Don't post spreads narrower than Binance's minimum tick ($0.01).
    """

    max_inventory_btc: float = 0.1
    max_drawdown_usd: float = 50.0
    max_spread_usd: float = 100.0
    min_spread_usd: float = 0.01

    _peak_pnl: float = field(init=False, default=0.0)
    _current_pnl: float = field(init=False, default=0.0)
    _initialized: bool = field(init=False, default=False)

    def check(
        self,
        inventory: float,
        bid: float,
        ask: float,
        current_pnl: float,
    ) -> RiskStatus:
        """
        Run all pre-trade checks.

        Returns RiskStatus.OK if safe to quote, otherwise the breached limit.
        """
        self._current_pnl = current_pnl
        self._peak_pnl = max(self._peak_pnl, current_pnl)
        self._initialized = True

        if abs(inventory) >= self.max_inventory_btc:
            return RiskStatus.INVENTORY_LIMIT

        drawdown = self._peak_pnl - current_pnl
        if drawdown >= self.max_drawdown_usd:
            return RiskStatus.DRAWDOWN_LIMIT

        spread = ask - bid
        if spread > self.max_spread_usd:
            return RiskStatus.SPREAD_TOO_WIDE

        if spread < self.min_spread_usd:
            return RiskStatus.SPREAD_TOO_TIGHT

        return RiskStatus.OK

    @property
    def drawdown(self) -> float:
        return self._peak_pnl - self._current_pnl

    @property
    def is_blown(self) -> bool:
        """True if drawdown limit is permanently hit."""
        return self.drawdown >= self.max_drawdown_usd
