"""
Stress Test & Failure Mode Analysis — does the edge survive when the world breaks?

We deliberately destroy the market conditions to find the breaking points:

  1. FLASH_CRASH      — sudden 2-5% price drop in 1 second; adverse selection spikes
  2. SPREAD_EXPLOSION — bid-ask spreads widen 5-10x; quoted spread captures collapse
  3. ZERO_LIQUIDITY   — book depth drops to near zero; fills stop; inventory builds
  4. LATENCY_SPIKE    — our latency jumps 500ms; we're always last in queue
  5. TOXIC_FLOW       — 80% of flow is informed; markout turns severely negative
  6. INVENTORY_LOCK   — inventory hits max; we can only quote one side
  7. REGIME_SHIFT     — vol doubles overnight; old baseline is wrong

For each scenario we simulate N fills, compute capital efficiency + markout stats,
and produce a SURVIVE / DEGRADE / COLLAPSE verdict.

Usage
-----
    from mm_live.research.stress_test import StressTester, StressScenario

    tester = StressTester(base_fills=normal_fills, max_position_qty=0.1)
    report = tester.run_all()
    tester.print_report(report)
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from mm_live.analytics.capital_efficiency import (
    CapitalEfficiencyReport,
    FillSummary,
    compute_capital_efficiency,
)


class StressScenario(Enum):
    FLASH_CRASH = "flash_crash"
    SPREAD_EXPLOSION = "spread_explosion"
    ZERO_LIQUIDITY = "zero_liquidity"
    LATENCY_SPIKE = "latency_spike"
    TOXIC_FLOW = "toxic_flow"
    INVENTORY_LOCK = "inventory_lock"
    REGIME_SHIFT = "regime_shift"


@dataclass
class StressResult:
    scenario: StressScenario
    n_fills: int
    baseline_roi: float
    stressed_roi: float
    baseline_hit_rate: float
    stressed_hit_rate: float
    baseline_markout: float
    stressed_markout: float
    roi_degradation: float        # (baseline - stressed) / |baseline|; > 0 = worse
    verdict: str                  # "SURVIVE" | "DEGRADE" | "COLLAPSE"
    notes: str


@dataclass
class StressTestReport:
    results: list[StressResult]
    n_survived: int
    n_degraded: int
    n_collapsed: int
    overall_verdict: str          # "ROBUST" | "FRAGILE" | "BROKEN"
    critical_failures: list[str]  # scenario names that collapsed


class StressTester:
    """
    Simulate failure scenarios by perturbing a baseline fill set.

    Parameters
    ----------
    base_fills       : representative fill history under normal conditions
    max_position_qty : max inventory limit (BTC)
    rng_seed         : reproducible results
    """

    # Verdict thresholds
    COLLAPSE_ROI_LOSS  = 0.5   # ROI drops >50% → COLLAPSE
    DEGRADE_ROI_LOSS   = 0.2   # ROI drops >20% → DEGRADE
    COLLAPSE_MARKOUT   = -2.0  # avg markout < -2x half_spread → COLLAPSE

    def __init__(
        self,
        base_fills: Sequence[FillSummary],
        max_position_qty: float = 0.1,
        rng_seed: int = 42,
    ) -> None:
        self._base = list(base_fills)
        self._max_pos = max_position_qty
        self._rng = random.Random(rng_seed)

        if not self._base:
            raise ValueError("Need at least one fill in base_fills.")

        self._baseline = compute_capital_efficiency(
            self._base, self._max_pos, label="baseline"
        )

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def run_all(self) -> StressTestReport:
        results = [self.run_scenario(s) for s in StressScenario]
        n_survived  = sum(1 for r in results if r.verdict == "SURVIVE")
        n_degraded  = sum(1 for r in results if r.verdict == "DEGRADE")
        n_collapsed = sum(1 for r in results if r.verdict == "COLLAPSE")

        if n_collapsed >= 3:
            overall = "BROKEN"
        elif n_collapsed >= 1 or n_degraded >= 4:
            overall = "FRAGILE"
        else:
            overall = "ROBUST"

        critical = [r.scenario.value for r in results if r.verdict == "COLLAPSE"]

        return StressTestReport(
            results=results,
            n_survived=n_survived,
            n_degraded=n_degraded,
            n_collapsed=n_collapsed,
            overall_verdict=overall,
            critical_failures=critical,
        )

    def run_scenario(self, scenario: StressScenario) -> StressResult:
        stressed_fills = self._apply_scenario(scenario)
        stressed = compute_capital_efficiency(
            stressed_fills, self._max_pos, label=scenario.value
        )
        return self._make_result(scenario, stressed)

    # -------------------------------------------------------------------
    # Scenario implementations
    # -------------------------------------------------------------------

    def _apply_scenario(self, scenario: StressScenario) -> list[FillSummary]:
        if scenario == StressScenario.FLASH_CRASH:
            return self._flash_crash()
        elif scenario == StressScenario.SPREAD_EXPLOSION:
            return self._spread_explosion()
        elif scenario == StressScenario.ZERO_LIQUIDITY:
            return self._zero_liquidity()
        elif scenario == StressScenario.LATENCY_SPIKE:
            return self._latency_spike()
        elif scenario == StressScenario.TOXIC_FLOW:
            return self._toxic_flow()
        elif scenario == StressScenario.INVENTORY_LOCK:
            return self._inventory_lock()
        elif scenario == StressScenario.REGIME_SHIFT:
            return self._regime_shift()
        raise ValueError(f"Unknown scenario: {scenario}")

    def _flash_crash(self) -> list[FillSummary]:
        """Price drops 3% mid-run; all buy fills immediately underwater."""
        crash_fraction = 0.03
        fills = []
        crash_point = len(self._base) // 2
        for i, f in enumerate(self._base):
            if i < crash_point:
                fills.append(f)
            else:
                # After crash: fill_price unchanged, but markout reflects crash
                new_markout = f.markout_100ms - f.fill_price * crash_fraction
                new_pnl = f.pnl - f.fill_price * crash_fraction * f.qty
                fills.append(_replace(f, markout_100ms=new_markout, pnl=new_pnl))
        return fills

    def _spread_explosion(self) -> list[FillSummary]:
        """Spreads widen 7x; capture per fill collapses (but still filled)."""
        fills = []
        for f in self._base:
            # Spread widened → fill_price worse by 3x half_spread extra
            extra_cost = f.fill_price * 0.002  # ~20bps extra slippage
            new_markout = f.markout_100ms - extra_cost
            new_pnl = f.pnl - extra_cost * f.qty * 100
            fills.append(_replace(f, markout_100ms=new_markout, pnl=new_pnl))
        return fills

    def _zero_liquidity(self) -> list[FillSummary]:
        """Only 20% of orders get filled; inventory builds; fewer fills."""
        return [
            f for i, f in enumerate(self._base)
            if self._rng.random() < 0.2
        ]

    def _latency_spike(self) -> list[FillSummary]:
        """500ms extra latency → we're always last in queue → worse fills."""
        fills = []
        for f in self._base:
            # Last-in-queue: price moved ~1bps against us before fill
            queue_slippage = f.fill_price * 0.0001
            new_markout = f.markout_100ms - queue_slippage
            new_pnl = f.pnl - queue_slippage * f.qty * 100
            fills.append(_replace(f, markout_100ms=new_markout, pnl=new_pnl))
        return fills

    def _toxic_flow(self) -> list[FillSummary]:
        """80% of flow is informed; markout turns severely negative."""
        fills = []
        for f in self._base:
            if self._rng.random() < 0.8:
                # Informed fill: price moves 2x half_spread against us at 100ms
                adverse = f.fill_price * 0.003
                new_markout = -adverse
                new_pnl = f.pnl - adverse * f.qty * 100
                fills.append(_replace(f, markout_100ms=new_markout, pnl=new_pnl))
            else:
                fills.append(f)
        return fills

    def _inventory_lock(self) -> list[FillSummary]:
        """Inventory at max; can only quote asks; fill rate halved; asymmetric."""
        fills = []
        for f in self._base:
            if f.side == "sell":  # only sell fills possible when long
                fills.append(f)
            # buy fills suppressed (inventory full)
        return fills if fills else self._base[:1]  # ensure at least 1 fill

    def _regime_shift(self) -> list[FillSummary]:
        """Vol doubles; old baseline wrong; spreads need to be 2x wider."""
        fills = []
        for f in self._base:
            # Higher vol → markout variance doubles; mean shifts negative
            vol_factor = 2.0
            extra_adverse = f.fill_price * 0.0015 * (vol_factor - 1)
            new_markout = f.markout_100ms - extra_adverse
            new_pnl = f.pnl - extra_adverse * f.qty * 50
            fills.append(_replace(f, markout_100ms=new_markout, pnl=new_pnl, regime="high_vol"))
        return fills

    # -------------------------------------------------------------------
    # Verdict logic
    # -------------------------------------------------------------------

    def _make_result(self, scenario: StressScenario, stressed: CapitalEfficiencyReport) -> StressResult:
        b = self._baseline

        # ROI degradation
        if abs(b.roi) > 1e-9:
            roi_deg = (b.roi - stressed.roi) / abs(b.roi)
        else:
            roi_deg = 0.0

        # Verdict
        avg_half_spread = sum(f.fill_price * 0.0005 for f in self._base) / len(self._base)
        markout_ratio = abs(stressed.avg_markout_100ms) / avg_half_spread if avg_half_spread > 0 else 0

        if roi_deg >= self.COLLAPSE_ROI_LOSS or markout_ratio >= abs(self.COLLAPSE_MARKOUT):
            verdict = "COLLAPSE"
            notes = f"ROI loss={roi_deg:.0%}, AS ratio={markout_ratio:.2f}"
        elif roi_deg >= self.DEGRADE_ROI_LOSS:
            verdict = "DEGRADE"
            notes = f"ROI loss={roi_deg:.0%}"
        else:
            verdict = "SURVIVE"
            notes = f"ROI loss={roi_deg:.0%}"

        return StressResult(
            scenario=scenario,
            n_fills=stressed.n_fills,
            baseline_roi=round(b.roi, 6),
            stressed_roi=round(stressed.roi, 6),
            baseline_hit_rate=round(b.hit_rate, 4),
            stressed_hit_rate=round(stressed.hit_rate, 4),
            baseline_markout=round(b.avg_markout_100ms, 6),
            stressed_markout=round(stressed.avg_markout_100ms, 6),
            roi_degradation=round(roi_deg, 4),
            verdict=verdict,
            notes=notes,
        )

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------

    def print_report(self, report: StressTestReport) -> None:
        width = 90
        print(f"\n{'=' * width}")
        print("  STRESS TEST & FAILURE MODE ANALYSIS".center(width))
        print(f"{'=' * width}")
        print(f"  {'Scenario':<22} {'Fills':>6} {'BaseROI':>8} {'StressROI':>10} "
              f"{'HitRate':>8} {'Mkout@100':>10} {'ROI-Loss':>9}  Verdict")
        print(f"  {'-' * 88}")

        for r in report.results:
            flag = "  <-- !!" if r.verdict == "COLLAPSE" else ""
            print(
                f"  {r.scenario.value:<22} {r.n_fills:>6d} "
                f"{r.baseline_roi:>8.4%} {r.stressed_roi:>10.4%} "
                f"{r.stressed_hit_rate:>8.1%} {r.stressed_markout:>+10.5f} "
                f"{r.roi_degradation:>9.1%}  {r.verdict}{flag}"
            )

        print(f"  {'-' * 88}")
        print(f"  Survived: {report.n_survived}  Degraded: {report.n_degraded}  "
              f"Collapsed: {report.n_collapsed}")
        print(f"  OVERALL: {report.overall_verdict}")
        if report.critical_failures:
            print(f"  Critical failures: {', '.join(report.critical_failures)}")
        print(f"{'=' * width}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _replace(f: FillSummary, **kwargs) -> FillSummary:
    """Return a copy of FillSummary with fields overridden."""
    return FillSummary(
        timestamp_ms=f.timestamp_ms,
        side=f.side,
        fill_price=f.fill_price,
        qty=f.qty,
        pnl=kwargs.get("pnl", f.pnl),
        markout_100ms=kwargs.get("markout_100ms", f.markout_100ms),
        regime=kwargs.get("regime", f.regime),
        asset=f.asset,
    )
