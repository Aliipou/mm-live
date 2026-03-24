"""Tests for stress_test module."""
from __future__ import annotations

import pytest

from mm_live.analytics.capital_efficiency import FillSummary
from mm_live.research.stress_test import (
    StressResult,
    StressScenario,
    StressTester,
    StressTestReport,
)


def _fill(pnl: float = 0.5, markout: float = 0.002,
          side: str = "buy", price: float = 50000.0) -> FillSummary:
    return FillSummary(
        timestamp_ms=1_700_000_000_000,
        side=side,
        fill_price=price,
        qty=0.01,
        pnl=pnl,
        markout_100ms=markout,
        regime="normal",
        asset="BTCUSDT",
    )


@pytest.fixture
def tester():
    fills = [_fill(pnl=0.5, markout=0.002) for _ in range(50)]
    return StressTester(fills, max_position_qty=0.1, rng_seed=42)


class TestStressTester:
    def test_run_all_returns_report(self, tester):
        report = tester.run_all()
        assert isinstance(report, StressTestReport)

    def test_run_all_has_all_scenarios(self, tester):
        report = tester.run_all()
        scenarios_tested = {r.scenario for r in report.results}
        assert scenarios_tested == set(StressScenario)

    def test_overall_verdict_valid(self, tester):
        report = tester.run_all()
        assert report.overall_verdict in ("ROBUST", "FRAGILE", "BROKEN")

    def test_verdict_counts_sum(self, tester):
        report = tester.run_all()
        total = report.n_survived + report.n_degraded + report.n_collapsed
        assert total == len(StressScenario)

    def test_single_scenario(self, tester):
        result = tester.run_scenario(StressScenario.FLASH_CRASH)
        assert isinstance(result, StressResult)
        assert result.scenario == StressScenario.FLASH_CRASH
        assert result.verdict in ("SURVIVE", "DEGRADE", "COLLAPSE")

    def test_flash_crash_worsens_markout(self, tester):
        result = tester.run_scenario(StressScenario.FLASH_CRASH)
        assert result.stressed_markout < result.baseline_markout

    def test_toxic_flow_worsens_hit_rate(self, tester):
        result = tester.run_scenario(StressScenario.TOXIC_FLOW)
        assert result.stressed_hit_rate <= result.baseline_hit_rate

    def test_zero_liquidity_reduces_fills(self, tester):
        result = tester.run_scenario(StressScenario.ZERO_LIQUIDITY)
        assert result.n_fills < 50  # should have fewer fills than baseline

    def test_empty_fills_raises(self):
        with pytest.raises(ValueError, match="at least one fill"):
            StressTester([], max_position_qty=0.1)

    def test_print_report_runs(self, tester, capsys):
        report = tester.run_all()
        tester.print_report(report)
        out = capsys.readouterr().out
        assert "STRESS TEST" in out
        assert "OVERALL" in out

    def test_roi_degradation_flash_crash(self, tester):
        result = tester.run_scenario(StressScenario.FLASH_CRASH)
        assert result.roi_degradation >= 0  # should be worse or same

    def test_inventory_lock_only_sell_fills(self):
        fills = (
            [_fill(side="buy")] * 30
            + [_fill(side="sell")] * 20
        )
        t = StressTester(fills, max_position_qty=0.1)
        result = t.run_scenario(StressScenario.INVENTORY_LOCK)
        assert result.n_fills <= 20  # only sell fills remain
