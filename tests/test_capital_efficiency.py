"""Tests for capital_efficiency module."""
from __future__ import annotations

import pytest

from mm_live.analytics.capital_efficiency import (
    FillSummary,
    compute_by_asset,
    compute_by_regime,
    compute_capital_efficiency,
)


def _fill(pnl: float, markout: float = 0.001, regime: str = "normal",
          asset: str = "BTCUSDT", price: float = 50000.0, qty: float = 0.01,
          ts: int = 1_700_000_000_000) -> FillSummary:
    return FillSummary(
        timestamp_ms=ts,
        side="buy",
        fill_price=price,
        qty=qty,
        pnl=pnl,
        markout_100ms=markout,
        regime=regime,
        asset=asset,
    )


class TestComputeCapitalEfficiency:
    def test_empty_fills_returns_zero_roi(self):
        r = compute_capital_efficiency([], max_position_qty=0.1)
        assert r.roi == 0.0
        assert r.n_fills == 0

    def test_positive_pnl(self):
        fills = [_fill(pnl=1.0) for _ in range(10)]
        r = compute_capital_efficiency(fills, max_position_qty=0.1)
        assert r.total_pnl == pytest.approx(10.0)
        assert r.roi > 0

    def test_negative_pnl(self):
        fills = [_fill(pnl=-0.5) for _ in range(10)]
        r = compute_capital_efficiency(fills, max_position_qty=0.1)
        assert r.roi < 0

    def test_hit_rate_all_positive(self):
        fills = [_fill(pnl=1.0, markout=0.001) for _ in range(20)]
        r = compute_capital_efficiency(fills, max_position_qty=0.1)
        assert r.hit_rate == pytest.approx(1.0)

    def test_hit_rate_all_negative(self):
        fills = [_fill(pnl=-1.0, markout=-0.001) for _ in range(20)]
        r = compute_capital_efficiency(fills, max_position_qty=0.1)
        assert r.hit_rate == pytest.approx(0.0)

    def test_sharpe_positive_for_consistent_pnl(self):
        # Different days with varying positive PnL → positive Sharpe
        fills = [
            _fill(pnl=1.0 + (i % 3) * 0.1, ts=1_700_000_000_000 + i * 86_400_000)
            for i in range(30)
        ]
        r = compute_capital_efficiency(fills, max_position_qty=0.1)
        assert r.sharpe_annualised > 0

    def test_capital_turnover_positive(self):
        fills = [_fill(pnl=0.1) for _ in range(5)]
        r = compute_capital_efficiency(fills, max_position_qty=0.1)
        assert r.capital_turnover > 0
        assert r.total_volume > 0

    def test_report_label(self):
        fills = [_fill(pnl=1.0)]
        r = compute_capital_efficiency(fills, max_position_qty=0.1, label="MY_LABEL")
        assert r.label == "MY_LABEL"


class TestComputeByRegime:
    def test_splits_by_regime(self):
        fills = (
            [_fill(pnl=1.0, regime="normal")] * 5
            + [_fill(pnl=-0.5, regime="high_vol")] * 5
        )
        result = compute_by_regime(fills, max_position_qty=0.1)
        assert "normal" in result
        assert "high_vol" in result
        assert result["normal"].total_pnl > result["high_vol"].total_pnl

    def test_single_regime(self):
        fills = [_fill(pnl=1.0, regime="low_vol")] * 3
        result = compute_by_regime(fills, max_position_qty=0.1)
        assert set(result.keys()) == {"low_vol"}


class TestComputeByAsset:
    def test_splits_by_asset(self):
        fills = (
            [_fill(pnl=2.0, asset="BTCUSDT")] * 5
            + [_fill(pnl=0.5, asset="ETHUSDT", price=3000.0)] * 5
        )
        result = compute_by_asset(fills, max_position_qty=0.1)
        assert "BTCUSDT" in result
        assert "ETHUSDT" in result

    def test_separate_pnl(self):
        fills = (
            [_fill(pnl=1.0, asset="BTCUSDT")] * 4
            + [_fill(pnl=3.0, asset="ETHUSDT", price=3000.0)] * 4
        )
        result = compute_by_asset(fills, max_position_qty=0.1)
        assert result["ETHUSDT"].total_pnl > result["BTCUSDT"].total_pnl
