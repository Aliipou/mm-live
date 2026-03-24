"""Tests for multi_asset validation module."""
from __future__ import annotations

import pytest

from mm_live.analytics.capital_efficiency import FillSummary
from mm_live.research.multi_asset import (
    AssetSlice,
    MultiAssetValidator,
    MultiAssetValidationReport,
)


def _fill(pnl: float = 1.0, markout: float = 0.001,
          regime: str = "normal", asset: str = "BTCUSDT",
          price: float = 50000.0) -> FillSummary:
    return FillSummary(
        timestamp_ms=1_700_000_000_000,
        side="buy",
        fill_price=price,
        qty=0.01,
        pnl=pnl,
        markout_100ms=markout,
        regime=regime,
        asset=asset,
    )


def _good_slice(asset: str, regime: str, n: int = 20) -> AssetSlice:
    return AssetSlice(
        asset=asset, regime=regime,
        fills=[_fill(pnl=0.5, markout=0.002, asset=asset, regime=regime) for _ in range(n)]
    )


def _bad_slice(asset: str, regime: str, n: int = 20) -> AssetSlice:
    return AssetSlice(
        asset=asset, regime=regime,
        fills=[_fill(pnl=-0.5, markout=-0.005, asset=asset, regime=regime) for _ in range(n)]
    )


class TestMultiAssetValidator:
    def test_all_good_generalises(self):
        slices = [
            _good_slice("BTCUSDT", "normal"),
            _good_slice("BTCUSDT", "high_vol"),
            _good_slice("ETHUSDT", "normal"),
        ]
        v = MultiAssetValidator(slices, max_position_qty=0.1)
        report = v.validate()
        assert report.generalises is True

    def test_one_bad_does_not_generalise(self):
        slices = [
            _good_slice("BTCUSDT", "normal"),
            _bad_slice("BTCUSDT", "high_vol"),
            _good_slice("ETHUSDT", "normal"),
        ]
        v = MultiAssetValidator(slices, max_position_qty=0.1)
        report = v.validate()
        assert report.generalises is False

    def test_empty_slices_returns_false(self):
        v = MultiAssetValidator([], max_position_qty=0.1)
        report = v.validate()
        assert report.generalises is False
        assert report.slices == []

    def test_too_few_fills_excluded(self):
        slices = [
            AssetSlice("BTCUSDT", "normal",
                       fills=[_fill() for _ in range(5)])  # < min_fills=10
        ]
        v = MultiAssetValidator(slices, max_position_qty=0.1, min_fills=10)
        report = v.validate()
        assert len(report.slices) == 0

    def test_weakest_and_strongest(self):
        slices = [
            _good_slice("BTCUSDT", "normal"),
            _bad_slice("ETHUSDT", "high_vol"),
        ]
        v = MultiAssetValidator(slices, max_position_qty=0.1)
        report = v.validate()
        assert report.weakest_slice is not None
        assert report.strongest_slice is not None
        assert report.weakest_slice.hit_rate <= report.strongest_slice.hit_rate

    def test_consistency_metrics_are_floats(self):
        slices = [_good_slice("BTCUSDT", "normal"), _good_slice("ETHUSDT", "normal")]
        v = MultiAssetValidator(slices, max_position_qty=0.1)
        report = v.validate()
        assert isinstance(report.cross_asset_consistency, float)
        assert isinstance(report.cross_regime_consistency, float)

    def test_slice_results_count(self):
        slices = [
            _good_slice("BTCUSDT", "normal"),
            _good_slice("BTCUSDT", "high_vol"),
            _good_slice("ETHUSDT", "normal"),
        ]
        v = MultiAssetValidator(slices, max_position_qty=0.1)
        report = v.validate()
        assert len(report.slices) == 3

    def test_print_report_runs(self, capsys):
        slices = [_good_slice("BTCUSDT", "normal")]
        v = MultiAssetValidator(slices, max_position_qty=0.1)
        report = v.validate()
        v.print_report(report)
        out = capsys.readouterr().out
        assert "Multi-Asset" in out
