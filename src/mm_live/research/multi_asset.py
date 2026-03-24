"""
Multi-Asset Validation — test whether edge generalises across BTC, ETH, and regimes.

The core question: does your signal work because of genuine microstructure alpha,
or because you overfit to BTC's specific patterns in a specific period?

A real edge:
  - Works on ETH (different liquidity profile)
  - Works in both high-vol and low-vol regimes
  - Hit rate and markout ratio remain positive across all conditions

A fragile edge:
  - Only works on BTC
  - Collapses in high-vol (adverse selection dominates)
  - Hit rate drops below 50% when regime changes

Usage
-----
    from mm_live.research.multi_asset import MultiAssetValidator, AssetSlice

    slices = [
        AssetSlice(asset="BTCUSDT", regime="normal",   fills=[...]),
        AssetSlice(asset="BTCUSDT", regime="high_vol", fills=[...]),
        AssetSlice(asset="ETHUSDT", regime="normal",   fills=[...]),
    ]
    validator = MultiAssetValidator(slices)
    report = validator.validate()
    validator.print_report(report)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from mm_live.analytics.capital_efficiency import (
    CapitalEfficiencyReport,
    FillSummary,
    compute_capital_efficiency,
)


@dataclass
class AssetSlice:
    """One (asset, regime) bucket of fills."""
    asset: str
    regime: str
    fills: list[FillSummary]


@dataclass
class SliceResult:
    asset: str
    regime: str
    n_fills: int
    hit_rate: float
    avg_markout_100ms: float
    roi: float
    sharpe: float
    verdict: str   # "PASS" | "WARN" | "FAIL"


@dataclass
class MultiAssetValidationReport:
    slices: list[SliceResult]
    generalises: bool         # True if edge holds across ALL slices
    weakest_slice: SliceResult | None
    strongest_slice: SliceResult | None
    cross_asset_consistency: float   # std(hit_rate) across assets; lower = more consistent
    cross_regime_consistency: float  # std(hit_rate) across regimes


class MultiAssetValidator:
    """
    Validate edge consistency across assets and regimes.

    Parameters
    ----------
    slices          : list of AssetSlice, one per (asset, regime) combination
    max_position_qty: max inventory per asset (for capital calculations)
    min_fills       : minimum fills per slice to include in analysis
    """

    PASS_HIT_RATE = 0.52      # above 52% hit rate = edge present
    WARN_HIT_RATE = 0.48      # 48-52% = marginal
    PASS_MARKOUT  = 0.0       # positive avg markout = no adverse selection
    WARN_MARKOUT  = -0.5      # small negative = marginal

    def __init__(
        self,
        slices: Sequence[AssetSlice],
        max_position_qty: float = 0.1,
        min_fills: int = 10,
    ) -> None:
        self._slices = list(slices)
        self._max_pos = max_position_qty
        self._min_fills = min_fills

    def validate(self) -> MultiAssetValidationReport:
        results: list[SliceResult] = []

        for s in self._slices:
            if len(s.fills) < self._min_fills:
                continue
            rep = compute_capital_efficiency(
                s.fills, self._max_pos,
                label=f"{s.asset}/{s.regime}",
            )
            verdict = self._verdict(rep)
            results.append(SliceResult(
                asset=s.asset,
                regime=s.regime,
                n_fills=rep.n_fills,
                hit_rate=rep.hit_rate,
                avg_markout_100ms=rep.avg_markout_100ms,
                roi=rep.roi,
                sharpe=rep.sharpe_annualised,
                verdict=verdict,
            ))

        if not results:
            return MultiAssetValidationReport(
                slices=[], generalises=False,
                weakest_slice=None, strongest_slice=None,
                cross_asset_consistency=0.0, cross_regime_consistency=0.0,
            )

        generalises = all(r.verdict in ("PASS", "WARN") for r in results)
        weakest = min(results, key=lambda r: r.hit_rate)
        strongest = max(results, key=lambda r: r.hit_rate)

        # Cross-asset consistency: std of hit_rate grouped by asset
        asset_hit_rates = _group_mean([r.hit_rate for r in results], [r.asset for r in results])
        cross_asset_cons = _std(list(asset_hit_rates.values()))

        # Cross-regime consistency
        regime_hit_rates = _group_mean([r.hit_rate for r in results], [r.regime for r in results])
        cross_regime_cons = _std(list(regime_hit_rates.values()))

        return MultiAssetValidationReport(
            slices=results,
            generalises=generalises,
            weakest_slice=weakest,
            strongest_slice=strongest,
            cross_asset_consistency=round(cross_asset_cons, 4),
            cross_regime_consistency=round(cross_regime_cons, 4),
        )

    def print_report(self, report: MultiAssetValidationReport) -> None:
        width = 80
        print(f"\n{'=' * width}")
        print("  Multi-Asset / Multi-Regime Validation".center(width))
        print(f"{'=' * width}")

        if not report.slices:
            print("  No slices with sufficient fills.")
            print(f"{'=' * width}\n")
            return

        # Table header
        print(f"  {'Slice':<28} {'Fills':>6} {'HitRate':>8} {'Mkout@100ms':>12} {'ROI':>8} {'Sharpe':>7}  Verdict")
        print(f"  {'-' * 78}")

        for r in report.slices:
            print(
                f"  {r.asset}/{r.regime:<20} {r.n_fills:>6d} "
                f"{r.hit_rate:>8.1%} {r.avg_markout_100ms:>+12.5f} "
                f"{r.roi:>8.4%} {r.sharpe:>7.2f}  {r.verdict}"
            )

        print(f"  {'-' * 78}")
        status = "GENERALISES" if report.generalises else "DOES NOT GENERALISE"
        print(f"  Overall: {status}")
        print(f"  Cross-asset hit-rate std:  {report.cross_asset_consistency:.4f}  (lower = more consistent)")
        print(f"  Cross-regime hit-rate std: {report.cross_regime_consistency:.4f}")

        if report.weakest_slice:
            w = report.weakest_slice
            print(f"  Weakest slice:   {w.asset}/{w.regime} (hit_rate={w.hit_rate:.1%}, verdict={w.verdict})")
        if report.strongest_slice:
            s = report.strongest_slice
            print(f"  Strongest slice: {s.asset}/{s.regime} (hit_rate={s.hit_rate:.1%})")

        print(f"{'=' * width}\n")

    # -------------------------------------------------------------------
    def _verdict(self, rep: CapitalEfficiencyReport) -> str:
        if rep.hit_rate >= self.PASS_HIT_RATE and rep.avg_markout_100ms >= self.PASS_MARKOUT:
            return "PASS"
        if rep.hit_rate >= self.WARN_HIT_RATE and rep.avg_markout_100ms >= self.WARN_MARKOUT:
            return "WARN"
        return "FAIL"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_mean(values: list[float], keys: list[str]) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for k, v in zip(keys, values):
        groups.setdefault(k, []).append(v)
    return {k: sum(vs) / len(vs) for k, vs in groups.items()}


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(var)
