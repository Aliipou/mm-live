"""
mm_live.research — edge-validation modules.

Exports
-------
imbalance_prediction
    EdgeTestResult, ImbalanceEdgeTest, ImbalanceSample

regime_attribution
    FillRecord, Regime, RegimeStats, RegimeAttributionTracker

benchmark
    BacktestResult, BenchmarkRunner, FixedSpreadMaker, MarketTick, NaiveMaker
"""

from __future__ import annotations

from mm_live.research.benchmark import (
    BacktestResult,
    BenchmarkRunner,
    FixedSpreadMaker,
    MarketTick,
    NaiveMaker,
)
from mm_live.research.imbalance_prediction import (
    EdgeTestResult,
    ImbalanceEdgeTest,
    ImbalanceSample,
)
from mm_live.research.regime_attribution import (
    FillRecord,
    Regime,
    RegimeAttributionTracker,
    RegimeStats,
)

__all__ = [
    # imbalance_prediction
    "EdgeTestResult",
    "ImbalanceEdgeTest",
    "ImbalanceSample",
    # regime_attribution
    "FillRecord",
    "Regime",
    "RegimeAttributionTracker",
    "RegimeStats",
    # benchmark
    "BacktestResult",
    "BenchmarkRunner",
    "FixedSpreadMaker",
    "MarketTick",
    "NaiveMaker",
]
