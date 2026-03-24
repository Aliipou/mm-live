"""
Capital Efficiency — return per unit of risk deployed.

The market measures you on return-per-risk, not absolute PnL.

Metrics produced
----------------
roi                 : total_pnl / (avg_position_size * avg_price)
sharpe_annualised   : daily_pnl mean/std * sqrt(252)
sortino_annualised  : downside-deviation Sharpe (only penalises losses)
calmar              : annualised_return / max_drawdown
pnl_per_fill        : gross PnL divided by number of fills
capital_turnover    : volume_traded / avg_capital_deployed
hit_rate            : fraction of fills where markout > 0

All metrics are split by regime (LOW_VOL / NORMAL / HIGH_VOL) and
by asset so the caller can compare performance across conditions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class FillSummary:
    """
    Minimal per-fill data needed for capital efficiency calculations.

    Parameters
    ----------
    timestamp_ms  : fill timestamp in Unix milliseconds
    side          : "buy" or "sell"
    fill_price    : execution price
    qty           : base-asset quantity (e.g. BTC)
    pnl           : realised PnL for this fill in quote currency
    markout_100ms : mid(t+100ms) - fill_price (sign-adjusted for side)
    regime        : "low_vol" | "normal" | "high_vol"
    asset         : e.g. "BTCUSDT", "ETHUSDT"
    """
    timestamp_ms: int
    side: str
    fill_price: float
    qty: float
    pnl: float
    markout_100ms: float
    regime: str = "normal"
    asset: str = "BTCUSDT"


@dataclass
class CapitalEfficiencyReport:
    """Aggregate capital-efficiency metrics for a given slice."""

    label: str                    # e.g. "ALL" | "high_vol" | "ETHUSDT"
    n_fills: int

    total_pnl: float
    avg_pnl_per_fill: float

    # ROI: PnL / capital deployed
    avg_capital_deployed: float   # avg_position_size * avg_price
    roi: float                    # total_pnl / avg_capital_deployed

    # Risk-adjusted returns
    sharpe_annualised: float
    sortino_annualised: float
    calmar: float

    # Edge quality
    hit_rate: float               # fraction markout_100ms > 0
    avg_markout_100ms: float      # negative = adverse selection

    # Turnover
    total_volume: float           # sum(qty * fill_price)
    capital_turnover: float       # total_volume / avg_capital_deployed


def compute_capital_efficiency(
    fills: Sequence[FillSummary],
    max_position_qty: float,
    label: str = "ALL",
) -> CapitalEfficiencyReport:
    """
    Compute capital efficiency metrics from a sequence of fills.

    Parameters
    ----------
    fills            : per-fill records
    max_position_qty : maximum inventory limit (base asset units)
    label            : tag for the report (e.g. "high_vol", "ETHUSDT")

    Returns
    -------
    CapitalEfficiencyReport
    """
    if not fills:
        return _empty_report(label)

    n = len(fills)
    total_pnl = sum(f.pnl for f in fills)
    avg_pnl_per_fill = total_pnl / n

    # Capital deployed: max_position_qty * avg price
    avg_price = sum(f.fill_price for f in fills) / n
    avg_capital = max_position_qty * avg_price
    roi = (total_pnl / avg_capital) if avg_capital > 0 else 0.0

    # Daily PnL series (group by day bucket)
    daily_pnl = _daily_pnl(fills)
    sharpe = _sharpe_annualised(daily_pnl)
    sortino = _sortino_annualised(daily_pnl)
    calmar = _calmar(daily_pnl, total_pnl)

    # Edge
    hit_rate = sum(1 for f in fills if f.markout_100ms > 0) / n
    avg_markout = sum(f.markout_100ms for f in fills) / n

    # Turnover
    total_vol = sum(f.qty * f.fill_price for f in fills)
    capital_turnover = (total_vol / avg_capital) if avg_capital > 0 else 0.0

    return CapitalEfficiencyReport(
        label=label,
        n_fills=n,
        total_pnl=round(total_pnl, 6),
        avg_pnl_per_fill=round(avg_pnl_per_fill, 6),
        avg_capital_deployed=round(avg_capital, 2),
        roi=round(roi, 6),
        sharpe_annualised=round(sharpe, 4),
        sortino_annualised=round(sortino, 4),
        calmar=round(calmar, 4),
        hit_rate=round(hit_rate, 4),
        avg_markout_100ms=round(avg_markout, 6),
        total_volume=round(total_vol, 2),
        capital_turnover=round(capital_turnover, 4),
    )


def compute_by_regime(
    fills: Sequence[FillSummary],
    max_position_qty: float,
) -> dict[str, CapitalEfficiencyReport]:
    """Return one CapitalEfficiencyReport per regime label."""
    from itertools import groupby
    by_regime: dict[str, list[FillSummary]] = {}
    for f in fills:
        by_regime.setdefault(f.regime, []).append(f)
    return {
        regime: compute_capital_efficiency(group, max_position_qty, label=regime)
        for regime, group in by_regime.items()
    }


def compute_by_asset(
    fills: Sequence[FillSummary],
    max_position_qty: float,
) -> dict[str, CapitalEfficiencyReport]:
    """Return one CapitalEfficiencyReport per asset."""
    by_asset: dict[str, list[FillSummary]] = {}
    for f in fills:
        by_asset.setdefault(f.asset, []).append(f)
    return {
        asset: compute_capital_efficiency(group, max_position_qty, label=asset)
        for asset, group in by_asset.items()
    }


def print_capital_report(report: CapitalEfficiencyReport) -> None:
    """Print a formatted capital efficiency table."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  Capital Efficiency Report: {report.label}".center(width))
    print(f"{'=' * width}")
    rows = [
        ("Fills", str(report.n_fills)),
        ("Total PnL", f"{report.total_pnl:+.4f} USD"),
        ("PnL per fill", f"{report.avg_pnl_per_fill:+.6f} USD"),
        ("Capital deployed", f"{report.avg_capital_deployed:,.2f} USD"),
        ("ROI", f"{report.roi:.4%}"),
        ("Sharpe (annualised)", f"{report.sharpe_annualised:.3f}"),
        ("Sortino (annualised)", f"{report.sortino_annualised:.3f}"),
        ("Calmar ratio", f"{report.calmar:.3f}"),
        ("Hit rate (markout>0)", f"{report.hit_rate:.1%}"),
        ("Avg markout @100ms", f"{report.avg_markout_100ms:+.6f}"),
        ("Total volume", f"{report.total_volume:,.2f} USD"),
        ("Capital turnover", f"{report.capital_turnover:.2f}x"),
    ]
    for label, value in rows:
        print(f"  {label:<26} {value}")
    print(f"{'=' * width}\n")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _daily_pnl(fills: Sequence[FillSummary]) -> list[float]:
    """Bucket fills by day (ms // 86_400_000) and sum PnL per day."""
    buckets: dict[int, float] = {}
    for f in fills:
        day = f.timestamp_ms // 86_400_000
        buckets[day] = buckets.get(day, 0.0) + f.pnl
    return list(buckets.values())


def _sharpe_annualised(daily: list[float]) -> float:
    if len(daily) < 2:
        return 0.0
    n = len(daily)
    mean = sum(daily) / n
    var = sum((x - mean) ** 2 for x in daily) / (n - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(252)


def _sortino_annualised(daily: list[float]) -> float:
    if len(daily) < 2:
        return 0.0
    mean = sum(daily) / len(daily)
    neg = [x for x in daily if x < 0]
    if not neg:
        return float("inf")
    downside_var = sum(x ** 2 for x in neg) / len(daily)
    downside_std = math.sqrt(downside_var)
    if downside_std == 0:
        return 0.0
    return (mean / downside_std) * math.sqrt(252)


def _calmar(daily: list[float], total_pnl: float) -> float:
    if len(daily) < 2:
        return 0.0
    # Max drawdown from cumulative PnL curve
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for d in daily:
        cum += d
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    if max_dd == 0:
        return float("inf") if total_pnl > 0 else 0.0
    annualised_return = total_pnl * (252 / len(daily))
    return annualised_return / max_dd


def _empty_report(label: str) -> CapitalEfficiencyReport:
    return CapitalEfficiencyReport(
        label=label, n_fills=0,
        total_pnl=0.0, avg_pnl_per_fill=0.0,
        avg_capital_deployed=0.0, roi=0.0,
        sharpe_annualised=0.0, sortino_annualised=0.0, calmar=0.0,
        hit_rate=0.0, avg_markout_100ms=0.0,
        total_volume=0.0, capital_turnover=0.0,
    )
