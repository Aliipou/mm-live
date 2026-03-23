"""
Edge Test 2: Vol Regime → Fill Quality

Hypothesis: in HIGH_VOL regime, adverse selection dominates spread capture.
If true: we should widen spreads more aggressively or pause in high-vol.
If false: regime detection adds no value.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING


class Regime(Enum):
    """Volatility regime labels aligned with DualVolatility.regime."""

    LOW_VOL = "low_vol"
    NORMAL = "normal"
    HIGH_VOL = "high_vol"


@dataclass
class FillRecord:
    """A single simulated or live fill, tagged with its regime context.

    Parameters
    ----------
    timestamp_ms:
        Unix timestamp of the fill in milliseconds.
    side:
        ``"buy"`` if we bought (bid was hit) or ``"sell"`` if we sold (ask was hit).
    fill_price:
        The price at which the fill was executed.
    fair_value_at_fill:
        The engine's estimate of fair value at the moment of the fill.
    regime:
        The volatility regime active when the fill occurred.
    spread_at_fill:
        The full bid-ask spread (ask − bid) at the time of the fill.
    """

    timestamp_ms: int
    side: str  # "buy" | "sell"
    fill_price: float
    fair_value_at_fill: float
    regime: Regime
    spread_at_fill: float

    @property
    def spread_capture(self) -> float:
        """Edge collected by quoting on the right side of fair value.

        Positive means we bought below fair (or sold above fair) — good.
        Negative means the fill was through fair value against us.
        """
        if self.side == "buy":
            return self.fair_value_at_fill - self.fill_price
        return self.fill_price - self.fair_value_at_fill

    @property
    def adverse_selection(self) -> float:
        """Cost extracted by the informed counterparty.

        This is the opposite sign of spread_capture:
        a positive adverse_selection value means we were picked off.
        """
        return -self.spread_capture


@dataclass
class RegimeStats:
    """Aggregated fill-quality statistics for a single volatility regime.

    Parameters
    ----------
    regime:
        The regime these statistics belong to.
    n_fills:
        Total number of fills in this regime.
    avg_spread_capture:
        Mean spread capture per fill (positive = good).
    avg_adverse_selection:
        Mean adverse-selection cost per fill (positive = bad).
    net_edge_per_fill:
        ``avg_spread_capture − avg_adverse_selection``.  Positive means we are
        on net extracting edge; negative means we are being picked off.
    fill_rate:
        Observed fills per minute over the recording window.
    sharpe:
        Risk-adjusted net edge: ``mean(net_edge) / std(net_edge)``.
        Returns 0.0 when fewer than two fills are available.
    win_rate:
        Fraction of fills whose individual net edge is strictly positive.
    """

    regime: Regime
    n_fills: int
    avg_spread_capture: float
    avg_adverse_selection: float
    net_edge_per_fill: float
    fill_rate: float
    sharpe: float
    win_rate: float


class RegimeAttributionTracker:
    """Accumulates fills and computes per-regime edge statistics.

    Usage::

        tracker = RegimeAttributionTracker()
        tracker.record_fill(fill_price=100.0, fair_value=100.3,
                            side="buy", regime=Regime.HIGH_VOL, spread=0.8)
        stats = tracker.compute_stats()
        tracker.print_report()
        mults = tracker.recommendation()
    """

    def __init__(self) -> None:
        # Raw fill records keyed by regime for fast per-regime grouping.
        self._fills: dict[Regime, list[FillRecord]] = {r: [] for r in Regime}
        # Track the start time so we can compute fills-per-minute.
        self._start_ms: int | None = None

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def record_fill(
        self,
        fill_price: float,
        fair_value: float,
        side: str,
        regime: Regime,
        spread: float,
        timestamp_ms: int | None = None,
    ) -> None:
        """Append a fill to the internal store.

        Parameters
        ----------
        fill_price:
            Execution price of the fill.
        fair_value:
            Model fair value at the time of the fill.
        side:
            ``"buy"`` or ``"sell"``.
        regime:
            Volatility regime at fill time.
        spread:
            Full bid-ask spread in USD at fill time.
        timestamp_ms:
            Optional explicit timestamp; defaults to ``time.time_ns() // 1_000_000``.
        """
        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
        if self._start_ms is None:
            self._start_ms = ts

        record = FillRecord(
            timestamp_ms=ts,
            side=side,
            fill_price=fill_price,
            fair_value_at_fill=fair_value,
            regime=regime,
            spread_at_fill=spread,
        )
        self._fills[regime].append(record)

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def compute_stats(self) -> dict[Regime, RegimeStats]:
        """Compute per-regime fill-quality statistics.

        Returns
        -------
        dict[Regime, RegimeStats]
            A mapping from each Regime to its ``RegimeStats``.  Regimes with
            zero fills still appear in the dict but carry zero/NaN-safe values.
        """
        # Derive overall window length in minutes for fill-rate calculation.
        all_fills = [f for fills in self._fills.values() for f in fills]
        if len(all_fills) >= 2:
            window_ms = max(f.timestamp_ms for f in all_fills) - min(
                f.timestamp_ms for f in all_fills
            )
            window_min = max(window_ms / 60_000.0, 1e-9)
        else:
            window_min = 1.0

        result: dict[Regime, RegimeStats] = {}
        for regime, fills in self._fills.items():
            n = len(fills)
            if n == 0:
                result[regime] = RegimeStats(
                    regime=regime,
                    n_fills=0,
                    avg_spread_capture=0.0,
                    avg_adverse_selection=0.0,
                    net_edge_per_fill=0.0,
                    fill_rate=0.0,
                    sharpe=0.0,
                    win_rate=0.0,
                )
                continue

            sc_vals = [f.spread_capture for f in fills]
            as_vals = [f.adverse_selection for f in fills]
            net_vals = [sc - av for sc, av in zip(sc_vals, as_vals)]

            avg_sc = sum(sc_vals) / n
            avg_as = sum(as_vals) / n
            avg_net = sum(net_vals) / n

            # Sharpe: mean / std; guard against zero variance
            if n >= 2:
                mean_net = avg_net
                variance = sum((x - mean_net) ** 2 for x in net_vals) / (n - 1)
                std_net = math.sqrt(variance) if variance > 0 else 1e-10
                sharpe = mean_net / std_net
            else:
                sharpe = 0.0

            win_rate = sum(1 for v in net_vals if v > 0) / n
            fill_rate = n / window_min

            result[regime] = RegimeStats(
                regime=regime,
                n_fills=n,
                avg_spread_capture=avg_sc,
                avg_adverse_selection=avg_as,
                net_edge_per_fill=avg_net,
                fill_rate=fill_rate,
                sharpe=sharpe,
                win_rate=win_rate,
            )

        return result

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def print_report(self) -> None:
        """Print a formatted regime-attribution table to stdout.

        Columns: regime | n_fills | spread_capture | adverse_sel | net_edge |
                 sharpe | win_rate

        Followed by a human-readable conclusion comparing HIGH_VOL vs NORMAL.
        """
        stats = self.compute_stats()

        header = (
            f"{'Regime':<10} {'N':<6} {'SprdCapt':>10} {'AdvSel':>10} "
            f"{'NetEdge':>10} {'Sharpe':>8} {'WinRate':>9}"
        )
        sep = "-" * len(header)

        print()
        print("=== Regime Attribution Report ===")
        print(sep)
        print(header)
        print(sep)

        for regime in Regime:
            s = stats[regime]
            print(
                f"{regime.value:<10} {s.n_fills:<6} {s.avg_spread_capture:>+10.4f} "
                f"{s.avg_adverse_selection:>+10.4f} {s.net_edge_per_fill:>+10.4f} "
                f"{s.sharpe:>8.3f} {s.win_rate:>8.1%}"
            )

        print(sep)
        print()

        # Conclusion line
        hv = stats[Regime.HIGH_VOL]
        nm = stats[Regime.NORMAL]

        if hv.n_fills == 0 or nm.n_fills == 0:
            print("Insufficient data: need fills in both HIGH_VOL and NORMAL regime.")
            return

        # Adverse selection ratio: how much worse is HIGH_VOL?
        if abs(nm.avg_adverse_selection) > 1e-10:
            as_ratio = abs(hv.avg_adverse_selection) / abs(nm.avg_adverse_selection)
        else:
            as_ratio = float("nan")

        mults = self.recommendation()
        widen_pct = (mults.get(Regime.HIGH_VOL, 1.0) - 1.0) * 100

        if math.isnan(as_ratio):
            print("Conclusion: NORMAL regime has no adverse selection to compare against.")
        elif as_ratio > 1.05:
            print(
                f"Conclusion: HIGH_VOL regime has {as_ratio:.1f}x worse adverse selection "
                f"-> widen by {widen_pct:.0f}%"
            )
        else:
            print(
                "Conclusion: HIGH_VOL regime shows no meaningful adverse selection premium "
                "(regime detection may not add value for spread widening)."
            )

        print()

    # ------------------------------------------------------------------ #
    # Recommendation                                                       #
    # ------------------------------------------------------------------ #

    def recommendation(self) -> dict[Regime, float]:
        """Return a spread multiplier for each regime.

        Logic
        -----
        * If ``HIGH_VOL.net_edge < NORMAL.net_edge * 0.5`` → recommend 1.5× for HIGH_VOL.
        * Otherwise → 1.0× (no change) for HIGH_VOL.
        * LOW_VOL: 0.8× (tighten, collect more flow).
        * NORMAL: 1.0× (baseline).

        Returns
        -------
        dict[Regime, float]
            ``{Regime.LOW_VOL: 0.8, Regime.NORMAL: 1.0, Regime.HIGH_VOL: <1.0 or 1.5>}``
        """
        stats = self.compute_stats()
        hv_net = stats[Regime.HIGH_VOL].net_edge_per_fill
        nm_net = stats[Regime.NORMAL].net_edge_per_fill

        # Determine whether HIGH_VOL edge is sufficiently degraded.
        high_vol_mult: float
        if stats[Regime.HIGH_VOL].n_fills == 0 or stats[Regime.NORMAL].n_fills == 0:
            # Not enough data — be conservative, widen.
            high_vol_mult = 1.5
        elif hv_net < nm_net * 0.5:
            # Edge in high vol is less than half of normal → widen.
            high_vol_mult = 1.5
        else:
            # Regime discrimination not yet warranted.
            high_vol_mult = 1.0

        return {
            Regime.LOW_VOL: 0.8,
            Regime.NORMAL: 1.0,
            Regime.HIGH_VOL: high_vol_mult,
        }
