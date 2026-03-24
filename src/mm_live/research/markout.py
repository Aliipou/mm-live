"""
Markout Analysis — the only honest measure of adverse selection.

After every fill, track where mid-price goes at multiple horizons.
A negative markout means informed traders are consistently on the
other side of your fills — you are being picked off.

    buy fill:  markout[h] = mid[t+h] - fill_price   (negative = bought too high)
    sell fill: markout[h] = fill_price - mid[t+h]   (negative = sold too low)

If avg_markout[100ms] < -0.3 * half_spread → adverse selection is eating your edge.
If avg_markout[30s] < avg_markout[100ms]   → informed flow, price keeps moving against you.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass
from typing import Any

# --------------------------------------------------------------------------- #
# Data containers                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class FillMark:
    """
    One fill, plus the markout measurements resolved over time.

    Created on every fill; markouts are filled in as mid prices arrive
    at each horizon deadline.
    """

    fill_id: str
    timestamp_ms: int
    side: str                     # "buy" | "sell"
    fill_price: float
    fair_value_at_fill: float
    half_spread_at_fill: float
    mid_at_fill: float
    # resolved later:
    markouts: dict[int, float]    # {horizon_ms: markout value}
    resolved: set[int]            # which horizons are done

    @property
    def is_fully_resolved(self) -> bool:
        return len(self.resolved) == len(self.markouts)

    def compute_markout(self, horizon_ms: int, mid_now: float) -> float:
        """
        Return the signed markout for a given horizon.

        Positive markout  = price moved in our favour (good).
        Negative markout  = price moved against us (adverse selection).
        """
        if self.side == "buy":
            return mid_now - self.fill_price
        else:
            return self.fill_price - mid_now


@dataclass
class MarkoutStats:
    """Per-horizon aggregate statistics across all resolved fills."""

    horizon_ms: int
    n_fills: int
    avg_markout: float           # negative = adverse selection
    std_markout: float
    pct_negative: float          # fraction of fills where price moved against us
    adverse_selection_ratio: float  # abs(avg_markout) / half_spread — > 0.5 is bad
    verdict: str                 # "CLEAN" | "MILD_AS" | "SEVERE_AS"


# --------------------------------------------------------------------------- #
# Tracker                                                                      #
# --------------------------------------------------------------------------- #


class MarkoutTracker:
    """
    Feed it mid prices every tick. It resolves fill markouts automatically.

    Usage::

        tracker = MarkoutTracker(horizons_ms=[100, 500, 1000, 5000, 30000])

        # on every tick:
        tracker.update_mid(timestamp_ms, mid)

        # on every fill:
        fill_id = tracker.record_fill(timestamp_ms, side, fill_price,
                                      fair_value, half_spread, mid)

        # get results any time:
        stats = tracker.compute_stats()
        tracker.print_report(stats)
    """

    def __init__(
        self,
        horizons_ms: list[int] | None = None,
    ) -> None:
        self._horizons: list[int] = sorted(
            horizons_ms if horizons_ms is not None else [100, 500, 1000, 5000, 30000]
        )
        # fill_id → FillMark (still waiting for some horizons)
        self._pending: dict[str, FillMark] = {}
        # fill_id → FillMark (all horizons resolved)
        self._resolved: dict[str, FillMark] = {}
        # last known mid
        self._last_mid: float | None = None
        self._last_ts: int = 0

    # ----------------------------------------------------------------------- #
    # Public API                                                               #
    # ----------------------------------------------------------------------- #

    def record_fill(
        self,
        timestamp_ms: int,
        side: str,
        fill_price: float,
        fair_value: float,
        half_spread: float,
        mid: float,
    ) -> str:
        """
        Record a new fill and return its fill_id.

        Parameters
        ----------
        timestamp_ms:
            Unix timestamp of the fill in milliseconds.
        side:
            "buy" or "sell" — our side.
        fill_price:
            The price at which we were filled.
        fair_value:
            Model fair value at the time of the fill.
        half_spread:
            Half the quoted spread at fill time (used for AS ratio).
        mid:
            Raw order-book mid at the time of the fill.
        """
        fill_id = str(uuid.uuid4())
        fm = FillMark(
            fill_id=fill_id,
            timestamp_ms=timestamp_ms,
            side=side,
            fill_price=fill_price,
            fair_value_at_fill=fair_value,
            half_spread_at_fill=half_spread,
            mid_at_fill=mid,
            markouts={h: 0.0 for h in self._horizons},
            resolved=set(),
        )
        self._pending[fill_id] = fm
        self._last_mid = mid
        self._last_ts = timestamp_ms
        return fill_id

    def update_mid(self, timestamp_ms: int, mid: float) -> list[str]:
        """
        Called on every market tick.

        Resolves pending markouts for any fill where enough time has
        passed since the fill timestamp.

        Returns
        -------
        list[str]
            fill_ids that were fully resolved during this tick.
        """
        self._last_mid = mid
        self._last_ts = timestamp_ms

        fully_resolved_now: list[str] = []

        for fill_id, fm in list(self._pending.items()):
            elapsed_ms = timestamp_ms - fm.timestamp_ms

            for h in self._horizons:
                if h not in fm.resolved and elapsed_ms >= h:
                    fm.markouts[h] = fm.compute_markout(h, mid)
                    fm.resolved.add(h)

            if fm.is_fully_resolved:
                self._resolved[fill_id] = fm
                del self._pending[fill_id]
                fully_resolved_now.append(fill_id)

        return fully_resolved_now

    def compute_stats(self) -> list[MarkoutStats]:
        """
        Compute per-horizon statistics across all fully-resolved fills.

        Returns an empty list if no fills have been resolved yet.
        """
        if not self._resolved:
            return []

        # Collect all half-spreads (for AS ratio denominator)
        fills = list(self._resolved.values())

        result: list[MarkoutStats] = []
        for h in self._horizons:
            values = [fm.markouts[h] for fm in fills]
            half_spreads = [fm.half_spread_at_fill for fm in fills]
            n = len(values)

            if n == 0:
                continue

            avg_markout = sum(values) / n
            variance = sum((v - avg_markout) ** 2 for v in values) / n
            std_markout = math.sqrt(variance)
            pct_negative = sum(1 for v in values if v < 0) / n
            avg_half_spread = sum(half_spreads) / n

            if avg_half_spread > 0:
                adverse_selection_ratio = abs(avg_markout) / avg_half_spread
            else:
                adverse_selection_ratio = 0.0

            if adverse_selection_ratio < 0.2:
                verdict = "CLEAN"
            elif adverse_selection_ratio < 0.5:
                verdict = "MILD_AS"
            else:
                verdict = "SEVERE_AS"

            result.append(
                MarkoutStats(
                    horizon_ms=h,
                    n_fills=n,
                    avg_markout=avg_markout,
                    std_markout=std_markout,
                    pct_negative=pct_negative,
                    adverse_selection_ratio=adverse_selection_ratio,
                    verdict=verdict,
                )
            )

        return result

    def print_report(self, stats: list[MarkoutStats]) -> None:
        """
        Print a formatted table of markout statistics.

        Columns: horizon | n | avg_markout | std | pct_neg | AS_ratio | verdict
        """
        if not stats:
            print("No resolved fills yet — cannot produce markout report.")
            return

        header = (
            f"{'Horizon':>10}  {'N':>6}  {'AvgMkout':>10}  "
            f"{'Std':>8}  {'%Neg':>6}  {'AS_ratio':>8}  Verdict"
        )
        sep = "-" * len(header)
        print()
        print("=== Markout Analysis Report ===")
        print(sep)
        print(header)
        print(sep)

        has_severe = False
        for s in stats:
            h_label = _fmt_horizon(s.horizon_ms)
            flag = " <-- SEVERE" if s.verdict == "SEVERE_AS" else ""
            if s.verdict == "SEVERE_AS":
                has_severe = True
            print(
                f"{h_label:>10}  {s.n_fills:>6d}  {s.avg_markout:>+10.4f}  "
                f"{s.std_markout:>8.4f}  {s.pct_negative:>6.1%}  "
                f"{s.adverse_selection_ratio:>8.3f}  {s.verdict}{flag}"
            )

        print(sep)

        # Summary line
        net = self.net_edge_per_fill()
        sign = "+" if net >= 0 else ""
        print(f"Net edge after adverse selection: {sign}{net:.4f} USD per fill")

        if has_severe:
            print()
            print(
                "WARNING: SEVERE adverse selection detected at one or more horizons. "
                "Informed flow is consistently on the other side of your quotes. "
                "Consider widening spreads or adding a toxic-flow filter."
            )

        print()

    def net_edge_per_fill(self) -> float:
        """
        Estimate net edge per fill after adverse selection.

        Returns
        -------
        float
            half_spread_at_fill - abs(avg_markout[shortest_horizon]).
            Positive → still profitable on net.
            Negative → adverse selection is erasing your edge.

        Returns 0.0 if there are no resolved fills.
        """
        if not self._resolved:
            return 0.0

        fills = list(self._resolved.values())
        shortest = self._horizons[0]

        avg_half_spread = sum(fm.half_spread_at_fill for fm in fills) / len(fills)
        avg_markout_short = sum(fm.markouts[shortest] for fm in fills) / len(fills)

        return avg_half_spread - abs(avg_markout_short)

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serialisable summary of the tracker state.

        Includes metadata, per-horizon stats, and net edge.
        """
        stats = self.compute_stats()
        return {
            "meta": {
                "n_pending": len(self._pending),
                "n_resolved": len(self._resolved),
                "horizons_ms": self._horizons,
                "last_timestamp_ms": self._last_ts,
                "generated_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            },
            "net_edge_per_fill": self.net_edge_per_fill(),
            "stats": [
                {
                    "horizon_ms": s.horizon_ms,
                    "n_fills": s.n_fills,
                    "avg_markout": s.avg_markout,
                    "std_markout": s.std_markout,
                    "pct_negative": s.pct_negative,
                    "adverse_selection_ratio": s.adverse_selection_ratio,
                    "verdict": s.verdict,
                }
                for s in stats
            ],
        }

    # ----------------------------------------------------------------------- #
    # Convenience properties                                                   #
    # ----------------------------------------------------------------------- #

    @property
    def n_pending(self) -> int:
        return len(self._pending)

    @property
    def n_resolved(self) -> int:
        return len(self._resolved)

    @property
    def horizons_ms(self) -> list[int]:
        return list(self._horizons)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _fmt_horizon(ms: int) -> str:
    """Human-readable horizon label, e.g. 100ms, 1s, 30s."""
    if ms < 1000:
        return f"{ms}ms"
    elif ms % 1000 == 0:
        return f"{ms // 1000}s"
    else:
        return f"{ms / 1000:.1f}s"
