"""
Edge Test 1: Order Flow Imbalance -> Future Mid Prediction

Hypothesis: OFI at time t predicts (mid[t+h] - mid[t]) / mid[t]
where h in {100ms, 500ms, 1s, 5s}

If true: we have a real statistical edge in fair value estimation.
If false: our Kalman + imbalance signal is decorative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ImbalanceSample:
    """
    A single observation: OFI at time t paired with future mid-price outcomes.

    Attributes
    ----------
    timestamp_ms:
        Wall-clock timestamp in milliseconds when the sample was captured.
    imbalance:
        Order flow imbalance in [-1, +1] at time t.
    mid_now:
        Mid-price at time t.
    mid_future:
        Mapping of horizon_ms -> mid-price observed at t + horizon_ms.
        Populated lazily as future ticks arrive.
    future_return:
        Mapping of horizon_ms -> (mid_future - mid_now) / mid_now.
        Populated once mid_future[horizon] is set.
    """

    timestamp_ms: int
    imbalance: float
    mid_now: float
    mid_future: dict[int, float] = field(default_factory=dict)
    future_return: dict[int, float] = field(default_factory=dict)

    def resolve_horizon(self, horizon_ms: int, mid_at_horizon: float) -> None:
        """Record the mid-price that arrived at t + horizon_ms and compute return."""
        self.mid_future[horizon_ms] = mid_at_horizon
        if self.mid_now != 0.0:
            self.future_return[horizon_ms] = (mid_at_horizon - self.mid_now) / self.mid_now
        else:
            self.future_return[horizon_ms] = 0.0


@dataclass
class EdgeTestResult:
    """
    OLS regression result for a single prediction horizon.

    Attributes
    ----------
    horizon_ms:
        Prediction horizon in milliseconds.
    n_samples:
        Number of (imbalance, return) pairs used in the regression.
    correlation:
        Pearson r between imbalance at t and future return at t + horizon_ms.
    r_squared:
        Coefficient of determination (R^2).
    t_statistic:
        t-test statistic for the slope coefficient.
    p_value:
        Two-tailed p-value for the slope.
    beta:
        OLS slope (expected return per unit of imbalance).
    intercept:
        OLS intercept.
    significant:
        True if p_value < 0.05.
    summary:
        Human-readable one-liner describing the result.
    """

    horizon_ms: int
    n_samples: int
    correlation: float
    r_squared: float
    t_statistic: float
    p_value: float
    beta: float
    intercept: float
    significant: bool
    summary: str


# ---------------------------------------------------------------------------
# Main test class
# ---------------------------------------------------------------------------


class ImbalanceEdgeTest:
    """
    Accumulates live (timestamp, imbalance, mid) observations and tests whether
    OFI at time t predicts mid-price returns over multiple horizons.

    The test uses OLS regression via ``scipy.stats.linregress``.  No pandas or
    matplotlib are required.

    Parameters
    ----------
    horizons_ms:
        List of forward-looking horizons (in milliseconds) to test.
        Default: [100, 500, 1000, 5000].

    Example
    -------
    ::

        test = ImbalanceEdgeTest()
        # feed with live data for a while...
        for ts, imb, mid in stream:
            test.add_sample(ts, imb, mid)
        results = test.run_test()
        test.print_report(results)
    """

    def __init__(self, horizons_ms: list[int] | None = None) -> None:
        self.horizons_ms: list[int] = (
            horizons_ms if horizons_ms is not None else [100, 500, 1000, 5000]
        )
        self._max_horizon: int = max(self.horizons_ms)

        # Pending samples: timestamp_ms -> ImbalanceSample
        # Once all horizons are resolved the sample moves to _completed.
        self._pending: dict[int, ImbalanceSample] = {}

        # Fully-resolved samples (all horizons present)
        self._completed: list[ImbalanceSample] = []

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_sample(self, timestamp_ms: int, imbalance: float, mid: float) -> None:
        """
        Record a new (timestamp, imbalance, mid) observation.

        This method also scans existing pending samples and resolves any
        horizons that the current tick satisfies - i.e., for each pending
        sample S where ``timestamp_ms - S.timestamp_ms >= horizon_ms``, the
        current ``mid`` is used as the future mid for that horizon.

        Once all horizons of a pending sample are resolved it is moved to the
        completed pool used by :meth:`run_test`.

        Parameters
        ----------
        timestamp_ms:
            Current wall-clock time in milliseconds.
        imbalance:
            OFI signal in [-1, +1].
        mid:
            Current mid-price (best_bid + best_ask) / 2.
        """
        # 1. Record this tick as a new pending sample.
        sample = ImbalanceSample(
            timestamp_ms=timestamp_ms,
            imbalance=imbalance,
            mid_now=mid,
        )
        self._pending[timestamp_ms] = sample

        # 2. Resolve horizons for all older pending samples.
        elapsed_keys = list(self._pending.keys())
        graduated: list[int] = []

        for key in elapsed_keys:
            pending_sample = self._pending[key]
            elapsed = timestamp_ms - pending_sample.timestamp_ms

            for horizon in self.horizons_ms:
                if horizon not in pending_sample.mid_future and elapsed >= horizon:
                    pending_sample.resolve_horizon(horizon, mid)

            # If all horizons are resolved, graduate this sample.
            if all(h in pending_sample.mid_future for h in self.horizons_ms):
                self._completed.append(pending_sample)
                graduated.append(key)

        for key in graduated:
            del self._pending[key]

    # ------------------------------------------------------------------
    # Statistical test
    # ------------------------------------------------------------------

    def run_test(self) -> list[EdgeTestResult]:
        """
        Run OLS regression of future return ~ imbalance for each horizon.

        Uses only ``scipy.stats.linregress``.  Requires at least 3 completed
        samples; returns an empty list otherwise.

        Returns
        -------
        list[EdgeTestResult]
            One result per horizon, ordered by ``self.horizons_ms``.
        """
        if len(self._completed) < 3:
            return []

        results: list[EdgeTestResult] = []

        for horizon in self.horizons_ms:
            xs: list[float] = []
            ys: list[float] = []

            for s in self._completed:
                if horizon in s.future_return:
                    xs.append(s.imbalance)
                    ys.append(s.future_return[horizon])

            n = len(xs)
            if n < 3:
                # Not enough data for this horizon yet - skip.
                continue

            x_arr = np.asarray(xs, dtype=np.float64)
            y_arr = np.asarray(ys, dtype=np.float64)

            lr = stats.linregress(x_arr, y_arr)

            slope: float = float(lr.slope)          # type: ignore[arg-type]
            intercept: float = float(lr.intercept)  # type: ignore[arg-type]
            r_value: float = float(lr.rvalue)       # type: ignore[arg-type]
            p_value: float = float(lr.pvalue)       # type: ignore[arg-type]
            t_stat: float = (
                float(lr.slope / lr.stderr)         # type: ignore[operator]
                if lr.stderr and lr.stderr != 0
                else 0.0
            )
            r_sq: float = r_value ** 2
            significant = p_value < 0.05

            # Human-readable summary
            horizon_label = _format_horizon(horizon)
            if significant:
                direction = "positive" if slope > 0 else "negative"
                summary = (
                    f"EDGE DETECTED at {horizon_label}: OFI has a {direction} "
                    f"predictive relationship with future returns "
                    f"(r={r_value:+.3f}, p={p_value:.4f}, beta={slope:.2e})."
                )
            else:
                summary = (
                    f"No significant edge at {horizon_label}: "
                    f"OFI does not reliably predict future returns "
                    f"(r={r_value:+.3f}, p={p_value:.4f})."
                )

            results.append(
                EdgeTestResult(
                    horizon_ms=horizon,
                    n_samples=n,
                    correlation=r_value,
                    r_squared=r_sq,
                    t_statistic=t_stat,
                    p_value=p_value,
                    beta=slope,
                    intercept=intercept,
                    significant=significant,
                    summary=summary,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, results: list[EdgeTestResult]) -> None:
        """
        Print a formatted table of edge test results to stdout.

        Output format::

            horizon  |    n  |      r  |    R^2  |  t-stat  |  p-value  |  sig
            -----------------------------------------------------------------------
            100ms    |   847 |  +0.120 |  0.0144 |    3.41  |  0.0007   |  YES
            500ms    |   845 |  +0.083 |  0.0069 |    2.41  |  0.0162   |  YES
            1000ms   |   841 |  +0.031 |  0.0010 |    0.90  |  0.3687   |  no
            5000ms   |   823 |  -0.008 |  0.0001 |   -0.23  |  0.8193   |  no

        Parameters
        ----------
        results:
            Output of :meth:`run_test`.
        """
        if not results:
            print("ImbalanceEdgeTest: no results (insufficient data).")
            return

        header = (
            f"{'horizon':>8}  | {'n':>6}  | {'r':>7}  | {'R^2':>7}  "
            f"| {'t-stat':>8}  | {'p-value':>9}  | sig"
        )
        sep = "-" * len(header)
        print()
        print("=== OFI Edge Test: Imbalance -> Future Return ===")
        print(sep)
        print(header)
        print(sep)
        for r in results:
            label = _format_horizon(r.horizon_ms)
            sig_str = "YES **" if r.significant else "no"
            print(
                f"{label:>8}  | {r.n_samples:>6}  | {r.correlation:>+7.3f}  | "
                f"{r.r_squared:>7.4f}  | {r.t_statistic:>8.2f}  | "
                f"{r.p_value:>9.4f}  |  {sig_str}"
            )
        print(sep)

        # Edge verdict
        sig_count = sum(1 for r in results if r.significant)
        total = len(results)
        print(
            f"\nVerdict: {sig_count}/{total} horizons show significant predictive power (p < 0.05)."
        )
        if sig_count > 0:
            print("-> OFI carries a measurable edge. Kalman + imbalance adjustment is justified.")
        else:
            print(
                "-> No edge detected. Consider tuning depth_levels / ema_alpha "
                "or reviewing signal construction."
            )
        print()

        for r in results:
            print(f"  [{_format_horizon(r.horizon_ms)}] {r.summary}")
        print()

    def plot_ascii(self, results: list[EdgeTestResult]) -> str:
        """
        Return an ASCII bar chart of R^2 per horizon.

        No matplotlib dependency - pure string manipulation.

        Example output::

            OFI Predictive Power: R^2 per Horizon
            ======================================
              100ms  | ########............  0.0144 *
              500ms  | #####...............  0.0069 *
                 1s  | ##..................  0.0010
                 5s  | ....................  0.0001

          * = significant at p < 0.05
          Scale: max R^2 = 0.0144

        Parameters
        ----------
        results:
            Output of :meth:`run_test`.

        Returns
        -------
        str
            Multi-line ASCII chart string (ready for ``print()``).
        """
        if not results:
            return "ImbalanceEdgeTest: no results to plot."

        bar_width = 20
        max_r2 = max(r.r_squared for r in results)
        # Avoid division by zero when all R^2 are 0.
        scale = max_r2 if max_r2 > 0 else 1.0

        lines: list[str] = [
            "",
            "OFI Predictive Power: R^2 per Horizon",
            "======================================",
        ]

        for r in results:
            label = _format_horizon(r.horizon_ms)
            filled = math.floor((r.r_squared / scale) * bar_width)
            empty = bar_width - filled
            bar = "#" * filled + "." * empty
            sig_marker = " *" if r.significant else "  "
            lines.append(f"  {label:>6}  | {bar}  {r.r_squared:.4f}{sig_marker}")

        lines.append("")
        lines.append("  * = significant at p < 0.05")
        lines.append(f"  Scale: max R^2 = {max_r2:.4f}")
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    @property
    def n_completed(self) -> int:
        """Number of fully-resolved samples available for regression."""
        return len(self._completed)

    @property
    def n_pending(self) -> int:
        """Number of samples still awaiting future mid-prices."""
        return len(self._pending)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_horizon(horizon_ms: int) -> str:
    """Return a human-readable label for a horizon in milliseconds."""
    if horizon_ms < 1000:
        return f"{horizon_ms}ms"
    return f"{horizon_ms // 1000}s"
