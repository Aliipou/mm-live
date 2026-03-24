"""Latency measurement utilities — rolling percentile tracker, pure Python."""

from __future__ import annotations

import collections
from typing import NamedTuple


class _Stats(NamedTuple):
    p50: float
    p99: float
    count: int


class LatencyTracker:
    """Rolling latency tracker with per-label P50 / P99 percentiles.

    Each label maintains a fixed-size circular buffer (``collections.deque``
    with ``maxlen=window``).  Percentiles are computed on demand by sorting a
    snapshot of the buffer — no NumPy required.

    Example::

        tracker = LatencyTracker(window=500)

        t0 = time.monotonic()
        await place_order(...)
        tracker.record("order_rtt", (time.monotonic() - t0) * 1000)

        print(tracker.p99("order_rtt"))   # 99th-percentile in ms
        print(tracker.summary())

    Thread-safety: the individual ``deque.append`` calls are GIL-protected
    in CPython, so concurrent recording from multiple threads is safe.
    Percentile reads take a ``list()`` snapshot before sorting to avoid
    issues if a writer is active during the read.
    """

    def __init__(self, window: int = 1000) -> None:
        """Initialise the tracker.

        Args:
            window: Maximum number of samples retained per label.
                    Older samples are evicted automatically once the buffer
                    is full (FIFO).
        """
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")

        self._window: int = window
        self._data: dict[str, collections.deque[float]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bucket(self, label: str) -> collections.deque[float]:
        """Return (creating if necessary) the deque for *label*."""
        if label not in self._data:
            self._data[label] = collections.deque(maxlen=self._window)
        return self._data[label]

    @staticmethod
    def _percentile(sorted_values: list[float], pct: float) -> float:
        """Return the *pct*-th percentile from a **sorted** list.

        Uses the nearest-rank method.  Returns ``0.0`` for an empty list.

        Args:
            sorted_values: Pre-sorted list of floats.
            pct: Percentile in the range [0, 100].
        """
        n = len(sorted_values)
        if n == 0:
            return 0.0
        # nearest-rank: index = ceil(pct/100 * n) - 1, clamped
        idx = max(0, min(n - 1, int((pct / 100.0) * n + 0.5) - 1))
        return sorted_values[idx]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, label: str, latency_ms: float) -> None:
        """Append a latency sample for *label*.

        Args:
            label:      Arbitrary string key (e.g. ``"order_rtt"``).
            latency_ms: Observed latency in milliseconds.
        """
        self._bucket(label).append(latency_ms)

    def p50(self, label: str) -> float:
        """Return the 50th-percentile (median) latency for *label* in ms.

        Returns ``0.0`` if no samples have been recorded.
        """
        return self._percentile(sorted(self._bucket(label)), 50.0)

    def p99(self, label: str) -> float:
        """Return the 99th-percentile latency for *label* in ms.

        Returns ``0.0`` if no samples have been recorded.
        """
        return self._percentile(sorted(self._bucket(label)), 99.0)

    def p_n(self, label: str, percentile: float) -> float:
        """Return an arbitrary percentile for *label*.

        Args:
            label:      Sample label.
            percentile: Desired percentile in the range [0, 100].
        """
        if not (0.0 <= percentile <= 100.0):
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")
        return self._percentile(sorted(self._bucket(label)), percentile)

    def count(self, label: str) -> int:
        """Return the number of samples currently held for *label*."""
        return len(self._bucket(label))

    def labels(self) -> list[str]:
        """Return all labels that have at least one recorded sample."""
        return [lbl for lbl, buf in self._data.items() if buf]

    def summary(self) -> dict[str, dict[str, float | int]]:
        """Return a summary of all labels.

        Returns:
            A dict mapping each label to a nested dict with keys
            ``p50``, ``p99``, and ``count``::

                {
                    "order_rtt": {"p50": 1.2, "p99": 4.7, "count": 342},
                    "ws_tick":   {"p50": 0.3, "p99": 1.1, "count": 1000},
                }
        """
        result: dict[str, dict[str, float | int]] = {}
        for label, buf in self._data.items():
            if not buf:
                continue
            snapshot = sorted(buf)
            result[label] = {
                "p50": self._percentile(snapshot, 50.0),
                "p99": self._percentile(snapshot, 99.0),
                "count": len(snapshot),
            }
        return result

    def reset(self, label: str | None = None) -> None:
        """Clear recorded samples.

        Args:
            label: When provided, clear only that label.  When ``None``
                   (default), clear **all** labels.
        """
        if label is None:
            self._data.clear()
        elif label in self._data:
            self._data[label].clear()
