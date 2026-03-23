"""
collect_and_test_edge.py
========================

Connects to the Binance live order-book feed for a configurable duration,
collects (timestamp, imbalance, mid) samples, then runs the OFI edge test
and saves results to ``edge_results.json``.

Usage
-----
::

    # Default: 300 seconds, BTCUSDT, saves to edge_results.json
    python scripts/collect_and_test_edge.py

    # Custom duration and symbol
    python scripts/collect_and_test_edge.py --duration 60 --symbol ethusdt

    # Write output to a different file
    python scripts/collect_and_test_edge.py --output /tmp/my_results.json

Environment variables (optional)
---------------------------------
EDGE_DURATION   Collection window in seconds (default: 300)
EDGE_SYMBOL     Trading pair, lower-case (default: btcusdt)
EDGE_OUTPUT     Output JSON file path (default: edge_results.json)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Ensure the package is importable when running from the repo root.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from mm_live.feed.binance_ws import BinanceOrderBookFeed
from mm_live.research.imbalance_prediction import ImbalanceEdgeTest
from mm_live.signals.imbalance import OrderFlowImbalance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("edge_test")


# ---------------------------------------------------------------------------
# Async collection loop
# ---------------------------------------------------------------------------


async def collect(
    symbol: str,
    duration_s: float,
    horizons_ms: list[int],
) -> ImbalanceEdgeTest:
    """
    Stream live Binance data for ``duration_s`` seconds and feed
    (timestamp, imbalance, mid) tuples into an :class:`ImbalanceEdgeTest`.

    Parameters
    ----------
    symbol:
        Binance symbol in lower-case (e.g. ``"btcusdt"``).
    duration_s:
        How long to collect data (seconds).
    horizons_ms:
        Prediction horizons forwarded to :class:`ImbalanceEdgeTest`.

    Returns
    -------
    ImbalanceEdgeTest
        Populated test object ready for :meth:`~ImbalanceEdgeTest.run_test`.
    """
    edge_test = ImbalanceEdgeTest(horizons_ms=horizons_ms)
    imbalance_signal = OrderFlowImbalance()

    feed = BinanceOrderBookFeed(symbol=symbol)

    deadline = time.monotonic() + duration_s
    n_ticks = 0
    last_log = time.monotonic()
    log_interval = 10.0  # seconds between progress messages

    logger.info(
        "Starting collection: symbol=%s duration=%.0fs horizons=%s",
        symbol.upper(),
        duration_s,
        horizons_ms,
    )

    async for book, _trade in feed.stream():
        if time.monotonic() >= deadline:
            logger.info("Collection window elapsed — stopping feed.")
            break

        if not book.is_ready:
            continue

        mid = book.mid
        if mid is None:
            continue

        ts_ms = int(time.time() * 1000)
        imb = imbalance_signal.update(book)

        edge_test.add_sample(ts_ms, imb, mid)
        n_ticks += 1

        now = time.monotonic()
        if now - last_log >= log_interval:
            remaining = max(0.0, deadline - now)
            logger.info(
                "Progress: ticks=%d completed=%d pending=%d remaining=%.0fs",
                n_ticks,
                edge_test.n_completed,
                edge_test.n_pending,
                remaining,
            )
            last_log = now

    logger.info(
        "Collection complete: ticks=%d completed=%d pending=%d",
        n_ticks,
        edge_test.n_completed,
        edge_test.n_pending,
    )
    return edge_test


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------


def _results_to_dict(edge_test: ImbalanceEdgeTest, results: list, *, symbol: str, duration_s: float) -> dict:
    """Convert edge test output to a JSON-serialisable dictionary."""
    return {
        "meta": {
            "symbol": symbol.upper(),
            "duration_s": duration_s,
            "n_completed_samples": edge_test.n_completed,
            "collected_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "results": [asdict(r) for r in results],
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect live OFI + mid samples from Binance and run the edge test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=float(os.environ.get("EDGE_DURATION", 300)),
        help="Collection window in seconds.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=os.environ.get("EDGE_SYMBOL", "btcusdt"),
        help="Binance symbol (lower-case).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.environ.get("EDGE_OUTPUT", "edge_results.json"),
        help="Path for the JSON results file.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 5000],
        metavar="MS",
        help="Prediction horizons in milliseconds.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()

    # --- Collect ---
    edge_test = await collect(
        symbol=args.symbol,
        duration_s=args.duration,
        horizons_ms=args.horizons,
    )

    # --- Run test ---
    results = edge_test.run_test()

    if not results:
        logger.warning(
            "Not enough completed samples to run the test "
            "(need at least 3, got %d). Try a longer --duration.",
            edge_test.n_completed,
        )
        sys.exit(1)

    # --- Report ---
    edge_test.print_report(results)
    ascii_chart = edge_test.plot_ascii(results)
    print(ascii_chart)

    # --- Save ---
    output_path = Path(args.output)
    payload = _results_to_dict(
        edge_test,
        results,
        symbol=args.symbol,
        duration_s=args.duration,
    )
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("Results saved to %s", output_path.resolve())


if __name__ == "__main__":
    asyncio.run(_main())
