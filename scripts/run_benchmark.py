"""
run_benchmark.py
================

Collects N ticks from the live Binance feed, runs BenchmarkRunner over all
three strategies (AdaptiveQuoteEngine, FixedSpreadMaker, NaiveMaker), prints a
comparison report with ASCII P&L curves, and saves results to
``benchmark_results.json``.

Usage
-----
::

    # Default: 500 ticks, BTCUSDT, saves to benchmark_results.json
    python scripts/run_benchmark.py

    # Custom tick count and symbol
    python scripts/run_benchmark.py --n-ticks 1000 --symbol ethusdt

    # Write output to a different file
    python scripts/run_benchmark.py --output /tmp/bench.json

Environment variables (optional)
----------------------------------
BENCH_N_TICKS    Number of ticks to collect (default: 500)
BENCH_SYMBOL     Trading pair, lower-case (default: btcusdt)
BENCH_OUTPUT     Output JSON file path (default: benchmark_results.json)
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
from mm_live.research.benchmark import BenchmarkRunner, MarketTick
from mm_live.signals.fair_value import FairValueSignal
from mm_live.signals.imbalance import OrderFlowImbalance
from mm_live.signals.volatility import DualVolatility
from mm_live.strategy.quoting import AdaptiveQuoteEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_benchmark")


# ---------------------------------------------------------------------------
# Async tick collection
# ---------------------------------------------------------------------------


async def collect_ticks(
    symbol: str,
    n_ticks: int,
) -> list[MarketTick]:
    """Stream live Binance data and assemble ``n_ticks`` MarketTick snapshots.

    Each tick is emitted on every timer heartbeat (i.e. after each order-book
    update that carries a non-None mid price).  The signals pipeline mirrors
    the Engine so that ``sigma`` and ``imbalance`` are consistent with what
    AdaptiveQuoteEngine would see in production.

    Parameters
    ----------
    symbol:
        Binance symbol in lower-case (e.g. ``"btcusdt"``).
    n_ticks:
        How many populated MarketTick objects to collect before returning.

    Returns
    -------
    list[MarketTick]
        Ordered list of collected ticks.
    """
    ticks: list[MarketTick] = []

    # Signal pipeline (mirrors Engine.__init__)
    imbalance_signal = OrderFlowImbalance()
    vol_signal = DualVolatility(
        short_halflife=50.0,
        long_halflife=300.0,
        short_weight=0.7,
    )
    fv_signal = FairValueSignal(
        process_noise_var=0.01,
        measurement_noise_var=0.25,
        imbalance_alpha=2.0,
    )

    feed = BinanceOrderBookFeed(symbol=symbol)

    last_trade_price: float | None = None
    last_trade_is_buyer_maker: bool | None = None
    last_log = time.monotonic()
    log_interval = 5.0

    logger.info(
        "Starting tick collection: symbol=%s target=%d ticks",
        symbol.upper(),
        n_ticks,
    )

    async for book, trade in feed.stream():
        if len(ticks) >= n_ticks:
            logger.info("Target tick count reached — stopping feed.")
            break

        # Keep last trade for the tick snapshot.
        if trade is not None:
            last_trade_price = trade.price
            last_trade_is_buyer_maker = trade.is_buyer_maker

        if not book.is_ready:
            continue

        mid = book.mid
        if mid is None:
            continue

        ts_ms = int(time.time() * 1000)

        # Compute signals
        imb = imbalance_signal.update(book)
        fv = fv_signal.update(mid=mid, imbalance=imb)
        sigma = vol_signal.update(fv)

        ticks.append(
            MarketTick(
                timestamp_ms=ts_ms,
                mid=mid,
                fair_value=fv,
                sigma=sigma,
                imbalance=imb,
                trade_price=last_trade_price,
                trade_is_buyer_maker=last_trade_is_buyer_maker,
            )
        )

        now = time.monotonic()
        if now - last_log >= log_interval:
            logger.info(
                "Progress: collected=%d / %d ticks",
                len(ticks),
                n_ticks,
            )
            last_log = now

    logger.info("Collection complete: %d ticks gathered.", len(ticks))
    return ticks


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------


def _results_to_dict(
    results: list,
    *,
    symbol: str,
    n_ticks: int,
) -> dict:
    """Convert BacktestResult list to a JSON-serialisable dictionary."""
    return {
        "meta": {
            "symbol": symbol.upper(),
            "n_ticks_collected": n_ticks,
            "benchmarked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "results": [asdict(r) for r in results],
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect live Binance ticks and compare three market-making strategies."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-ticks",
        type=int,
        default=int(os.environ.get("BENCH_N_TICKS", 500)),
        help="Number of ticks to collect from the live feed.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=os.environ.get("BENCH_SYMBOL", "btcusdt"),
        help="Binance symbol (lower-case).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.environ.get("BENCH_OUTPUT", "benchmark_results.json"),
        help="Path for the JSON results file.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()

    if args.n_ticks < 2:
        logger.error("--n-ticks must be at least 2.")
        sys.exit(1)

    # --- Collect ticks ---
    ticks = await collect_ticks(symbol=args.symbol, n_ticks=args.n_ticks)

    if len(ticks) < 2:
        logger.warning(
            "Only %d tick(s) collected — not enough for a meaningful benchmark.",
            len(ticks),
        )
        sys.exit(1)

    # --- Build engine (same defaults as main.py) ---
    engine = AdaptiveQuoteEngine(
        gamma=0.05,
        k=1.5,
        T_horizon=600.0,
        max_inventory=0.1,
        min_half_spread=0.5,
        imbalance_skew_factor=0.3,
    )

    # --- Run benchmark ---
    runner = BenchmarkRunner(ticks=ticks, fill_qty=0.001)
    results = runner.run_all(engine)

    # --- Print comparison table ---
    runner.print_report(results)

    # --- Print ASCII P&L curves ---
    ascii_chart = runner.print_ascii_pnl(results)
    print(ascii_chart)

    # --- Save to JSON ---
    output_path = Path(args.output)
    payload = _results_to_dict(results, symbol=args.symbol, n_ticks=len(ticks))
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("Results saved to %s", output_path.resolve())


if __name__ == "__main__":
    asyncio.run(_main())
