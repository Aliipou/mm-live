# mm-live

Real-time market making system. Live Binance + OKX feeds, Kalman fair value, adaptive Avellaneda-Stoikov quoting, full risk controls, and cross-venue arbitrage.

[![CI](https://github.com/Aliipou/mm-live/actions/workflows/ci.yml/badge.svg)](https://github.com/Aliipou/mm-live/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-336%20passing-brightgreen)](https://github.com/Aliipou/mm-live/actions)

---

## Overview

mm-live is an event-driven market making engine designed for paper trading and live deployment on crypto spot markets. The system is built around a single `asyncio.Queue` that decouples market data ingestion from quoting decisions — quotes are computed on a fixed 100ms timer, not on every trade.

The quant layer implements Avellaneda-Stoikov (2008) optimal quoting with three live adaptations: regime-based spread widening, order flow imbalance skew, and one-sided quoting at inventory limits. An independent research layer provides statistical edge validation — testing whether signals predict future returns before trusting them in production.

---

## Architecture

```
┌──────────────────────┐
│   Binance WebSocket  │──► BOOK_UPDATE + TRADE ──────────┐
└──────────────────────┘                                   │
                                                     asyncio.Queue
┌──────────────────────┐                                   │
│   100ms Timer        │──► TICK ─────────────────────────►│
└──────────────────────┘                                   │
                                                           ▼
                                                      _on_timer()
                                                           │
                              ┌────────────────────────────┼──────────────────────────┐
                              ▼                            ▼                          ▼
                           signals                      strategy                     risk
                    Kalman FV + OFI           Avellaneda-Stoikov             inventory cap
                    dual-vol + regime         imbalance skew                 drawdown breaker
                                              one-sided at limit             spread sanity
```

**Key invariant:** quoting decisions happen exclusively on `TICK` events. Market data updates state only — they never trigger orders directly.

---

## Signal Pipeline

| Signal | Model | Formula |
|---|---|---|
| Fair value | Kalman filter + OFI | `fv = Kalman(mid) + α · imbalance` |
| Volatility | Dual EWMA blend | `σ = w·σ_short + (1−w)·σ_long` |
| Regime | Vol ratio | `HIGH_VOL if σ_short / σ_long > 1.5` |
| Imbalance | OFI, EMA smoothed | `I = EMA((bid_vol − ask_vol) / (bid_vol + ask_vol))` |
| Microprice | Stoikov (2018) | `mp = (ask·bid_sz + bid·ask_sz) / (bid_sz + ask_sz)` |
| Vol clustering | GARCH-inspired | `vol_f = α·\|ret\| + (1−α)·vol_ema` |
| Composite edge | Stacked, Welford-normalized | `score = w1·OFI + w2·(mp−mid) + w3·vol_urgency ∈ [−1,+1]` |

---

## Strategy

Avellaneda-Stoikov (2008) optimal quotes under inventory risk:

```
reservation price   r = fv − q · γ · σ² · T
half-spread         δ = γσ²T/2 + (1/γ) · ln(1 + γ/k)

bid = r − δ · regime_mult · (1 + α·|imbalance|)
ask = r + δ · regime_mult · (1 − α·|imbalance|)
```

At inventory limit: only the side that reduces position is posted.

---

## Phases

| Phase | Status | What |
|---|---|---|
| 1 — Paper trading | ✅ | Live feed, simulated fills, full P&L attribution |
| 2 — Live execution | ✅ | HMAC-signed REST, WebSocket user stream, cancel-and-replace |
| 3 — Latency | ✅ | Token bucket rate limiter, quote throttle, HTTPS connection pool |
| 4 — Cross-venue | ✅ | OKX feed, unified order book, arb signal + hedge logic |

---

## Edge Validation

Before trusting any signal in production, prove it statistically. Every fill is tracked for **markout** — where mid goes after the fill. Negative markout = adverse selection.

```bash
# Test 1: does OFI predict future mid-price movement?
python scripts/collect_and_test_edge.py --duration 300 --symbol btcusdt

# horizon  |   n   |   r   |   R²  | t-stat | p-value | significant
# 100ms    |  847  |  0.12 | 0.014 |  3.41  | 0.0007  |    YES
# 500ms    |  831  |  0.09 | 0.008 |  2.61  | 0.0091  |    YES
# 1000ms   |  812  |  0.05 | 0.002 |  1.41  | 0.1580  |     NO
# 5000ms   |  750  |  0.01 | 0.000 |  0.29  | 0.7720  |     NO

# Test 2: does vol regime predict fill quality?
# Test 3: does the model beat fixed-spread and naive baselines?
python scripts/run_benchmark.py --n-ticks 1000

# Strategy               Fills  Fill%    P&L    Sharpe   MaxDD  WinR%
# AdaptiveQuoteEngine     142   14.2%  +0.31    1.823   0.089  54.1%
# FixedSpreadMaker(±5)     89    8.9%  +0.18    0.941   0.134  51.2%
# NaiveMaker(±0.5)        201   20.1%  -0.22   -0.612   0.287  47.8%

# Test 4: markout analysis — is adverse selection eating edge?
python -c "
from mm_live.research.markout import MarkoutTracker
t = MarkoutTracker()
# ... feed fills + mid ticks
t.print_report(t.compute_stats())
"

# Horizon   N   AvgMkout    Std   %Neg  AS_ratio  Verdict
# 100ms    142    +0.0031  0.021  44.4%     0.062  CLEAN
# 500ms    142    +0.0018  0.034  46.5%     0.036  CLEAN
# 1s       142    -0.0002  0.051  50.0%     0.004  CLEAN
# Net edge after adverse selection: +0.0479 USD per fill
```

---

## Quickstart

```bash
git clone https://github.com/Aliipou/mm-live
cd mm-live
pip install -e ".[dev]"

# Paper trading — no API keys required
python main.py

# Live trading
BINANCE_API_KEY=your_key BINANCE_SECRET=your_secret python main.py
```

| Variable | Default | Description |
|---|---|---|
| `MM_SYMBOL` | `btcusdt` | Trading pair |
| `MM_GAMMA` | `0.05` | Risk aversion — higher = wider spreads |
| `MM_K` | `1.5` | Order arrival rate sensitivity |
| `MM_T_HORIZON` | `600` | Inventory horizon (ticks) |
| `MM_MAX_INVENTORY` | `0.1` | Max BTC position |
| `BINANCE_API_KEY` | — | Required for live execution |
| `BINANCE_SECRET` | — | Required for live execution |
| `BINANCE_TESTNET` | `false` | Use testnet |

---

## Tests

```bash
pytest tests/ -v
# 321 tests, ~3.5s, zero network calls
```

---

## Project Layout

```
mm-live/
├── main.py                         entry point
├── src/mm_live/
│   ├── core/
│   │   ├── engine.py               asyncio.Queue event loop
│   │   ├── events.py               EventType, MarketEvent
│   │   └── latency.py              p50/p99 latency tracker
│   ├── feed/
│   │   ├── binance_ws.py           Binance WebSocket + reconnect
│   │   ├── okx_ws.py               OKX WebSocket + reconnect
│   │   ├── orderbook.py            L2 order book state machine
│   │   └── unified_book.py         multi-venue aggregator
│   ├── signals/
│   │   ├── fair_value.py           Kalman filter + imbalance
│   │   ├── imbalance.py            order flow imbalance (OFI)
│   │   ├── volatility.py           dual EWMA + regime detection
│   │   ├── microprice.py           Stoikov queue-weighted price
│   │   ├── vol_clustering.py       GARCH-inspired vol urgency
│   │   └── composite.py            stacked edge score (Welford-normalized)
│   ├── strategy/
│   │   ├── quoting.py              adaptive Avellaneda-Stoikov
│   │   └── cross_venue.py          arb signal + hedge logic
│   ├── execution/
│   │   ├── binance_client.py       signed REST client
│   │   ├── user_stream.py          WebSocket fill stream
│   │   ├── order_manager.py        cancel-and-replace
│   │   ├── simulator.py            paper fill simulation
│   │   ├── rate_limiter.py         token bucket
│   │   ├── quote_throttle.py       price-move + time gate
│   │   └── connection_pool.py      HTTPS connection pool
│   ├── analytics/
│   │   ├── pnl.py                  realized/unrealized P&L + attribution
│   │   └── metrics.py              fill rate, vol, imbalance history
│   ├── risk/
│   │   └── limits.py               inventory cap, drawdown breaker
│   ├── analytics/
│   │   ├── pnl.py                  realized/unrealized P&L + attribution
│   │   ├── metrics.py              fill rate, vol, imbalance history
│   │   └── capital_efficiency.py   ROI, Sharpe, Sortino, Calmar, hit rate
│   └── research/
│       ├── imbalance_prediction.py OFI → future return regression
│       ├── regime_attribution.py   spread capture vs adverse selection
│       ├── benchmark.py            strategy comparison framework
│       ├── markout.py              post-fill adverse selection tracker
│       ├── multi_asset.py          BTC/ETH × low/normal/high-vol validation
│       └── stress_test.py          7 failure scenarios: flash crash, toxic flow, etc.
└── scripts/
    ├── collect_and_test_edge.py    live OFI edge test
    └── run_benchmark.py            strategy benchmark runner
```

---

## References

- Avellaneda & Stoikov (2008). *High-frequency trading in a limit order book.* Quantitative Finance, 8(3).
- Glosten & Milgrom (1985). *Bid, ask and transaction prices in a specialist market.* Journal of Financial Economics, 14(1).
- Kyle (1985). *Continuous auctions and insider trading.* Econometrica, 53(6).
- Budish, Cramton & Shim (2015). *The high-frequency trading arms race.* Quarterly Journal of Economics, 130(4).
- Stoikov S. (2018). *The micro-price: a high frequency estimator of future prices.* Quantitative Finance, 18(12).
