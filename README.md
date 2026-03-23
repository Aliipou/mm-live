# mm-live

Event-driven market making system. Connects to Binance and OKX live feeds, estimates fair value with a Kalman filter, models order flow imbalance and dual-timeframe volatility, and generates adaptive Avellaneda-Stoikov quotes with inventory risk control.

Paper trading in Phase 1. Live order execution, latency optimization, and cross-venue arbitrage in Phases 2–4.

---

## How it works

The engine runs three concurrent coroutines:

```
┌─────────────────┐
│  Binance WS     │──► BOOK_UPDATE, TRADE events ──►┐
└─────────────────┘                                  │
                                               asyncio.Queue
┌─────────────────┐                                  │
│  100ms timer    │──────────────── TICK event ──────►│
└─────────────────┘                                  │
                                                     ▼
                                               consumer
                                                     │
                                        on TICK only ▼
                                  signals → strategy → risk → execution
```

Quoting decisions happen **only on TICK events**, not on every trade. This prevents overtrading and decouples market data ingestion from quote logic.

---

## Signal pipeline

**Fair value** — Kalman filter on mid-price, adjusted by order flow imbalance:

```
fv = Kalman(mid) + α · imbalance
```

**Volatility** — two EWMA estimators blended, with regime detection:

```
σ_blend = w · σ_short + (1−w) · σ_long
regime  = HIGH_VOL if σ_short / σ_long > 1.5
```

**Imbalance** — top-N level order flow imbalance, EMA smoothed:

```
OFI = (bid_vol − ask_vol) / (bid_vol + ask_vol)
```

---

## Strategy

Avellaneda-Stoikov (2008) optimal quotes under inventory risk:

```
reservation price   r = fv − q · γ · σ² · T
half-spread         δ = γσ²T/2 + (1/γ) · ln(1 + γ/k)

bid = r − δ
ask = r + δ
```

Three live adaptations:
- **Regime multiplier** — spread widens in high-vol regime
- **Imbalance skew** — reservation price shifts with OFI signal
- **One-sided quoting** — only post the side that reduces inventory at the limit

---

## Risk controls

- Inventory hard cap — quoting halts if `|inventory| ≥ max_inventory`
- Drawdown circuit breaker — engine stops if loss from peak exceeds `max_drawdown_usd`
- Spread sanity check — rejects quotes wider than `max_spread_usd` or tighter than 1 tick
- Quote throttle — no re-post unless price moves ≥ `min_price_move` and ≥ 50ms has elapsed
- Binance rate limiter — token bucket enforcing 1200 weight/min, 10 orders/sec, 100k orders/day

---

## Phases

| | Phase | Description |
|---|---|---|
| ✅ | 1 — Paper trading | Live Binance feed, simulated fills, full P&L attribution |
| ✅ | 2 — Live execution | Signed REST orders, user data stream for real fills, cancel-and-replace |
| ✅ | 3 — Latency | Token bucket rate limiter, quote throttle, HTTPS connection pool, latency tracker |
| ✅ | 4 — Cross-venue | OKX feed, unified order book, cross-venue arb detection and hedge logic |

---

## Quickstart

```bash
pip install -e ".[dev]"
python main.py
```

**Paper trading** (default — no keys needed):

```bash
MM_SYMBOL=btcusdt MM_GAMMA=0.05 python main.py
```

**Live trading** (Phase 2):

```bash
BINANCE_API_KEY=... BINANCE_SECRET=... MM_SYMBOL=btcusdt python main.py
```

| Variable | Default | Description |
|---|---|---|
| `MM_SYMBOL` | `btcusdt` | Trading pair |
| `MM_GAMMA` | `0.05` | Risk aversion (higher = wider spreads) |
| `MM_K` | `1.5` | Order arrival rate sensitivity |
| `MM_T_HORIZON` | `600` | Rolling inventory horizon (ticks) |
| `MM_MAX_INVENTORY` | `0.1` | Max BTC position |
| `BINANCE_API_KEY` | — | Required for live execution |
| `BINANCE_SECRET` | — | Required for live execution |
| `BINANCE_TESTNET` | `false` | Use Binance testnet |

---

## Tests

```bash
pytest tests/ -v
```

215 tests, ~1.6s. No network calls — all I/O is mocked.

---

## Structure

```
src/mm_live/
├── core/
│   ├── engine.py          asyncio.Queue event loop, 3-coroutine architecture
│   ├── events.py          EventType enum, MarketEvent dataclass
│   └── latency.py         Per-label latency tracker, p50/p99
├── feed/
│   ├── binance_ws.py      Binance WebSocket with auto-reconnect
│   ├── okx_ws.py          OKX WebSocket, same interface
│   ├── orderbook.py       L2 order book state machine
│   └── unified_book.py    Multi-venue aggregator, cross-spread detection
├── signals/
│   ├── fair_value.py      Kalman filter + imbalance adjustment
│   ├── imbalance.py       Order flow imbalance, EMA smoothed
│   └── volatility.py      Dual EWMA + regime detection
├── strategy/
│   ├── quoting.py         Adaptive Avellaneda-Stoikov engine
│   └── cross_venue.py     Cross-venue arb signal + hedge logic
├── execution/
│   ├── binance_client.py  HMAC-signed Binance REST client
│   ├── user_stream.py     WebSocket user data stream (fills)
│   ├── order_manager.py   Cancel-and-replace, paper + live modes
│   ├── simulator.py       Paper trading fill simulation
│   ├── rate_limiter.py    Token bucket, Binance rate limits
│   ├── quote_throttle.py  Price-move + time gate
│   └── connection_pool.py HTTPS connection pool
├── analytics/
│   ├── pnl.py             Realized/unrealized P&L, spread capture vs adverse selection
│   └── metrics.py         Fill rate, vol history, imbalance stats
└── risk/
    └── limits.py          Inventory cap, drawdown circuit breaker, spread checks
```

---

## References

- Avellaneda & Stoikov (2008) — *High-frequency trading in a limit order book*
- Glosten & Milgrom (1985) — *Bid, ask and transaction prices in a specialist market*
- Kyle (1985) — *Continuous auctions and insider trading*
- Budish, Cramton & Shim (2015) — *The high-frequency trading arms race*
