# mm-live

Real-time market making system. Connects to Binance WebSocket, computes fair value with a Kalman filter and order flow imbalance, runs dual-timeframe volatility estimation, generates adaptive Avellaneda-Stoikov quotes, and simulates fills in paper trading mode.

## Architecture

```
Binance WS ──► ws_producer ──►┐
                               │  asyncio.Queue  ──► consumer ──► _on_timer
100ms timer ──► timer_producer ►┘                                     │
                                                                       ▼
                                              signals → strategy → risk → execution → analytics
```

**Key design decisions:**
- Single `asyncio.Queue` — one source of truth, no shared mutable state between producers
- Quoting decisions happen only on `TICK` events (100ms timer), not on every trade — prevents overtrading
- Each layer is a stateless processor; all state mutations live in the engine

## Signal Pipeline

| Component | Algorithm |
|---|---|
| Fair Value | Kalman filter on mid-price + `α × OFI` imbalance adjustment |
| Volatility | Dual EWMA (short τ=50, long τ=300), blended sigma, regime detection |
| Imbalance | `(bid_vol − ask_vol) / (bid_vol + ask_vol)` over top-5 levels, EMA smoothed |

## Strategy

Avellaneda-Stoikov optimal quoting:

```
reservation  r = fv − q·γ·σ²·T
half-spread  δ = γσ²T/2 + (1/γ)·ln(1 + γ/k)
```

Adaptations on top of the base model:
- **Regime multiplier**: spread widens in high-vol regime (`short_vol / long_vol > 1.5`)
- **Imbalance skew**: reservation price shifted by imbalance signal
- **One-sided quoting**: only post on the side that reduces inventory at the limit

## Phases

| Phase | Status | Description |
|---|---|---|
| 1 | ✅ Done | Paper trading — live feed, simulated fills, full analytics |
| 2 | 🔜 | Live execution — real orders via Binance REST/WS |
| 3 | 🔜 | Latency optimization — WebSocket co-location, sub-ms quote updates |
| 4 | 🔜 | Cross-venue — multi-exchange arbitrage, unified order book |

## Quickstart

```bash
pip install -e ".[dev]"
python main.py
```

Environment variables (all optional):

| Variable | Default | Description |
|---|---|---|
| `MM_SYMBOL` | `btcusdt` | Trading pair |
| `MM_GAMMA` | `0.05` | Risk aversion coefficient |
| `MM_K` | `1.5` | Order arrival sensitivity |
| `MM_T_HORIZON` | `600` | Rolling horizon (ticks) |
| `MM_MAX_INVENTORY` | `0.1` | Max BTC position |

## Module Map

```
src/mm_live/
├── core/
│   ├── engine.py        # asyncio.Queue event loop, 3-coroutine architecture
│   └── events.py        # EventType enum, MarketEvent dataclass
├── feed/
│   ├── binance_ws.py    # WebSocket feed with auto-reconnect
│   └── orderbook.py     # L2 order book state machine
├── signals/
│   ├── fair_value.py    # Kalman filter + imbalance adjustment
│   ├── imbalance.py     # Order flow imbalance (EMA smoothed)
│   └── volatility.py    # Dual EWMA + regime detection
├── strategy/
│   └── quoting.py       # Adaptive Avellaneda-Stoikov engine
├── execution/
│   ├── simulator.py     # Paper trading fill simulation
│   └── order_manager.py # Cancel-and-replace quote management
├── analytics/
│   ├── pnl.py           # Realized/unrealized P&L + attribution
│   └── metrics.py       # Fill rate, vol history, imbalance stats
└── risk/
    └── limits.py        # Inventory cap, drawdown circuit breaker
```

## Sample Output

```
12:34:01 INFO mm_live.core.engine — Engine starting — symbol=BTCUSDT gamma=0.050 k=1.50 T=600
12:34:06 INFO mm_live.core.engine — mid=67823.45 fv=67821.12 σ_blend=0.0031 imb=0.142
                                     bid=67815.30 ask=67826.94 spread=11.64
                                     inv=0.0010 pnl=0.01 unreal=0.07
                                     fills=3 risk=ok ticks=47
```
