# mm-live

Production-grade market making system built on asyncio. Connects to Binance and OKX live feeds, estimates fair value with a Kalman filter and order flow imbalance, models dual-timeframe volatility with regime detection, and generates adaptive Avellaneda-Stoikov quotes with full risk controls.

Four phases from paper trading to cross-venue arbitrage.

---

## Phases

| | Phase | Description |
|---|---|---|
| ✅ | 1 — Paper trading | Live Binance feed, simulated fills, full P&L and spread attribution |
| ✅ | 2 — Live execution | HMAC-signed REST orders, WebSocket user stream, cancel-and-replace |
| ✅ | 3 — Latency | Token bucket rate limiter, quote throttle, HTTPS connection pool |
| ✅ | 4 — Cross-venue | OKX feed, unified order book, cross-venue arb detection |

---

## Install & Run

```bash
pip install -e ".[dev]"

# Paper trading (no keys needed)
python main.py

# Live trading
BINANCE_API_KEY=... BINANCE_SECRET=... python main.py
```

---

## Quick Example

```python
from mm_live.core.engine import Engine, EngineConfig

# All parameters in one place
config = EngineConfig(
    gamma=0.05,           # risk aversion — higher = wider spreads
    k=1.5,                # order arrival sensitivity
    T_horizon=600,        # rolling inventory horizon (ticks)
    max_inventory=0.1,    # max BTC position before one-sided quoting
    max_drawdown_usd=50,  # circuit breaker
)

engine = Engine(config)
await engine.run_with_feed("btcusdt")
```

```python
from mm_live.feed.unified_book import UnifiedBook
from mm_live.strategy.cross_venue import CrossVenueStrategy

# Cross-venue arb detection
book = UnifiedBook()
book.update("binance", binance_orderbook)
book.update("okx", okx_orderbook)

strategy = CrossVenueStrategy(fee_bps=7.0, min_net_spread=5.0)
signal = strategy.check_arb(book)

if signal.exists:
    print(f"Arb: buy {signal.qty} BTC on {signal.buy_venue} @ {signal.buy_price}")
    print(f"     sell on {signal.sell_venue} @ {signal.sell_price}")
    print(f"     net P&L: ${signal.net_spread * signal.qty:.2f}")
```

---

## Architecture

The engine runs three concurrent coroutines on a single `asyncio.Queue`. Quoting decisions happen **only on TICK events** — never on every trade.

```
Binance WS ──► BOOK_UPDATE, TRADE ──►┐
                                      │  asyncio.Queue ──► consumer
100ms timer ──► TICK ────────────────►┘                        │
                                                      on TICK only
                                                               ▼
                                             signals → strategy → risk → execution
```

**Signal pipeline:**

```
fair value  fv = Kalman(mid) + α · OFI
volatility  σ  = w · σ_short + (1−w) · σ_long    (HIGH_VOL if σ_short/σ_long > 1.5)
imbalance   I  = EMA((bid_vol − ask_vol) / (bid_vol + ask_vol))
```

**Avellaneda-Stoikov quotes:**

```
reservation  r = fv − q · γ · σ² · T
half-spread  δ = γσ²T/2 + (1/γ) · ln(1 + γ/k)
```

With regime multiplier, imbalance skew, and one-sided quoting at inventory limits.

---

## Tests

```bash
pytest tests/ -v   # 215 tests, ~1.6s, no network calls
```

---

## References

- Avellaneda & Stoikov (2008). *High-frequency trading in a limit order book.* Quantitative Finance.
- Glosten & Milgrom (1985). *Bid, ask and transaction prices in a specialist market.* Journal of Financial Economics.
- Kyle (1985). *Continuous auctions and insider trading.* Econometrica.
- Budish, Cramton & Shim (2015). *The high-frequency trading arms race.* Quarterly Journal of Economics.
