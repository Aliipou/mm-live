"""
Core event loop engine — queue-based, event-driven.

Architecture:
    Multiple async PRODUCERS put events onto a single asyncio.Queue.
    One CONSUMER (engine.run) drains the queue and dispatches events.

    Producers:
        ws_producer: WebSocket → BOOK_UPDATE + TRADE events
        timer_producer: 100ms heartbeat → TIMER events

    Consumer:
        handle_event: routes each event to the right handler.
        on_timer: the ONLY place where quoting decisions are made.
            (never quote on every trade — that's overtrading)

This separation ensures:
    - No blocking code on the hot path
    - Rate-limited quote decisions (TIMER, not TRADE)
    - Single source of truth: the queue
    - Easy to add new producers (e.g., news feed, cross-exchange)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from mm_live.analytics.metrics import StrategyMetrics
from mm_live.analytics.pnl import PnLTracker
from mm_live.core.events import EventType, MarketEvent
from mm_live.execution.order_manager import OrderManager
from mm_live.execution.simulator import FillSimulator
from mm_live.feed.binance_ws import BinanceOrderBookFeed, TradeEvent
from mm_live.feed.orderbook import OrderBook
from mm_live.risk.limits import RiskLimits, RiskStatus
from mm_live.signals.fair_value import FairValueSignal
from mm_live.signals.imbalance import OrderFlowImbalance
from mm_live.signals.volatility import DualVolatility
from mm_live.strategy.quoting import AdaptiveQuoteEngine, Quotes

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """All tunable parameters in one place."""

    kalman_process_noise: float = 0.01
    kalman_measurement_noise: float = 0.25
    imbalance_alpha_impact: float = 2.0

    vol_short_halflife: float = 50.0
    vol_long_halflife: float = 300.0
    vol_short_weight: float = 0.7

    gamma: float = 0.05
    k: float = 1.5
    T_horizon: float = 600.0
    max_inventory: float = 0.1
    fill_qty: float = 0.001

    max_drawdown_usd: float = 50.0
    max_spread_usd: float = 200.0

    timer_interval_sec: float = 0.1   # quote decisions every 100ms
    log_interval_sec: float = 5.0


class Engine:
    """
    Event-driven market making engine.

    Usage::

        config = EngineConfig()
        engine = Engine(config)
        await engine.run_with_feed("btcusdt")
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.cfg = config or EngineConfig()

        # Single event queue — the only communication channel
        self.event_queue: asyncio.Queue[MarketEvent] = asyncio.Queue(maxsize=10_000)

        # State
        self.book = OrderBook(symbol="BTCUSDT")
        self._last_trade: TradeEvent | None = None
        self._current_quotes: Quotes | None = None

        # Signal pipeline
        self.fv_signal = FairValueSignal(
            process_noise_var=self.cfg.kalman_process_noise,
            measurement_noise_var=self.cfg.kalman_measurement_noise,
            imbalance_alpha=self.cfg.imbalance_alpha_impact,
        )
        self.vol = DualVolatility(
            short_halflife=self.cfg.vol_short_halflife,
            long_halflife=self.cfg.vol_long_halflife,
            short_weight=self.cfg.vol_short_weight,
        )
        self.imbalance = OrderFlowImbalance()

        # Strategy
        self.strategy = AdaptiveQuoteEngine(
            gamma=self.cfg.gamma,
            k=self.cfg.k,
            T_horizon=self.cfg.T_horizon,
            max_inventory=self.cfg.max_inventory,
        )

        # Execution
        self.executor = FillSimulator(fill_qty=self.cfg.fill_qty)
        self.order_manager = OrderManager(paper_mode=True)

        # Risk
        self.risk = RiskLimits(
            max_inventory_btc=self.cfg.max_inventory,
            max_drawdown_usd=self.cfg.max_drawdown_usd,
            max_spread_usd=self.cfg.max_spread_usd,
        )

        # Analytics
        self.pnl = PnLTracker()
        self.metrics = StrategyMetrics()

        self._n_ticks = 0
        self._last_log = time.monotonic()
        self._running = False

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    async def run_with_feed(self, symbol: str = "btcusdt") -> None:
        """Start all producers and the consumer. Runs until circuit breaker."""
        self._running = True
        feed = BinanceOrderBookFeed(symbol=symbol)

        logger.info("Engine starting — symbol=%s gamma=%.3f k=%.2f T=%.0f",
                    symbol.upper(), self.cfg.gamma, self.cfg.k, self.cfg.T_horizon)

        await asyncio.gather(
            self._ws_producer(feed),
            self._timer_producer(),
            self._consumer(),
        )

    # ------------------------------------------------------------------ #
    # Producers                                                            #
    # ------------------------------------------------------------------ #

    async def _ws_producer(self, feed: BinanceOrderBookFeed) -> None:
        """Drain WebSocket feed into the event queue."""
        import time as _time
        async for book, trade in feed.stream():
            ts = int(_time.time() * 1000)

            if book.is_ready:
                await self.event_queue.put(MarketEvent(
                    type=EventType.BOOK_UPDATE,
                    timestamp_ms=ts,
                    data={"book": book},
                ))

            if trade is not None:
                await self.event_queue.put(MarketEvent(
                    type=EventType.TRADE,
                    timestamp_ms=ts,
                    data={"trade": trade},
                ))

    async def _timer_producer(self) -> None:
        """Emit TIMER events at cfg.timer_interval_sec rate."""
        import time as _time
        while self._running:
            await asyncio.sleep(self.cfg.timer_interval_sec)
            await self.event_queue.put(MarketEvent(
                type=EventType.TICK,
                timestamp_ms=int(_time.time() * 1000),
                data={},
            ))

    # ------------------------------------------------------------------ #
    # Consumer                                                             #
    # ------------------------------------------------------------------ #

    async def _consumer(self) -> None:
        """Drain the queue and dispatch each event."""
        while self._running:
            event = await self.event_queue.get()
            await self._handle_event(event)
            self.event_queue.task_done()

            if self.risk.is_blown:
                logger.error("CIRCUIT BREAKER TRIGGERED — stopping engine")
                self._running = False
                self.order_manager.cancel_all()

    async def _handle_event(self, event: MarketEvent) -> None:
        """Route event to the appropriate handler."""
        if event.type == EventType.BOOK_UPDATE:
            self.book = event.data["book"]

        elif event.type == EventType.TRADE:
            self._last_trade = event.data["trade"]

        elif event.type == EventType.TICK:
            # TIMER: the only place we make quoting decisions
            await self._on_timer(event.timestamp_ms)

        elif event.type == EventType.FILL:
            fill = event.data["fill"]
            self.pnl.record_fill(
                side=fill["side"],
                price=fill["price"],
                qty=fill["qty"],
                fair_value=fill.get("fair_value", 0.0),
            )
            self.metrics.record_fill(fill)

    async def _on_timer(self, ts_ms: int) -> None:
        """
        Make all quoting decisions here — called every 100ms.

        Steps:
            1. Extract current state
            2. Compute signals (fv, vol, imbalance)
            3. Generate quotes via strategy
            4. Run risk check
            5. Update order manager
            6. Simulate fills from last trade
            7. Log periodically
        """
        if not self.book.is_ready:
            return

        mid = self.book.mid
        if mid is None:
            return

        self._n_ticks += 1

        # 1. Signals
        imb = self.imbalance.update(self.book)
        fv = self.fv_signal.update(mid=mid, imbalance=imb)
        sigma = self.vol.update(fv)
        regime = self.vol.regime

        # 2. Strategy
        quotes = self.strategy.compute(
            fair_value=fv,
            sigma=sigma,
            inventory=self.pnl.inventory,
            imbalance=imb,
            regime=regime,
        )
        self._current_quotes = quotes

        # 3. Risk
        pnl_total = self.pnl.total(fv)
        status = self.risk.check(
            inventory=self.pnl.inventory,
            bid=quotes.bid,
            ask=quotes.ask,
            current_pnl=pnl_total,
        )

        if status != RiskStatus.OK:
            self.order_manager.cancel_all()
            return

        # 4. Update order manager (cancel-and-replace if moved)
        self.order_manager.update_quotes(quotes.bid, quotes.ask, self.cfg.fill_qty)

        # 5. Simulate fills from last observed trade
        if self._last_trade is not None:
            fill = self.executor.simulate_fill(self._last_trade, quotes)
            if fill is not None:
                fill["fair_value"] = fv
                await self.event_queue.put(MarketEvent(
                    type=EventType.FILL,
                    timestamp_ms=ts_ms,
                    data={"fill": fill},
                ))
            self._last_trade = None  # consume it

        # 6. Metrics
        self.metrics.record_quote(quotes, sigma, imb)

        # 7. Periodic log
        now = time.monotonic()
        if now - self._last_log >= self.cfg.log_interval_sec:
            logger.info(
                "[%s] mid=%.2f fv=%.2f σ=%.4f imb=%+.3f regime=%s | "
                "bid=%.2f ask=%.2f spread=%.2f | "
                "inv=%+.4f pnl_r=%.2f pnl_u=%.2f | "
                "fills=%d fill_rate=%.3f risk=%s ticks=%d",
                symbol if (symbol := "BTCUSDT") else "?",
                mid, fv, sigma, imb, regime,
                quotes.bid, quotes.ask, quotes.spread,
                self.pnl.inventory,
                self.pnl.realized_pnl,
                self.pnl.unrealized_pnl(fv),
                self.metrics.total_fills,
                self.metrics.fill_rate,
                status.value,
                self._n_ticks,
            )
            self._last_log = now
