"""
OKX public WebSocket feed for real-time order book and trade data.

Connects to:
  wss://ws.okx.com:8443/ws/v5/public

Subscribes to:
  - books5   (5-level depth snapshots, pushed on every change)
  - trades   (individual trade events)

Emits the same (OrderBook, TradeEvent | None) tuple interface as
BinanceOrderBookFeed so both feeds are drop-in substitutable.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import orjson
import websockets
from websockets.exceptions import ConnectionClosed

from mm_live.feed.binance_ws import TradeEvent
from mm_live.feed.orderbook import OrderBook

logger = logging.getLogger(__name__)

OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

# OKX sends a ping every 30 s and expects a "pong" text frame back.
# We send our own application-level ping to keep NAT sessions alive.
_APP_PING_INTERVAL = 25.0  # seconds


class OKXOrderBookFeed:
    """
    Async iterator that yields (OrderBook, TradeEvent | None) on each OKX message.

    The order book is rebuilt from OKX's ``books5`` snapshot channel, which
    delivers the full 5-level state on every update (no local delta merging
    needed).  Trades arrive on the ``trades`` channel.

    Usage::

        feed = OKXOrderBookFeed("BTC-USDT")
        async for book, trade in feed.stream():
            if book.is_ready:
                print(f"Mid: {book.mid:.2f}, Spread: {book.spread:.2f}")
            if trade:
                print(f"Trade: {trade.price} x {trade.qty}")
    """

    def __init__(self, symbol: str = "BTC-USDT", depth: int = 5) -> None:
        self.symbol = symbol.upper()
        self.depth = depth
        self._book = OrderBook(symbol=self.symbol)

    @property
    def book(self) -> OrderBook:
        return self._book

    # ------------------------------------------------------------------
    # Public streaming interface
    # ------------------------------------------------------------------

    async def stream(self) -> AsyncIterator[tuple[OrderBook, TradeEvent | None]]:
        """Async generator: yields on every WebSocket message from OKX."""
        reconnect_delay = 1.0

        while True:
            try:
                async with websockets.connect(
                    OKX_WS_URL,
                    ping_interval=None,  # We handle pings manually (OKX protocol)
                    open_timeout=10,
                ) as ws:
                    logger.info("Connected to OKX WebSocket: %s", OKX_WS_URL)
                    reconnect_delay = 1.0  # reset backoff on successful connect

                    await self._subscribe(ws)

                    async for raw in self._iter_with_ping(ws):
                        result = self._handle_message(raw)
                        if result is not None:
                            yield result

            except ConnectionClosed as exc:
                logger.warning(
                    "OKX WebSocket closed (%s), reconnecting in %.1fs",
                    exc,
                    reconnect_delay,
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60.0)
            except Exception:
                logger.exception("Unexpected OKX WebSocket error, reconnecting in %.1fs", reconnect_delay)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _subscribe(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Send OKX subscription request for books5 and trades channels."""
        payload = {
            "op": "subscribe",
            "args": [
                {"channel": "books5", "instId": self.symbol},
                {"channel": "trades", "instId": self.symbol},
            ],
        }
        await ws.send(orjson.dumps(payload))
        logger.debug("Sent OKX subscription: %s", payload)

    async def _iter_with_ping(
        self, ws: websockets.WebSocketClientProtocol
    ) -> AsyncIterator[str | bytes]:
        """
        Yield raw frames from ``ws``.

        Interleaves an application-level ``ping`` text frame every
        ``_APP_PING_INTERVAL`` seconds so that OKX doesn't close the
        connection due to inactivity.  OKX responds with ``pong``; we
        silently drop those frames here.
        """
        last_ping = time.monotonic()

        while True:
            now = time.monotonic()
            timeout = _APP_PING_INTERVAL - (now - last_ping)

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(timeout, 0.1))
            except TimeoutError:
                # Time to send a keepalive ping.
                await ws.send("ping")
                last_ping = time.monotonic()
                continue

            # OKX acknowledges our ping with a plain "pong" text frame.
            if raw == "pong":
                last_ping = time.monotonic()
                continue

            yield raw

    def _handle_message(
        self, raw: str | bytes
    ) -> tuple[OrderBook, TradeEvent | None] | None:
        """
        Parse a raw OKX WebSocket frame.

        Returns ``(OrderBook, TradeEvent | None)`` or ``None`` if the frame
        does not carry actionable market data (e.g. subscription confirmations).
        """
        try:
            msg: dict[str, Any] = orjson.loads(raw)
        except Exception:
            logger.debug("Failed to parse OKX message: %.100s", raw)
            return None

        # Subscription / error events carry an "event" key – skip them.
        if "event" in msg:
            event = msg["event"]
            if event == "error":
                logger.error("OKX subscription error: %s", msg)
            else:
                logger.debug("OKX event: %s", msg)
            return None

        channel: str = msg.get("arg", {}).get("channel", "")
        data_list: list[dict[str, Any]] = msg.get("data", [])

        if not data_list:
            return None

        trade: TradeEvent | None = None

        if channel == "books5":
            # OKX books5 delivers full 5-level snapshots each time.
            # data[0] keys: bids, asks, ts, instId, seqId
            data = data_list[0]
            self._apply_okx_book(data)

        elif channel == "trades":
            data = data_list[0]
            trade = TradeEvent(
                price=float(data["px"]),
                qty=float(data["sz"]),
                # OKX side: "sell" means the taker sold → buyer was maker
                is_buyer_maker=(data.get("side", "") == "sell"),
                timestamp_ms=int(data["ts"]),
            )

        else:
            # Unknown channel – still return the current book state so callers
            # that listen to all frames don't miss a heartbeat.
            return None

        return (self._book, trade)

    def _apply_okx_book(self, data: dict[str, Any]) -> None:
        """
        Replace the order book state from an OKX books5 snapshot.

        OKX format:
          ``{"bids": [["price", "qty", "0", "n"], ...], "asks": [...], "ts": "..."}``

        Unlike Binance delta updates, books5 is always a full snapshot so we
        clear and replace rather than merge.
        """
        new_bids: dict[float, float] = {}
        new_asks: dict[float, float] = {}

        for level in data.get("bids", []):
            price, qty = float(level[0]), float(level[1])
            if qty > 0.0:
                new_bids[price] = qty

        for level in data.get("asks", []):
            price, qty = float(level[0]), float(level[1])
            if qty > 0.0:
                new_asks[price] = qty

        self._book.bids = new_bids
        self._book.asks = new_asks

        ts = data.get("ts")
        if ts is not None:
            self._book.last_update_id = int(ts)
