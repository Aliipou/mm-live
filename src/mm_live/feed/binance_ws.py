"""
Binance WebSocket feed for real-time order book and trade data.

Connects to:
  wss://stream.binance.com:9443/stream?streams=btcusdt@depth@100ms/btcusdt@trade

Emits:
  - OrderBook updates (depth stream)
  - Trade events (price, qty, side)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import orjson
import websockets
from websockets.exceptions import ConnectionClosed

from mm_live.feed.orderbook import OrderBook

logger = logging.getLogger(__name__)

BINANCE_WS_URL = (
    "wss://stream.binance.com:9443/stream"
    "?streams={symbol}@depth@100ms/{symbol}@trade"
)


@dataclass(frozen=True)
class TradeEvent:
    """A single trade from the exchange."""

    price: float
    qty: float
    is_buyer_maker: bool  # True = sell-initiated (buyer was maker)
    timestamp_ms: int


class BinanceOrderBookFeed:
    """
    Async iterator that yields (OrderBook, TradeEvent | None) on each message.

    Usage::

        feed = BinanceOrderBookFeed("btcusdt")
        async for book, trade in feed.stream():
            if book.is_ready:
                print(f"Mid: {book.mid:.2f}, Spread: {book.spread:.2f}")
            if trade:
                print(f"Trade: {trade.price} x {trade.qty}")
    """

    def __init__(self, symbol: str = "btcusdt", reconnect_delay: float = 1.0) -> None:
        self.symbol = symbol.lower()
        self.reconnect_delay = reconnect_delay
        self._book = OrderBook(symbol=symbol.upper())

    @property
    def book(self) -> OrderBook:
        return self._book

    async def stream(self) -> AsyncIterator[tuple[OrderBook, TradeEvent | None]]:
        """Async generator: yields on every WebSocket message."""
        url = BINANCE_WS_URL.format(symbol=self.symbol)

        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    logger.info("Connected to Binance WebSocket: %s", url)
                    async for raw in ws:
                        result = self._handle_message(raw)
                        if result is not None:
                            yield result

            except ConnectionClosed as exc:
                logger.warning("WebSocket closed (%s), reconnecting in %.1fs", exc, self.reconnect_delay)
                await asyncio.sleep(self.reconnect_delay)
            except Exception:
                logger.exception("Unexpected WebSocket error, reconnecting")
                await asyncio.sleep(self.reconnect_delay)

    def _handle_message(
        self, raw: str | bytes
    ) -> tuple[OrderBook, TradeEvent | None] | None:
        """Parse a raw WebSocket message. Returns None if unrecognised."""
        try:
            msg: dict[str, Any] = orjson.loads(raw)
        except Exception:
            logger.debug("Failed to parse message: %.100s", raw)
            return None

        stream: str = msg.get("stream", "")
        data: dict[str, Any] = msg.get("data", {})

        trade: TradeEvent | None = None

        if "@depth" in stream:
            self._book.apply_update(data)

        elif "@trade" in stream:
            trade = TradeEvent(
                price=float(data["p"]),
                qty=float(data["q"]),
                is_buyer_maker=bool(data["m"]),
                timestamp_ms=int(data["T"]),
            )

        return (self._book, trade)
