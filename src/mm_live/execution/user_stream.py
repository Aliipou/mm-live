"""
WebSocket user-data stream for real-time fill and order-status notifications.

Flow
----
1. Obtain a ``listenKey`` via POST /api/v3/userDataStream (through
   :class:`~mm_live.execution.binance_client.BinanceClient`).
2. Open a WebSocket connection to ``wss://stream.binance.com:9443/ws/{listenKey}``.
3. Yield normalised fill-event dicts for every ``executionReport`` message
   whose ``execType`` is ``TRADE`` (i.e. a real fill).
4. Renew the ``listenKey`` every 30 minutes (Binance invalidates it after 60
   minutes of inactivity).
5. Reconnect automatically on any WebSocket disconnect.

Testnet URLs
------------
- REST base: ``https://testnet.binance.vision``
- WS base:   ``wss://testnet.binance.vision/ws/{listenKey}``
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import orjson
import websockets
from websockets.exceptions import ConnectionClosed

from mm_live.execution.binance_client import BinanceClient

logger = logging.getLogger(__name__)

_MAINNET_WS_BASE = "wss://stream.binance.com:9443/ws"
_TESTNET_WS_BASE = "wss://testnet.binance.vision/ws"

# Binance recommends keepalive every 30 min (key expires after 60 min idle).
_KEEPALIVE_INTERVAL_S: int = 30 * 60


class UserDataStream:
    """
    Async generator that yields normalised fill events from Binance's
    private user-data WebSocket stream.

    Parameters
    ----------
    client:
        An authenticated :class:`~mm_live.execution.binance_client.BinanceClient`
        instance.  It is used to obtain / renew the ``listenKey`` and is
        expected to carry the correct ``testnet`` flag.

    Yields
    ------
    dict
        Fill-event dict with keys:

        - ``type``     → ``"fill"``
        - ``symbol``   → e.g. ``"BTCUSDT"``
        - ``order_id`` → Binance numeric order ID (int)
        - ``side``     → ``"BUY"`` or ``"SELL"``
        - ``price``    → fill price (float)
        - ``qty``      → filled quantity (float)
        - ``status``   → Binance order status string, e.g. ``"FILLED"``,
          ``"PARTIALLY_FILLED"``
    """

    def __init__(self, client: BinanceClient) -> None:
        self._client = client
        self._ws_base = _TESTNET_WS_BASE if client.testnet else _MAINNET_WS_BASE
        self._listen_key: str | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """
        Async generator that continuously yields fill events.

        Handles:
        - Initial ``listenKey`` creation.
        - Periodic keepalive (every 30 minutes) via a background task.
        - Auto-reconnect on WebSocket disconnect or any unexpected error.
        """
        reconnect_delay: float = 1.0

        while True:
            keepalive_task: asyncio.Task[None] | None = None
            try:
                self._listen_key = await self._client.create_listen_key()
                logger.info("Obtained listenKey: %.20s…", self._listen_key)

                keepalive_task = asyncio.create_task(
                    self._keepalive_loop(self._listen_key)
                )

                url = f"{self._ws_base}/{self._listen_key}"
                async with websockets.connect(url, ping_interval=20) as ws:
                    logger.info("UserDataStream connected: %s", url)
                    reconnect_delay = 1.0  # reset after successful connect

                    async for raw in ws:
                        event = self._parse_message(raw)
                        if event is not None:
                            yield event

            except ConnectionClosed as exc:
                logger.warning(
                    "UserDataStream disconnected (%s), reconnecting in %.1fs",
                    exc,
                    reconnect_delay,
                )
            except Exception:
                logger.exception(
                    "UserDataStream unexpected error, reconnecting in %.1fs",
                    reconnect_delay,
                )
            finally:
                if keepalive_task is not None and not keepalive_task.done():
                    keepalive_task.cancel()
                    try:
                        await keepalive_task
                    except asyncio.CancelledError:
                        pass

            await asyncio.sleep(reconnect_delay)
            # Exponential back-off, capped at 60 s.
            reconnect_delay = min(reconnect_delay * 2, 60.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _keepalive_loop(self, listen_key: str) -> None:
        """
        Background task: send a PUT keepalive to Binance every 30 minutes.

        Runs until cancelled (typically when the WebSocket closes and we
        reconstruct both the key and this task).
        """
        while True:
            await asyncio.sleep(_KEEPALIVE_INTERVAL_S)
            try:
                await self._client.keepalive_listen_key(listen_key)
                logger.debug("listenKey keepalive sent")
            except Exception:
                logger.warning("listenKey keepalive failed — will reconnect", exc_info=True)
                # Raise so the outer loop's ``except`` clause handles reconnect.
                raise

    def _parse_message(self, raw: str | bytes) -> dict[str, Any] | None:
        """
        Parse a raw WebSocket frame and return a normalised fill dict, or
        ``None`` if the message is not a fill event we care about.

        Binance ``executionReport`` events with ``execType == "TRADE"``
        represent actual fills (full or partial).
        """
        try:
            msg: dict[str, Any] = orjson.loads(raw)
        except Exception:
            logger.debug("UserDataStream: failed to parse frame: %.120s", raw)
            return None

        event_type: str = msg.get("e", "")

        if event_type != "executionReport":
            # Could be outboundAccountPosition, balanceUpdate, etc.
            logger.debug("UserDataStream: ignored event type %r", event_type)
            return None

        exec_type: str = msg.get("x", "")
        if exec_type != "TRADE":
            # NEW, CANCELED, EXPIRED — not a fill.
            return None

        return {
            "type": "fill",
            "symbol": msg.get("s", ""),
            "order_id": msg.get("i"),          # Binance numeric order ID
            "side": msg.get("S", ""),           # "BUY" or "SELL"
            "price": float(msg.get("L", 0)),    # last executed price
            "qty": float(msg.get("l", 0)),      # last executed quantity
            "status": msg.get("X", ""),         # FILLED / PARTIALLY_FILLED
        }
