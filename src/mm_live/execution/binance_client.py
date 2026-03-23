"""
HMAC-SHA256 signed REST client for the Binance Spot API.

Uses only Python's built-in urllib / http.client — no third-party HTTP library.
All network I/O is dispatched through asyncio.get_event_loop().run_in_executor so
the caller's event loop is never blocked.

Mainnet base:  https://api.binance.com
Testnet base:  https://testnet.binance.vision
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_MAINNET_BASE = "https://api.binance.com"
_TESTNET_BASE = "https://testnet.binance.vision"


class BinanceClient:
    """
    Thin, async-friendly wrapper around the Binance Spot REST API.

    All public methods are coroutines.  Internally they use
    ``run_in_executor`` so the underlying urllib calls never block the
    event loop.

    Parameters
    ----------
    api_key:
        Binance API key (``X-MBX-APIKEY`` header).
    secret:
        Binance API secret used for HMAC-SHA256 request signing.
    testnet:
        When ``True`` all requests go to ``testnet.binance.vision``.
    """

    def __init__(self, api_key: str, secret: str, testnet: bool = False) -> None:
        self._api_key = api_key
        self._secret = secret.encode()
        self._base = _TESTNET_BASE if testnet else _MAINNET_BASE
        self.testnet = testnet

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        order_type: str = "LIMIT",
    ) -> dict[str, Any]:
        """
        POST /api/v3/order — place a new limit order (GTC).

        Parameters
        ----------
        symbol:
            E.g. ``"BTCUSDT"``.
        side:
            ``"BUY"`` or ``"SELL"`` (case-insensitive).
        price:
            Limit price as a float; formatted to 2 decimal places.
        qty:
            Order quantity; formatted to 6 decimal places.
        order_type:
            Defaults to ``"LIMIT"``.  Other types (``"MARKET"`` etc.) do
            not require ``price`` / ``timeInForce`` — pass them explicitly
            if needed.

        Returns
        -------
        dict
            Raw Binance order-response JSON.
        """
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": f"{qty:.6f}",
            "timestamp": self._timestamp(),
        }
        if order_type.upper() == "LIMIT":
            params["price"] = f"{price:.2f}"
            params["timeInForce"] = "GTC"

        params["signature"] = self._sign(params)
        return await self._request("POST", "/api/v3/order", params)

    async def cancel_order(self, symbol: str, order_id: int | str) -> dict[str, Any]:
        """
        DELETE /api/v3/order — cancel a single order by exchange order ID.

        Parameters
        ----------
        symbol:
            Instrument symbol, e.g. ``"BTCUSDT"``.
        order_id:
            The numeric ``orderId`` returned by Binance when the order was
            placed.

        Returns
        -------
        dict
            Raw Binance cancellation-response JSON.
        """
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "orderId": str(order_id),
            "timestamp": self._timestamp(),
        }
        params["signature"] = self._sign(params)
        return await self._request("DELETE", "/api/v3/order", params)

    async def cancel_all_orders(self, symbol: str) -> list[dict[str, Any]]:
        """
        DELETE /api/v3/openOrders — cancel every open order for a symbol.

        Parameters
        ----------
        symbol:
            Instrument symbol, e.g. ``"BTCUSDT"``.

        Returns
        -------
        list[dict]
            List of cancellation-response objects from Binance.
        """
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "timestamp": self._timestamp(),
        }
        params["signature"] = self._sign(params)
        result = await self._request("DELETE", "/api/v3/openOrders", params)
        # Binance returns a list here; guard against unexpected shapes.
        if isinstance(result, list):
            return result
        return [result]

    async def get_account(self) -> dict[str, Any]:
        """
        GET /api/v3/account — fetch account balances and permissions.

        Returns
        -------
        dict
            Raw Binance account-information JSON.
        """
        params: dict[str, Any] = {"timestamp": self._timestamp()}
        params["signature"] = self._sign(params)
        return await self._request("GET", "/api/v3/account", params)

    # ------------------------------------------------------------------
    # User-data stream key management (used by UserDataStream)
    # ------------------------------------------------------------------

    async def create_listen_key(self) -> str:
        """POST /api/v3/userDataStream → return listenKey string."""
        result = await self._request("POST", "/api/v3/userDataStream", {})
        return result["listenKey"]

    async def keepalive_listen_key(self, listen_key: str) -> None:
        """PUT /api/v3/userDataStream — extend a listen key's validity."""
        await self._request("PUT", "/api/v3/userDataStream", {"listenKey": listen_key})

    async def delete_listen_key(self, listen_key: str) -> None:
        """DELETE /api/v3/userDataStream — invalidate a listen key."""
        await self._request(
            "DELETE", "/api/v3/userDataStream", {"listenKey": listen_key}
        )

    # ------------------------------------------------------------------
    # Signing helper
    # ------------------------------------------------------------------

    def _sign(self, params: dict[str, Any]) -> str:
        """
        Compute the HMAC-SHA256 signature required by Binance signed endpoints.

        The signature is computed over the URL-encoded query string of
        *params* (excluding the ``signature`` key itself, which must be
        appended afterwards).

        Parameters
        ----------
        params:
            Request parameters **before** the signature is added.

        Returns
        -------
        str
            Hex-encoded HMAC-SHA256 digest.
        """
        query_string = urllib.parse.urlencode(params)
        return hmac.new(self._secret, query_string.encode(), hashlib.sha256).hexdigest()

    # ------------------------------------------------------------------
    # Low-level HTTP helper
    # ------------------------------------------------------------------

    @staticmethod
    def _timestamp() -> int:
        """Current UTC timestamp in milliseconds."""
        return int(time.time() * 1000)

    def _build_request(
        self, method: str, path: str, params: dict[str, Any]
    ) -> urllib.request.Request:
        """Construct a ``urllib.request.Request`` for *method* + *path*."""
        encoded = urllib.parse.urlencode(params)
        headers = {
            "X-MBX-APIKEY": self._api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        }

        if method in ("GET", "DELETE"):
            url = f"{self._base}{path}?{encoded}" if encoded else f"{self._base}{path}"
            data = None
        else:
            url = f"{self._base}{path}"
            data = encoded.encode()

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        return req

    def _do_request(self, method: str, path: str, params: dict[str, Any]) -> Any:
        """
        Execute an HTTP request synchronously (runs in a thread-pool executor).

        Raises ``RuntimeError`` on non-2xx responses, embedding the raw
        Binance error body for diagnostics.
        """
        req = self._build_request(method, path, params)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read()
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            body = exc.read()
            logger.error(
                "Binance REST %s %s → HTTP %d: %s",
                method,
                path,
                exc.code,
                body[:500],
            )
            try:
                payload = json.loads(body)
            except Exception:
                payload = {"raw": body.decode(errors="replace")}
            raise RuntimeError(
                f"Binance {method} {path} failed with HTTP {exc.code}: {payload}"
            ) from exc

    async def _request(
        self, method: str, path: str, params: dict[str, Any]
    ) -> Any:
        """
        Dispatch an HTTP request without blocking the event loop.

        Uses ``asyncio.get_event_loop().run_in_executor`` so urllib's
        blocking I/O runs on the default thread-pool executor.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._do_request, method, path, params
        )
