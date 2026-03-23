"""HTTP connection pool for Binance REST API (reuses TCP connections)."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import http.client
import json
import time
import urllib.parse
from typing import Any


class BinanceConnectionPool:
    """A pool of persistent HTTPS connections to the Binance REST API.

    Reusing ``http.client.HTTPSConnection`` objects avoids the TLS handshake
    overhead on every request, which is a meaningful latency win for
    high-frequency order operations.

    Connections are stored in an :class:`asyncio.Queue`.  A caller checks
    out a connection, uses it, then returns it.  If the connection has gone
    stale the pool transparently replaces it.

    Usage::

        pool = BinanceConnectionPool(host="api.binance.com", maxsize=10)
        data = await pool.request("GET", "/api/v3/ticker/price",
                                  params={"symbol": "BTCUSDT"})
        await pool.close()

    Or as an async context manager::

        async with BinanceConnectionPool("api.binance.com") as pool:
            data = await pool.request("GET", "/api/v3/ticker/price",
                                      params={"symbol": "BTCUSDT"})
    """

    _TESTNET_HOST: str = "testnet.binance.vision"
    _MAINNET_HOST: str = "api.binance.com"

    def __init__(
        self,
        host: str = "",
        maxsize: int = 10,
        testnet: bool = False,
    ) -> None:
        """Initialise the pool.

        Args:
            host: Override the target hostname.  When empty the host is
                  chosen based on *testnet*.
            maxsize: Maximum number of pooled connections.
            testnet: When ``True`` and *host* is empty, connect to the
                     Binance testnet instead of the live API.
        """
        if not host:
            host = self._TESTNET_HOST if testnet else self._MAINNET_HOST

        self._host: str = host
        self._maxsize: int = maxsize
        self._testnet: bool = testnet

        # Pre-populate the queue with *maxsize* lazy connections.
        self._pool: asyncio.Queue[http.client.HTTPSConnection] = asyncio.Queue(
            maxsize=maxsize
        )
        for _ in range(maxsize):
            self._pool.put_nowait(self._make_connection())

        self._closed: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_connection(self) -> http.client.HTTPSConnection:
        """Create a new (unconnected) HTTPS connection object."""
        conn = http.client.HTTPSConnection(self._host, timeout=10)
        return conn

    @staticmethod
    def _sign(params: dict[str, Any], secret: str) -> dict[str, Any]:
        """Append a Binance HMAC-SHA256 signature to *params* (mutates a copy).

        Args:
            params: Query/body parameters including ``timestamp``.
            secret: The API secret key.

        Returns:
            A new dict with the ``signature`` field appended.
        """
        signed: dict[str, Any] = dict(params)
        query_string = urllib.parse.urlencode(signed)
        signature = hmac.new(
            secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        signed["signature"] = signature
        return signed

    def _build_url(
        self,
        path: str,
        params: dict[str, Any] | None,
        signed: bool,
        secret: str,
    ) -> str:
        """Build the full request path including query string."""
        effective: dict[str, Any] = dict(params) if params else {}

        if signed:
            effective["timestamp"] = int(time.time() * 1000)
            effective = self._sign(effective, secret)

        if effective:
            return f"{path}?{urllib.parse.urlencode(effective)}"
        return path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        api_key: str = "",
        secret: str = "",
    ) -> dict[str, Any]:
        """Send an HTTP request and return the parsed JSON response.

        Connections are checked out from the pool, used, and returned.  On
        any connection-level error the stale connection is discarded and a
        fresh one is returned to the pool for the next caller.

        Args:
            method: HTTP verb (``"GET"``, ``"POST"``, ``"DELETE"``).
            path:   API path, e.g. ``"/api/v3/order"``.
            params: Query parameters (GET) or body parameters (POST/DELETE).
            signed: When ``True`` a timestamp and HMAC-SHA256 signature are
                    appended, and *api_key* is sent in the ``X-MBX-APIKEY``
                    header.
            api_key: Binance API key (required when *signed* is ``True``).
            secret:  Binance secret key (required when *signed* is ``True``).

        Returns:
            Parsed JSON response as a Python dict.

        Raises:
            RuntimeError: If the pool has been closed.
            http.client.HTTPException: On low-level HTTP errors.
            json.JSONDecodeError: If the response body is not valid JSON.
        """
        if self._closed:
            raise RuntimeError("BinanceConnectionPool has been closed")

        url = self._build_url(path, params, signed, secret)

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if signed and api_key:
            headers["X-MBX-APIKEY"] = api_key

        # For POST/DELETE the params go in the body; for GET they're in the URL.
        body: str | None = None
        if method.upper() in {"POST", "DELETE", "PUT"} and params:
            body_params: dict[str, Any] = dict(params)
            if signed:
                body_params["timestamp"] = int(time.time() * 1000)
                body_params = self._sign(body_params, secret)
            body = urllib.parse.urlencode(body_params)
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            url = path  # params are in body, not URL

        conn = await self._pool.get()
        try:
            # Use a thread-pool executor so the blocking socket I/O does not
            # stall the event loop.
            loop = asyncio.get_running_loop()
            response_data = await loop.run_in_executor(
                None,
                lambda: self._send(conn, method, url, headers, body),
            )
        except Exception:
            # Replace broken connection.
            try:
                conn.close()
            except Exception:
                pass
            conn = self._make_connection()
            raise
        finally:
            await self._pool.put(conn)

        return response_data

    def _send(
        self,
        conn: http.client.HTTPSConnection,
        method: str,
        url: str,
        headers: dict[str, str],
        body: str | None,
    ) -> dict[str, Any]:
        """Blocking helper executed in a thread-pool executor."""
        try:
            conn.request(method, url, body=body, headers=headers)
            response = conn.getresponse()
            raw = response.read()
        except (http.client.HTTPException, OSError):
            # Force reconnect on the next attempt.
            try:
                conn.close()
            except Exception:
                pass
            conn.connect()
            conn.request(method, url, body=body, headers=headers)
            response = conn.getresponse()
            raw = response.read()

        return json.loads(raw.decode("utf-8"))

    async def close(self) -> None:
        """Close all pooled connections and mark the pool as shut down."""
        self._closed = True
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except asyncio.QueueEmpty:
                break

    # ------------------------------------------------------------------
    # Async context-manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> BinanceConnectionPool:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
