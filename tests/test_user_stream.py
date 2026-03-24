"""
Tests for mm_live.execution.user_stream.UserDataStream.

All WebSocket and REST I/O is mocked. Async tests use pytest-asyncio with
asyncio_mode = "auto" (set in pyproject.toml).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from mm_live.execution.user_stream import (
    _MAINNET_WS_BASE,
    _TESTNET_WS_BASE,
    UserDataStream,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(testnet: bool = False) -> MagicMock:
    client = MagicMock()
    client.testnet = testnet
    client.create_listen_key = AsyncMock(return_value="testlistenkey123")
    client.keepalive_listen_key = AsyncMock()
    client.delete_listen_key = AsyncMock()
    return client


def _execution_report(exec_type: str = "TRADE", **kwargs) -> bytes:
    base = {
        "e": "executionReport",
        "s": "BTCUSDT",
        "i": 12345,
        "S": "BUY",
        "L": "50000.00",
        "l": "0.001000",
        "X": "FILLED",
        "x": exec_type,
    }
    base.update(kwargs)
    return orjson.dumps(base)


# ---------------------------------------------------------------------------
# _parse_message unit tests (synchronous — no real WS needed)
# ---------------------------------------------------------------------------

class TestParseMessage:
    def test_trade_execution_returns_fill(self) -> None:
        client = _make_client()
        stream = UserDataStream(client)
        raw = _execution_report("TRADE")
        result = stream._parse_message(raw)
        assert result is not None
        assert result["type"] == "fill"

    def test_fill_fields_are_correct(self) -> None:
        client = _make_client()
        stream = UserDataStream(client)
        raw = _execution_report("TRADE", s="ETHUSDT", i=99, S="SELL", L="3000.50", l="1.5", X="PARTIALLY_FILLED")
        result = stream._parse_message(raw)
        assert result is not None
        assert result["symbol"] == "ETHUSDT"
        assert result["order_id"] == 99
        assert result["side"] == "SELL"
        assert result["price"] == pytest.approx(3000.50)
        assert result["qty"] == pytest.approx(1.5)
        assert result["status"] == "PARTIALLY_FILLED"

    def test_non_trade_exectype_returns_none(self) -> None:
        client = _make_client()
        stream = UserDataStream(client)
        for exec_type in ("NEW", "CANCELED", "EXPIRED", "REPLACED"):
            raw = _execution_report(exec_type)
            result = stream._parse_message(raw)
            assert result is None, f"Expected None for execType={exec_type}"

    def test_non_execution_report_event_returns_none(self) -> None:
        client = _make_client()
        stream = UserDataStream(client)
        msg = orjson.dumps({"e": "outboundAccountPosition", "E": 123})
        assert stream._parse_message(msg) is None

    def test_balance_update_returns_none(self) -> None:
        client = _make_client()
        stream = UserDataStream(client)
        msg = orjson.dumps({"e": "balanceUpdate", "a": "BTC", "d": "0.001"})
        assert stream._parse_message(msg) is None

    def test_malformed_json_returns_none(self) -> None:
        client = _make_client()
        stream = UserDataStream(client)
        assert stream._parse_message(b"not valid json!!!") is None

    def test_empty_bytes_returns_none(self) -> None:
        client = _make_client()
        stream = UserDataStream(client)
        assert stream._parse_message(b"") is None

    def test_unknown_event_type_returns_none(self) -> None:
        client = _make_client()
        stream = UserDataStream(client)
        msg = orjson.dumps({"e": "someRandomEvent"})
        assert stream._parse_message(msg) is None


# ---------------------------------------------------------------------------
# listenKey URL inclusion
# ---------------------------------------------------------------------------

class TestListenKeyUrl:
    def test_mainnet_ws_base_used_for_non_testnet(self) -> None:
        client = _make_client(testnet=False)
        stream = UserDataStream(client)
        assert stream._ws_base == _MAINNET_WS_BASE

    def test_testnet_ws_base_used_for_testnet(self) -> None:
        client = _make_client(testnet=True)
        stream = UserDataStream(client)
        assert stream._ws_base == _TESTNET_WS_BASE

    async def test_listen_key_appears_in_ws_url(self) -> None:
        """stream() constructs the WS URL as {ws_base}/{listen_key}."""
        client = _make_client()
        client.create_listen_key = AsyncMock(return_value="mylistenkey999")
        stream = UserDataStream(client)

        connected_urls: list[str] = []

        class _FakeWS:
            def __init__(self, url: str) -> None:
                connected_urls.append(url)

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

        class _FakeConnect:
            """Callable that acts as both the call result AND an async context manager."""
            def __call__(self, url, **kwargs):
                self._ws = _FakeWS(url)
                return self

            async def __aenter__(self):
                return self._ws

            async def __aexit__(self, *_):
                pass

        fake_connect = _FakeConnect()

        with patch("websockets.connect", fake_connect):
            gen = stream.stream()
            task = asyncio.create_task(_drain_one_iteration(gen))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass

        assert any("mylistenkey999" in u for u in connected_urls)


async def _drain_one_iteration(gen):
    """Helper: exhaust one pass of the generator until it loops back."""
    async for _ in gen:
        break


# ---------------------------------------------------------------------------
# stream() yields fill events
# ---------------------------------------------------------------------------

def _make_ws_connect_cm(messages: list) -> object:
    """
    Return a callable that behaves like websockets.connect used as
    ``async with websockets.connect(url) as ws:``.

    The returned object is callable (to accept the URL + kwargs) and is
    also an async context manager that yields a fake WS object.
    """
    from websockets.exceptions import ConnectionClosed

    class _FakeWS:
        def __init__(self):
            self._msgs = iter(messages)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._msgs)
            except StopIteration:
                raise ConnectionClosed(None, None)  # type: ignore[arg-type]

    class _FakeConnect:
        def __call__(self, url, **kwargs):
            return self

        async def __aenter__(self):
            return _FakeWS()

        async def __aexit__(self, *_):
            pass

    return _FakeConnect()


class TestStreamYieldsFills:
    async def test_stream_yields_fill_events(self) -> None:
        client = _make_client()
        stream = UserDataStream(client)

        trade_msg = _execution_report("TRADE", s="BTCUSDT", L="51000.00", l="0.002")
        collected: list[dict] = []

        fake_connect = _make_ws_connect_cm([trade_msg])

        with patch("websockets.connect", fake_connect):
            gen = stream.stream()

            async def collect_one():
                async for event in gen:
                    collected.append(event)
                    break

            task = asyncio.create_task(collect_one())
            await asyncio.wait_for(task, timeout=2.0)

        assert len(collected) == 1
        assert collected[0]["type"] == "fill"
        assert collected[0]["price"] == pytest.approx(51000.0)

    async def test_non_trade_messages_not_yielded(self) -> None:
        client = _make_client()
        stream = UserDataStream(client)

        new_order_msg = _execution_report("NEW")
        trade_msg = _execution_report("TRADE", L="52000.00", l="0.001")

        collected: list[dict] = []
        fake_connect = _make_ws_connect_cm([new_order_msg, trade_msg])

        with patch("websockets.connect", fake_connect):
            gen = stream.stream()

            async def collect_one():
                async for event in gen:
                    collected.append(event)
                    break

            task = asyncio.create_task(collect_one())
            await asyncio.wait_for(task, timeout=2.0)

        assert len(collected) == 1
        assert collected[0]["price"] == pytest.approx(52000.0)


# ---------------------------------------------------------------------------
# Keepalive loop
# ---------------------------------------------------------------------------

class TestKeepaliveLoop:
    async def test_keepalive_calls_keepalive_listen_key(self) -> None:
        client = _make_client()
        client.keepalive_listen_key = AsyncMock()
        stream = UserDataStream(client)

        # Patch sleep so we don't actually wait 30 minutes
        call_count = 0

        async def fast_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", side_effect=fast_sleep):
            with pytest.raises(asyncio.CancelledError):
                await stream._keepalive_loop("testkey")

        # keepalive should have been called at least once
        assert client.keepalive_listen_key.call_count >= 1

    async def test_keepalive_passes_correct_key(self) -> None:
        client = _make_client()
        client.keepalive_listen_key = AsyncMock()
        stream = UserDataStream(client)

        call_count = 0

        async def fast_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", side_effect=fast_sleep):
            with pytest.raises(asyncio.CancelledError):
                await stream._keepalive_loop("myspecifickey")

        client.keepalive_listen_key.assert_called_with("myspecifickey")
