"""
Tests for mm_live.execution.binance_client.BinanceClient.

All network I/O is mocked via unittest.mock.patch on urllib.request.urlopen
so no real HTTP connections are made.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from mm_live.execution.binance_client import _MAINNET_BASE, _TESTNET_BASE, BinanceClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(body: dict | list, status: int = 200) -> MagicMock:
    """Return a mock context-manager whose .read() yields JSON bytes."""
    raw = json.dumps(body).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_http_error(body: dict, code: int = 400) -> urllib.error.HTTPError:
    raw = json.dumps(body).encode()
    err = urllib.error.HTTPError(
        url="http://x",
        code=code,
        msg="Bad Request",
        hdrs=None,  # type: ignore[arg-type]
        fp=BytesIO(raw),
    )
    return err


def _expected_sig(secret: bytes, params: dict) -> str:
    qs = urllib.parse.urlencode(params)
    return hmac.new(secret, qs.encode(), hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> BinanceClient:
    return BinanceClient(api_key="test_key", secret="test_secret", testnet=False)


@pytest.fixture
def testnet_client() -> BinanceClient:
    return BinanceClient(api_key="tn_key", secret="tn_secret", testnet=True)


# ---------------------------------------------------------------------------
# _sign
# ---------------------------------------------------------------------------

class TestSign:
    def test_sign_produces_correct_hmac(self, client: BinanceClient) -> None:
        params = {"symbol": "BTCUSDT", "side": "BUY", "timestamp": 1700000000000}
        sig = client._sign(params)
        expected = _expected_sig(b"test_secret", params)
        assert sig == expected

    def test_sign_is_hex_string(self, client: BinanceClient) -> None:
        sig = client._sign({"timestamp": 123})
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA-256 hex digest is always 64 chars

    def test_sign_different_params_different_sig(self, client: BinanceClient) -> None:
        sig1 = client._sign({"a": "1"})
        sig2 = client._sign({"a": "2"})
        assert sig1 != sig2

    def test_sign_order_matters(self, client: BinanceClient) -> None:
        # urlencode is order-sensitive for dicts with different insertion order
        # The key point is _sign mirrors what Binance expects
        params = {"symbol": "BTCUSDT", "timestamp": 999}
        sig = client._sign(params)
        expected = _expected_sig(b"test_secret", params)
        assert sig == expected


# ---------------------------------------------------------------------------
# place_order
# ---------------------------------------------------------------------------

class TestPlaceOrder:
    @patch("urllib.request.urlopen")
    def test_place_order_uses_post(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"orderId": 1})
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            client.place_order("BTCUSDT", "BUY", 50000.0, 0.001)
        )
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        assert req.get_method() == "POST"

    @patch("urllib.request.urlopen")
    def test_place_order_correct_url(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"orderId": 2})
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            client.place_order("BTCUSDT", "BUY", 50000.0, 0.001)
        )
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        assert req.full_url == f"{_MAINNET_BASE}/api/v3/order"

    @patch("urllib.request.urlopen")
    def test_place_order_includes_symbol(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"orderId": 3})
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            client.place_order("btcusdt", "BUY", 50000.0, 0.001)
        )
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        body = req.data.decode()
        params = dict(urllib.parse.parse_qsl(body))
        assert params["symbol"] == "BTCUSDT"

    @patch("urllib.request.urlopen")
    def test_place_order_includes_side(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"orderId": 4})
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            client.place_order("BTCUSDT", "sell", 49000.0, 0.002)
        )
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        params = dict(urllib.parse.parse_qsl(req.data.decode()))
        assert params["side"] == "SELL"

    @patch("urllib.request.urlopen")
    def test_place_order_price_formatted(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"orderId": 5})
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            client.place_order("BTCUSDT", "BUY", 50000.123, 0.001)
        )
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        params = dict(urllib.parse.parse_qsl(req.data.decode()))
        assert params["price"] == "50000.12"

    @patch("urllib.request.urlopen")
    def test_place_order_qty_formatted(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"orderId": 6})
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            client.place_order("BTCUSDT", "BUY", 50000.0, 0.001)
        )
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        params = dict(urllib.parse.parse_qsl(req.data.decode()))
        assert params["quantity"] == "0.001000"

    @patch("urllib.request.urlopen")
    def test_place_order_time_in_force_gtc(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"orderId": 7})
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            client.place_order("BTCUSDT", "BUY", 50000.0, 0.001)
        )
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        params = dict(urllib.parse.parse_qsl(req.data.decode()))
        assert params["timeInForce"] == "GTC"

    @patch("urllib.request.urlopen")
    def test_place_order_api_key_header(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"orderId": 8})
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            client.place_order("BTCUSDT", "BUY", 50000.0, 0.001)
        )
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        assert req.get_header("X-mbx-apikey") == "test_key"

    @patch("urllib.request.urlopen")
    def test_place_order_signature_present(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"orderId": 9})
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            client.place_order("BTCUSDT", "BUY", 50000.0, 0.001)
        )
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        params = dict(urllib.parse.parse_qsl(req.data.decode()))
        assert "signature" in params
        assert len(params["signature"]) == 64


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------

class TestCancelOrder:
    @patch("urllib.request.urlopen")
    def test_cancel_order_uses_delete(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"orderId": 123})
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            client.cancel_order("BTCUSDT", 123)
        )
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        assert req.get_method() == "DELETE"

    @patch("urllib.request.urlopen")
    def test_cancel_order_url_contains_path(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"orderId": 123})
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            client.cancel_order("BTCUSDT", 123)
        )
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        assert "/api/v3/order" in req.full_url

    @patch("urllib.request.urlopen")
    def test_cancel_order_includes_order_id(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"orderId": 999})
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            client.cancel_order("BTCUSDT", 999)
        )
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        assert "orderId=999" in req.full_url

    @patch("urllib.request.urlopen")
    def test_cancel_order_api_key_header(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"orderId": 1})
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            client.cancel_order("BTCUSDT", 1)
        )
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        assert req.get_header("X-mbx-apikey") == "test_key"


# ---------------------------------------------------------------------------
# cancel_all_orders
# ---------------------------------------------------------------------------

class TestCancelAllOrders:
    @patch("urllib.request.urlopen")
    def test_cancel_all_orders_uses_delete(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response([{"orderId": 1}, {"orderId": 2}])
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            client.cancel_all_orders("BTCUSDT")
        )
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        assert req.get_method() == "DELETE"

    @patch("urllib.request.urlopen")
    def test_cancel_all_orders_correct_path(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response([])
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            client.cancel_all_orders("BTCUSDT")
        )
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        assert "/api/v3/openOrders" in req.full_url

    @patch("urllib.request.urlopen")
    def test_cancel_all_orders_returns_list(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response([{"orderId": 1}, {"orderId": 2}])
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            client.cancel_all_orders("BTCUSDT")
        )
        assert isinstance(result, list)
        assert len(result) == 2

    @patch("urllib.request.urlopen")
    def test_cancel_all_orders_wraps_dict_response(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        """If Binance returns a dict instead of a list, it should be wrapped."""
        mock_urlopen.return_value = _make_response({"orderId": 5})
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            client.cancel_all_orders("BTCUSDT")
        )
        assert isinstance(result, list)
        assert result[0]["orderId"] == 5


# ---------------------------------------------------------------------------
# get_account
# ---------------------------------------------------------------------------

class TestGetAccount:
    @patch("urllib.request.urlopen")
    def test_get_account_uses_get(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"balances": []})
        import asyncio
        asyncio.get_event_loop().run_until_complete(client.get_account())
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        assert req.get_method() == "GET"

    @patch("urllib.request.urlopen")
    def test_get_account_correct_path(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"balances": []})
        import asyncio
        asyncio.get_event_loop().run_until_complete(client.get_account())
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        assert "/api/v3/account" in req.full_url

    @patch("urllib.request.urlopen")
    def test_get_account_api_key_header(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"balances": []})
        import asyncio
        asyncio.get_event_loop().run_until_complete(client.get_account())
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        assert req.get_header("X-mbx-apikey") == "test_key"

    @patch("urllib.request.urlopen")
    def test_get_account_returns_dict(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        payload = {"balances": [{"asset": "BTC", "free": "0.5"}]}
        mock_urlopen.return_value = _make_response(payload)
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(client.get_account())
        assert result == payload


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @patch("urllib.request.urlopen")
    def test_non_2xx_raises_runtime_error(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.side_effect = _make_http_error({"code": -1121, "msg": "Invalid symbol"}, 400)
        import asyncio
        with pytest.raises(RuntimeError, match="Invalid symbol"):
            asyncio.get_event_loop().run_until_complete(client.get_account())

    @patch("urllib.request.urlopen")
    def test_non_2xx_includes_status_code(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.side_effect = _make_http_error({"code": -2011, "msg": "Unknown order"}, 400)
        import asyncio
        with pytest.raises(RuntimeError, match="HTTP 400"):
            asyncio.get_event_loop().run_until_complete(
                client.cancel_order("BTCUSDT", 999)
            )

    @patch("urllib.request.urlopen")
    def test_error_embeds_binance_message(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.side_effect = _make_http_error({"msg": "Account has insufficient balance"}, 400)
        import asyncio
        with pytest.raises(RuntimeError, match="Account has insufficient balance"):
            asyncio.get_event_loop().run_until_complete(
                client.place_order("BTCUSDT", "BUY", 50000.0, 10.0)
            )


# ---------------------------------------------------------------------------
# Testnet
# ---------------------------------------------------------------------------

class TestTestnet:
    @patch("urllib.request.urlopen")
    def test_testnet_uses_testnet_base_url(self, mock_urlopen: MagicMock, testnet_client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"balances": []})
        import asyncio
        asyncio.get_event_loop().run_until_complete(testnet_client.get_account())
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        assert req.full_url.startswith(_TESTNET_BASE)

    @patch("urllib.request.urlopen")
    def test_mainnet_does_not_use_testnet_url(self, mock_urlopen: MagicMock, client: BinanceClient) -> None:
        mock_urlopen.return_value = _make_response({"balances": []})
        import asyncio
        asyncio.get_event_loop().run_until_complete(client.get_account())
        req: urllib.request.Request = mock_urlopen.call_args[0][0]
        assert req.full_url.startswith(_MAINNET_BASE)
        assert _TESTNET_BASE not in req.full_url

    def test_testnet_flag_stored(self, testnet_client: BinanceClient) -> None:
        assert testnet_client.testnet is True

    def test_mainnet_flag_stored(self, client: BinanceClient) -> None:
        assert client.testnet is False
