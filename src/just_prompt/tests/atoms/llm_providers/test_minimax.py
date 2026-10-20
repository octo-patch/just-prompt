"""Tests for the MiniMax provider."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from just_prompt.atoms.llm_providers import minimax


class _CaptureHandler(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, format, *args):
        return

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        self.requests.append(
            {"path": self.path, "body": json.loads(body.decode("utf-8"))}
        )

        if self.path.endswith("/chat/completions"):
            response = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OpenAI response"},
                        "finish_reason": "stop",
                    }
                ],
            }
        elif self.path.endswith("/messages"):
            response = {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "MiniMax-M3",
                "content": [{"type": "text", "text": "Anthropic response"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        else:
            self.send_error(404)
            return

        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def capture_server():
    _CaptureHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


def test_endpoint_configuration_supports_both_regions(monkeypatch):
    monkeypatch.setenv("MINIMAX_REGION", "global_en")
    assert minimax._endpoint_config() == {
        "openai_base_url": "https://api.minimax.io/v1",
        "anthropic_base_url": "https://api.minimax.io/anthropic",
    }

    monkeypatch.setenv("MINIMAX_REGION", "cn_zh")
    assert minimax._endpoint_config() == {
        "openai_base_url": "https://api.minimaxi.com/v1",
        "anthropic_base_url": "https://api.minimaxi.com/anthropic",
    }


def test_anthropic_base_url_requires_anthropic_path(monkeypatch):
    monkeypatch.setenv("MINIMAX_ANTHROPIC_BASE_URL", "https://example.test/v1")
    with pytest.raises(ValueError, match="end with /anthropic"):
        minimax._endpoint_config()


def test_openai_request_capture(monkeypatch, capture_server):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setenv("MINIMAX_PROTOCOL", "openai")
    monkeypatch.setenv("MINIMAX_OPENAI_BASE_URL", f"{capture_server}/v1")

    assert minimax.prompt("Hello", "MiniMax-M3") == "OpenAI response"
    request = _CaptureHandler.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["body"]["model"] == "MiniMax-M3"


def test_anthropic_request_capture(monkeypatch, capture_server):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setenv("MINIMAX_PROTOCOL", "anthropic")
    monkeypatch.setenv("MINIMAX_ANTHROPIC_BASE_URL", f"{capture_server}/anthropic")

    assert minimax.prompt("Hello", "MiniMax-M3") == "Anthropic response"
    request = _CaptureHandler.requests[0]
    assert request["path"] == "/anthropic/v1/messages"
    assert request["body"]["model"] == "MiniMax-M3"


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_list_models_fallback_contains_target_models(monkeypatch, protocol):
    monkeypatch.setenv("MINIMAX_PROTOCOL", protocol)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    assert minimax.list_models() == ["MiniMax-M3.1-Flash", "MiniMax-M3"]


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_custom_gateway_does_not_guess_offline_models(monkeypatch, protocol):
    monkeypatch.setenv("MINIMAX_PROTOCOL", protocol)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("MINIMAX_OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("MINIMAX_ANTHROPIC_BASE_URL", "https://gateway.example/anthropic")
    assert minimax.list_models() == []
