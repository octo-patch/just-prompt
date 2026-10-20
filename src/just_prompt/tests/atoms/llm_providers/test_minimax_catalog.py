"""Exercise MiniMax discovery and public prompt routing with real SDK payloads."""

import json

import httpx
import pytest

from just_prompt.atoms.llm_providers import minimax
from just_prompt.molecules.list_models import list_models
from just_prompt.molecules.prompt import prompt


@pytest.fixture
def sdk_transport(monkeypatch):
    requests = []
    models = ["MiniMax-M2.7", "MiniMax-M3", "custom-model", "MiniMax-M3.1-Flash"]

    def handle(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={
                "object": "list", "has_more": False,
                "data": [{"id": model, "object": "model", "type": "model",
                          "display_name": model, "created_at": "2026-01-01T00:00:00Z"}
                         for model in models],
            })
        body = json.loads(request.content)
        if request.url.path.endswith("/chat/completions"):
            response = {"id": "test", "object": "chat.completion", "created": 0,
                        "model": body["model"], "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop"}]}
        else:
            response = {"id": "test", "type": "message", "role": "assistant",
                        "model": body["model"], "content": [{"type": "text", "text": "ok"}],
                        "stop_reason": "end_turn", "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1}}
        return httpx.Response(200, json=response)

    transport = httpx.MockTransport(handle)
    openai_class = minimax.OpenAI
    anthropic_class = minimax.anthropic.Anthropic
    monkeypatch.setattr(minimax, "OpenAI", lambda **kwargs: openai_class(
        **kwargs, http_client=httpx.Client(transport=transport)))
    monkeypatch.setattr(minimax.anthropic, "Anthropic", lambda **kwargs: anthropic_class(
        **kwargs, http_client=httpx.Client(transport=transport)))
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    for key in ("MINIMAX_OPENAI_BASE_URL", "MINIMAX_ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    return requests, models


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("region,host", [
    ("global_en", "api.minimax.io"), ("cn_zh", "api.minimaxi.com"),
])
@pytest.mark.parametrize("prefix", ["minimax", "m"])
def test_catalog_filters_and_orders_sdk_response(monkeypatch, sdk_transport, protocol, region, host, prefix):
    requests, models = sdk_transport
    monkeypatch.setenv("MINIMAX_PROTOCOL", protocol)
    monkeypatch.setenv("MINIMAX_REGION", region)
    assert list_models(prefix) == ["MiniMax-M3.1-Flash", "MiniMax-M3"]
    assert requests[0].url.host == host
    models.remove("MiniMax-M3.1-Flash")
    assert list_models(prefix) == ["MiniMax-M3"]


@pytest.mark.parametrize("protocol,path", [
    ("openai", "/v1/chat/completions"), ("anthropic", "/anthropic/v1/messages"),
])
@pytest.mark.parametrize("region,host", [
    ("global_en", "api.minimax.io"), ("cn_zh", "api.minimaxi.com"),
])
@pytest.mark.parametrize("prefix", ["minimax", "m"])
@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "custom-model"])
@pytest.mark.parametrize("saved_default", [False, True])
def test_public_prompt_preserves_model_at_http_boundary(
    monkeypatch, sdk_transport, protocol, path, region, host, prefix, model, saved_default
):
    requests, _ = sdk_transport
    monkeypatch.setenv("MINIMAX_PROTOCOL", protocol)
    monkeypatch.setenv("MINIMAX_REGION", region)
    selected = f"{prefix}:{model}"
    monkeypatch.setenv("DEFAULT_MODELS", selected)
    assert prompt("Hello", None if saved_default else [selected]) == ["ok"]
    assert len(requests) == 1  # No discovery or correction request for explicit IDs.
    request = requests[0]
    assert request.url.host == host
    assert request.url.path == path
    expected = {"model": model, "messages": [{"role": "user", "content": "Hello"}]}
    if protocol == "anthropic":
        expected["max_tokens"] = 4096
    assert json.loads(request.content) == expected


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "MiniMax-M3"])
def test_structured_content_is_preserved(monkeypatch, sdk_transport, protocol, model):
    requests, _ = sdk_transport
    monkeypatch.setenv("MINIMAX_PROTOCOL", protocol)
    content = [{"type": "text", "text": "Describe this"}]
    if protocol == "openai":
        content.extend([
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            {"type": "video_url", "video_url": {"url": "mm_file://video-id", "fps": 2}},
        ])
    else:
        content.append({"type": "image", "source": {
            "type": "url", "url": "https://example.com/image.png",
        }})
    assert prompt(content, [f"m:{model}"]) == ["ok"]
    assert len(requests) == 1
    assert json.loads(requests[0].content)["messages"][0]["content"] == content


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_custom_endpoint_keeps_explicit_model(monkeypatch, sdk_transport, protocol):
    requests, _ = sdk_transport
    monkeypatch.setenv("MINIMAX_PROTOCOL", protocol)
    monkeypatch.setenv("MINIMAX_OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("MINIMAX_ANTHROPIC_BASE_URL", "https://gateway.example/anthropic")
    assert prompt("Hello", ["m:custom-model"]) == ["ok"]
    assert len(requests) == 1
    assert requests[0].url.host == "gateway.example"
    assert json.loads(requests[0].content)["model"] == "custom-model"
