"""Tests for the MiniMax provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from just_prompt.atoms.llm_providers import minimax


def _openai_response(text="MiniMax response"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


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


@pytest.mark.parametrize("thinking_mode", ["adaptive", "disabled"])
def test_thinking_mode_is_sent_in_extra_body(monkeypatch, thinking_mode):
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_response()
    monkeypatch.setattr(minimax, "_openai_client", lambda: client)
    monkeypatch.setenv("MINIMAX_PROTOCOL", "openai")

    assert (
        minimax.prompt("Explain the result", f"MiniMax-M3:{thinking_mode}")
        == "MiniMax response"
    )
    client.chat.completions.create.assert_called_once_with(
        model="MiniMax-M3",
        messages=[{"role": "user", "content": "Explain the result"}],
        extra_body={"thinking": {"type": thinking_mode}},
    )


def test_image_and_video_content_parts_are_forwarded(monkeypatch):
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_response("A short scene")
    monkeypatch.setattr(minimax, "_openai_client", lambda: client)
    monkeypatch.setenv("MINIMAX_PROTOCOL", "openai")
    content = [
        {"type": "text", "text": "Describe the scene"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/frame.png",
                "detail": "high",
            },
        },
        {
            "type": "video_url",
            "video_url": {
                "url": "mm_file://video-id",
                "detail": "default",
                "fps": 2,
            },
        },
    ]

    assert minimax.prompt(content, "MiniMax-M3") == "A short scene"
    client.chat.completions.create.assert_called_once_with(
        model="MiniMax-M3",
        messages=[{"role": "user", "content": content}],
    )


def test_multimodal_input_rejects_text_only_model(monkeypatch):
    monkeypatch.setenv("MINIMAX_PROTOCOL", "openai")
    content = [
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/frame.png"},
        }
    ]

    with pytest.raises(ValueError, match="requires MiniMax-M3"):
        minimax.prompt(content, "MiniMax-M2.7")


def test_anthropic_request_translates_content_and_thinking(monkeypatch):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="A short scene")]
    )
    monkeypatch.setattr(minimax, "_anthropic_client", lambda: client)
    monkeypatch.setenv("MINIMAX_PROTOCOL", "anthropic")
    content = [
        {"type": "text", "text": "Describe this"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/frame.png"},
        },
        {
            "type": "video_url",
            "video_url": {"url": "mm_file://video-id"},
        },
    ]

    assert minimax.prompt(content, "MiniMax-M3:adaptive") == "A short scene"
    client.messages.create.assert_called_once_with(
        model="MiniMax-M3",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://example.com/frame.png",
                        },
                    },
                    {
                        "type": "video",
                        "source": {"type": "url", "url": "mm_file://video-id"},
                    },
                ],
            }
        ],
        thinking={"type": "adaptive"},
    )


def test_list_models_fallback_contains_target_models(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    assert minimax.list_models() == ["MiniMax-M3", "MiniMax-M2.7"]
