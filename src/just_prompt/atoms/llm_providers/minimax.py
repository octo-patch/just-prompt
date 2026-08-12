"""MiniMax provider implementation."""

import logging
import os
from typing import Any

import anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

PromptContent = str | list[dict[str, Any]]

_ENDPOINTS: dict[str, dict[str, str]] = {
    "global_en": {
        "openai_base_url": "https://api.minimax.io/v1",
        "anthropic_base_url": "https://api.minimax.io/anthropic",
    },
    "cn_zh": {
        "openai_base_url": "https://api.minimaxi.com/v1",
        "anthropic_base_url": "https://api.minimaxi.com/anthropic",
    },
}
_FALLBACK_MODELS = ["MiniMax-M3", "MiniMax-M2.7"]
_THINKING_MODES = {"adaptive", "disabled"}
_MULTIMODAL_MODEL = "MiniMax-M3"


def _endpoint_config() -> dict[str, str]:
    """Return the configured regional endpoint pair."""
    region = os.environ.get("MINIMAX_REGION", "global_en").strip()
    if region not in _ENDPOINTS:
        supported_regions = ", ".join(sorted(_ENDPOINTS))
        raise ValueError(f"MINIMAX_REGION must be one of: {supported_regions}")

    defaults = _ENDPOINTS[region]
    openai_url = os.environ.get("MINIMAX_OPENAI_BASE_URL", "").strip()
    anthropic_url = os.environ.get("MINIMAX_ANTHROPIC_BASE_URL", "").strip()
    config = {
        "openai_base_url": (openai_url or defaults["openai_base_url"]).rstrip("/"),
        "anthropic_base_url": (
            anthropic_url or defaults["anthropic_base_url"]
        ).rstrip("/"),
    }
    if not config["anthropic_base_url"].endswith("/anthropic"):
        raise ValueError("MINIMAX_ANTHROPIC_BASE_URL must end with /anthropic")
    return config


def _api_key() -> str:
    api_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if not api_key:
        raise ValueError("MINIMAX_API_KEY is not configured")
    return api_key


def _protocol() -> str:
    protocol = os.environ.get("MINIMAX_PROTOCOL", "openai").strip().lower()
    if protocol not in {"openai", "anthropic"}:
        raise ValueError("MINIMAX_PROTOCOL must be either openai or anthropic")
    return protocol


def _openai_client() -> OpenAI:
    return OpenAI(
        api_key=_api_key(),
        base_url=_endpoint_config()["openai_base_url"],
    )


def _anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key=_api_key(),
        base_url=_endpoint_config()["anthropic_base_url"],
    )


def parse_thinking_suffix(model: str) -> tuple[str, str]:
    """Parse an optional adaptive or disabled thinking suffix."""
    if ":" not in model:
        return model, ""

    base_model, suffix = model.rsplit(":", 1)
    thinking_mode = suffix.lower()
    if thinking_mode not in _THINKING_MODES:
        return model, ""
    if base_model != _MULTIMODAL_MODEL:
        raise ValueError("Selectable thinking is only supported by MiniMax-M3")
    return base_model, thinking_mode


def _validate_multimodal_content(
    content: PromptContent, model: str
) -> PromptContent:
    """Validate MiniMax-M3 OpenAI-compatible content parts."""
    if isinstance(content, str):
        return content
    if model != _MULTIMODAL_MODEL:
        raise ValueError("Structured image and video input requires MiniMax-M3")
    if not isinstance(content, list) or not content:
        raise ValueError("Prompt content must be text or a non-empty content list")

    for part in content:
        if not isinstance(part, dict):
            raise TypeError("Each prompt content part must be an object")
        part_type = part.get("type")
        if part_type == "text":
            if not isinstance(part.get("text"), str):
                raise ValueError("Text content parts require a text value")
            continue
        if part_type not in {"image_url", "video_url"}:
            raise ValueError(f"Unsupported MiniMax content part: {part_type}")

        media = part.get(part_type)
        if not isinstance(media, dict) or not isinstance(media.get("url"), str):
            raise TypeError(f"{part_type} content parts require a URL")
        detail = media.get("detail", "default")
        if detail not in {"low", "default", "high"}:
            raise ValueError("Media detail must be low, default, or high")
        if part_type == "video_url" and "fps" in media:
            fps = media["fps"]
            if not isinstance(fps, (int, float)) or not 0.2 <= fps <= 5:
                raise ValueError("Video fps must be between 0.2 and 5")

    return content


def _anthropic_content(content: PromptContent) -> PromptContent:
    """Convert OpenAI-compatible content parts to Anthropic-compatible blocks."""
    if isinstance(content, str):
        return content

    blocks: list[dict[str, Any]] = []
    for part in content:
        part_type = part["type"]
        if part_type == "text":
            blocks.append(part)
            continue

        media = part[part_type]
        blocks.append(
            {
                "type": part_type.removesuffix("_url"),
                "source": {"type": "url", "url": media["url"]},
            }
        )
    return blocks


def prompt(text: PromptContent, model: str) -> str:
    """Send a prompt through the selected MiniMax-compatible protocol."""
    try:
        base_model, thinking_mode = parse_thinking_suffix(model)
        protocol = _protocol()

        if protocol == "openai":
            content = _validate_multimodal_content(text, base_model)
            request: dict[str, Any] = {
                "model": base_model,
                "messages": [{"role": "user", "content": content}],
            }
            if thinking_mode:
                request["extra_body"] = {
                    "thinking": {"type": thinking_mode},
                }
            response = _openai_client().chat.completions.create(**request)
            result = response.choices[0].message.content
            if not isinstance(result, str):
                raise ValueError("No text content found in response")
            return result

        content = _validate_multimodal_content(text, base_model)
        request = {
            "model": base_model,
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": _anthropic_content(content)}
            ],
        }
        if thinking_mode:
            request["thinking"] = {"type": thinking_mode}
        message = _anthropic_client().messages.create(**request)
        text_blocks = [block.text for block in message.content if block.type == "text"]
        if not text_blocks:
            raise ValueError("No text content found in response")
        return text_blocks[0]
    except Exception as exc:
        logger.error("Error sending prompt to MiniMax: %s", exc)
        raise ValueError(f"Failed to get response from MiniMax: {exc}") from exc


def list_models() -> list[str]:
    """List available MiniMax models, with a local fallback for offline use."""
    try:
        if _protocol() == "openai":
            response = _openai_client().models.list()
        else:
            response = _anthropic_client().models.list()
        return [model.id for model in response.data]
    except Exception as exc:  # noqa: BLE001 - preserve the offline model fallback
        logger.warning("Error listing MiniMax models: %s", exc)
        return list(_FALLBACK_MODELS)
