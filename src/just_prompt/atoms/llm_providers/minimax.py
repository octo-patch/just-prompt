"""MiniMax provider implementation."""

import logging
import os
from typing import Dict, List

import anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_ENDPOINTS: Dict[str, Dict[str, str]] = {
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


def _endpoint_config() -> Dict[str, str]:
    """Return the configured regional endpoint pair."""
    region = os.environ.get("MINIMAX_REGION", "global_en").strip()
    if region not in _ENDPOINTS:
        supported_regions = ", ".join(sorted(_ENDPOINTS))
        raise ValueError(f"MINIMAX_REGION must be one of: {supported_regions}")

    defaults = _ENDPOINTS[region]
    config = {
        "openai_base_url": os.environ.get(
            "MINIMAX_OPENAI_BASE_URL", defaults["openai_base_url"]
        ).rstrip("/"),
        "anthropic_base_url": os.environ.get(
            "MINIMAX_ANTHROPIC_BASE_URL", defaults["anthropic_base_url"]
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
    config = _endpoint_config()
    return OpenAI(api_key=_api_key(), base_url=config["openai_base_url"])


def _anthropic_client() -> anthropic.Anthropic:
    config = _endpoint_config()
    return anthropic.Anthropic(
        api_key=_api_key(), base_url=config["anthropic_base_url"]
    )


def prompt(text: str, model: str) -> str:
    """Send a prompt through the selected MiniMax-compatible protocol."""
    try:
        if _protocol() == "openai":
            response = _openai_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": text}],
            )
            return response.choices[0].message.content

        message = _anthropic_client().messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": text}],
        )
        text_blocks = [block.text for block in message.content if block.type == "text"]
        if not text_blocks:
            raise ValueError("No text content found in response")
        return text_blocks[0]
    except Exception as exc:
        logger.error("Error sending prompt to MiniMax: %s", exc)
        raise ValueError(f"Failed to get response from MiniMax: {exc}") from exc


def list_models() -> List[str]:
    """List available MiniMax models, with a local fallback for offline use."""
    try:
        if _protocol() == "openai":
            response = _openai_client().models.list()
        else:
            response = _anthropic_client().models.list()
        return [model.id for model in response.data]
    except Exception as exc:
        logger.warning("Error listing MiniMax models: %s", exc)
        return list(_FALLBACK_MODELS)
