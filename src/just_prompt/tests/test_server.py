"""Tests for MCP input schemas."""

from just_prompt.server import PromptSchema


def test_prompt_schema_accepts_multimodal_content():
    content = [
        {"type": "text", "text": "Describe this"},
        {
            "type": "video_url",
            "video_url": {"url": "mm_file://video-id", "fps": 1},
        },
    ]

    schema = PromptSchema(
        text=content,
        models_prefixed_by_provider=["minimax:MiniMax-M3:adaptive"],
    )

    assert schema.text == content
