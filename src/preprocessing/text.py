"""Text preprocessing utilities for meme text."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict

import emoji

HASHTAG_PATTERN = re.compile(r"#(\w+)")
PUNCT_PATTERN = re.compile(r"([?!]{2,}|\.{3,}|\"[^\"]+\")")


COMMON_EMOJI_MAP: Dict[str, str] = {
    "😂": ":face_with_tears_of_joy:",
    "🤣": ":rolling_on_the_floor_laughing:",
    "😡": ":pouting_face:",
    "💀": ":skull:",
}


def normalize_text(text: str) -> str:
    """Apply NFKC normalization and trim redundant whitespace."""
    normalized = unicodedata.normalize("NFKC", text or "")
    return " ".join(normalized.split())


def split_hashtag_token(token: str) -> str:
    """Split hashtags using simple camel case + underscore heuristics."""
    token = token.strip("#")
    token = token.replace("_", " ")
    token = re.sub(r"([a-z])([A-Z])", r"\1 \2", token)
    return token


def segment_hashtags(text: str) -> str:
    """Expand hashtags into whitespace-separated words."""

    def _replace(match: re.Match[str]) -> str:
        split = split_hashtag_token(match.group(1))
        return split if split else match.group(0)

    return HASHTAG_PATTERN.sub(_replace, text)


def map_emojis(text: str) -> str:
    """Convert emojis to human-readable aliases for tokenizer compatibility."""
    text = emoji.demojize(text, language="en", delimiters=(":", ":"))
    for raw, alias in COMMON_EMOJI_MAP.items():
        text = text.replace(raw, alias)
    return text


def preserve_sarcasm_markers(text: str) -> list[str]:
    """Extract high-signal sarcasm punctuation markers for optional features."""
    return PUNCT_PATTERN.findall(text)


def preprocess_text(text: str) -> Dict[str, object]:
    """Run complete text preprocessing pipeline."""
    text = normalize_text(text)
    text = segment_hashtags(text)
    text = map_emojis(text)
    markers = preserve_sarcasm_markers(text)
    return {"text": text, "markers": markers}
