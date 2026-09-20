import html
import re


_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_BOUNDARY = re.compile(r"[。！？!?；;]|(?<!\d)\.(?!\d)")
_KEEP = re.compile(
    r"[^一-龥A-Za-z0-9\s"
    r"，。！？；：、,.!?;:()（）【】《》“”‘’'\"\[\]—…·_-]"
)


def clean_text(text: str) -> str:
    """Normalize supported text while removing HTML and noisy symbols."""
    text = html.unescape(text)
    text = _HTML_TAG.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    text = _KEEP.sub("", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries and discard empty segments."""
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
