import html
import re


_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_BOUNDARY = re.compile(r"[\u3002\uff01\uff1f!?\uff1b;]|(?<!\d)\.(?!\d)")
_KEEP = re.compile(
    r"[^\u4e00-\u9fa5A-Za-z0-9\s"
    r"\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001,.!?;:()\uff08\uff09\u3010\u3011\u300a\u300b"
    r"\u201c\u201d\u2018\u2019'\"\[\]\u2014\u2026\u00b7_-]"
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
