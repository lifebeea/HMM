import json
import re
from pathlib import Path

from .viterbi import STATES, tags_to_words, viterbi_decode


class ModelError(ValueError):
    """Raised when an HMM model is malformed or unreadable."""


class HMMSegmenter:
    def __init__(self, pi, A, B, metadata=None):
        self.pi = pi
        self.A = A
        self.B = B
        self.metadata = metadata or {}
        self._validate()

    def _validate(self) -> None:
        if set(self.pi) != set(STATES):
            raise ModelError("initial probability states are incomplete")
        if set(self.A) != set(STATES):
            raise ModelError("transition states are incomplete")
        if set(self.B) != set(STATES):
            raise ModelError("emission states are incomplete")
        for state in STATES:
            if set(self.A[state]) != set(STATES):
                raise ModelError(f"transition row is incomplete: {state}")
            if sum(self.A[state].values()) <= 0:
                raise ModelError(f"transition row has no probability mass: {state}")
        if sum(self.pi.values()) <= 0:
            raise ModelError("initial probabilities have no probability mass")

    def cut(self, sentence: str) -> list[str]:
        if not sentence or not sentence.strip():
            return []
        words: list[str] = []
        for chunk in (part for part in re.split(r"\s+", sentence.strip()) if part):
            chars = list(chunk)
            tags = viterbi_decode(chars, self.pi, self.A, self.B)
            words.extend(tags_to_words(chars, tags))
        return words

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "pi": self.pi,
            "A": self.A,
            "B": self.B,
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "HMMSegmenter":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            return cls(
                pi=payload["pi"],
                A=payload["A"],
                B=payload["B"],
                metadata=payload.get("metadata", {}),
            )
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ModelError(f"cannot load HMM model: {source}") from exc
